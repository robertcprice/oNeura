/**
 * MultiScaleCamera -- Multi-scale camera controller for vtk.js scientific visualization.
 *
 * Monitors camera distance from focal point to determine the current biological
 * scale level, enabling LOD transitions across five powers-of-ten zoom regimes.
 *
 * Scale thresholds (calibrated for web terrarium scene):
 *   Ecosystem  : distance > 5        (whole terrarium view)
 *   Organism   : 1.5 < distance <= 5 (individual plants / flies)
 *   Cellular   : 0.4 < distance <= 1.5 (cell-level detail)
 *   Molecular  : 0.1 < distance <= 0.4 (molecules)
 *   Atomic     : distance <= 0.1     (individual atoms)
 *
 * Architecture: Works with any vtk.js renderer's active camera. Uses
 * camera.onModified() to track changes and emits 'scalechange' events
 * when the user crosses a scale boundary.
 */

const SCALE_THRESHOLDS = {
  organism: 5.0,
  cellular: 1.5,
  molecular: 0.4,
  atomic: 0.1,
};

const SCALE_ORDER = ['atomic', 'molecular', 'cellular', 'organism', 'ecosystem'];

const ZOOM_TARGETS = {
  ecosystem: 10.0,
  organism: 3.0,
  cellular: 0.8,
  molecular: 0.2,
  atomic: 0.05,
};

const SCALE_INFO = {
  ecosystem: { label: 'Ecosystem', range: '1\u201310 m' },
  organism: { label: 'Organism', range: '1 mm\u201310 cm' },
  cellular: { label: 'Cellular', range: '1\u2013100 \u00b5m' },
  molecular: { label: 'Molecular', range: '1 \u00c5\u201310 nm' },
  atomic: { label: 'Atomic', range: '1 pm\u20131 \u00c5' },
};

const LERP_FACTOR = 0.08;
const LERP_EPSILON = 0.001;

/**
 * Compute the Euclidean distance between two 3-element arrays.
 */
function vec3Distance(a, b) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Linearly interpolate between two values.
 */
function lerp(a, b, t) {
  return a + (b - a) * t;
}

/**
 * Classify a camera distance into its scale level name.
 */
function classifyScale(distance) {
  if (distance <= SCALE_THRESHOLDS.atomic) return 'atomic';
  if (distance <= SCALE_THRESHOLDS.molecular) return 'molecular';
  if (distance <= SCALE_THRESHOLDS.cellular) return 'cellular';
  if (distance <= SCALE_THRESHOLDS.organism) return 'organism';
  return 'ecosystem';
}

export class MultiScaleCamera {
  /**
   * @param {vtkRenderer} renderer - The vtk.js renderer whose active camera to monitor.
   * @param {vtkRenderWindow} renderWindow - The vtk.js render window (used to trigger re-renders).
   */
  constructor(renderer, renderWindow) {
    this._renderer = renderer;
    this._renderWindow = renderWindow;
    this._listeners = new Map();
    this._currentScale = 'ecosystem';
    this._currentDistance = 10.0;
    this._animationId = null;
    this._cameraSubscription = null;

    // Default camera state for resetView
    this._defaultPosition = [4, -6, 8];
    this._defaultFocalPoint = [4, 4, 0];
    this._defaultViewUp = [0, 0, 1];

    this._bindCameraListener();
  }

  // ---------------------------------------------------------------------------
  // Public read-only properties
  // ---------------------------------------------------------------------------

  /** Current scale level name: 'ecosystem' | 'organism' | 'cellular' | 'molecular' | 'atomic' */
  get currentScale() {
    return this._currentScale;
  }

  /** Current camera distance from focal point (world units). */
  get currentDistance() {
    return this._currentDistance;
  }

  // ---------------------------------------------------------------------------
  // Event system
  // ---------------------------------------------------------------------------

  /**
   * Register an event listener.
   * @param {string} event - Event name (e.g. 'scalechange').
   * @param {function} fn - Callback receiving event data.
   */
  on(event, fn) {
    if (!this._listeners.has(event)) {
      this._listeners.set(event, []);
    }
    this._listeners.get(event).push(fn);
  }

  /**
   * Remove an event listener.
   * @param {string} event - Event name.
   * @param {function} fn - The callback to remove.
   */
  off(event, fn) {
    const fns = this._listeners.get(event);
    if (!fns) return;
    const idx = fns.indexOf(fn);
    if (idx !== -1) fns.splice(idx, 1);
  }

  /**
   * Emit an event to all registered listeners.
   * @param {string} event - Event name.
   * @param {*} data - Event payload.
   */
  _emit(event, data) {
    const fns = this._listeners.get(event);
    if (fns) {
      for (let i = 0; i < fns.length; i++) {
        fns[i](data);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Camera monitoring
  // ---------------------------------------------------------------------------

  /**
   * Subscribe to the active camera's onModified callback so we detect every
   * orbit, pan, dolly, or programmatic camera change.
   */
  _bindCameraListener() {
    const camera = this._renderer.getActiveCamera();
    if (!camera) return;

    this._cameraSubscription = camera.onModified(() => {
      this._onCameraChanged();
    });

    // Run once to set initial state
    this._onCameraChanged();
  }

  /**
   * Called whenever the camera is modified. Computes distance, classifies scale,
   * and fires 'scalechange' if the level has changed.
   */
  _onCameraChanged() {
    const camera = this._renderer.getActiveCamera();
    if (!camera) return;

    const position = camera.getPosition();
    const focalPoint = camera.getFocalPoint();
    const distance = vec3Distance(position, focalPoint);

    this._currentDistance = distance;

    const newScale = classifyScale(distance);
    if (newScale !== this._currentScale) {
      const from = this._currentScale;
      this._currentScale = newScale;
      this._emit('scalechange', { from, to: newScale, distance });
      this._updateScaleIndicator(newScale);
    }
  }

  // ---------------------------------------------------------------------------
  // Zoom animation
  // ---------------------------------------------------------------------------

  /**
   * Smoothly animate the camera to the canonical distance for a given scale level.
   *
   * The camera moves along its current view direction toward or away from
   * the focal point. Uses requestAnimationFrame with exponential lerp.
   *
   * @param {string} scaleName - Target scale: 'ecosystem' | 'organism' | 'cellular' | 'molecular' | 'atomic'
   */
  zoomTo(scaleName) {
    const targetDistance = ZOOM_TARGETS[scaleName];
    if (targetDistance === undefined) {
      console.warn(`MultiScaleCamera.zoomTo: unknown scale "${scaleName}"`);
      return;
    }

    // Cancel any in-progress animation
    this._cancelAnimation();

    const camera = this._renderer.getActiveCamera();
    if (!camera) return;

    const focalPoint = camera.getFocalPoint();

    const animate = () => {
      const pos = camera.getPosition();
      const currentDist = vec3Distance(pos, focalPoint);

      if (Math.abs(currentDist - targetDistance) < LERP_EPSILON) {
        // Snap to exact target and stop
        this._setCameraDistance(camera, focalPoint, pos, targetDistance);
        this._renderer.resetCameraClippingRange();
        this._renderWindow.render();
        this._animationId = null;
        return;
      }

      const newDist = lerp(currentDist, targetDistance, LERP_FACTOR);
      this._setCameraDistance(camera, focalPoint, pos, newDist);
      this._renderer.resetCameraClippingRange();
      this._renderWindow.render();

      this._animationId = requestAnimationFrame(animate);
    };

    this._animationId = requestAnimationFrame(animate);
  }

  /**
   * Move the camera position so it sits at `distance` from `focalPoint`,
   * preserving the current view direction.
   */
  _setCameraDistance(camera, focalPoint, currentPos, distance) {
    const dx = currentPos[0] - focalPoint[0];
    const dy = currentPos[1] - focalPoint[1];
    const dz = currentPos[2] - focalPoint[2];
    const currentDist = Math.sqrt(dx * dx + dy * dy + dz * dz);

    // Guard against degenerate case where camera sits exactly on focal point
    if (currentDist < 1e-12) {
      camera.setPosition(
        focalPoint[0],
        focalPoint[1],
        focalPoint[2] + distance
      );
      return;
    }

    const scale = distance / currentDist;
    camera.setPosition(
      focalPoint[0] + dx * scale,
      focalPoint[1] + dy * scale,
      focalPoint[2] + dz * scale
    );
  }

  /**
   * Cancel any in-progress zoom animation.
   */
  _cancelAnimation() {
    if (this._animationId !== null) {
      cancelAnimationFrame(this._animationId);
      this._animationId = null;
    }
  }

  // ---------------------------------------------------------------------------
  // Focus and reset
  // ---------------------------------------------------------------------------

  /**
   * Move the camera's focal point to a world position while preserving
   * the current distance and view direction.
   *
   * @param {number[]} worldPos - [x, y, z] target focal point.
   */
  focusOn(worldPos) {
    this._cancelAnimation();

    const camera = this._renderer.getActiveCamera();
    if (!camera) return;

    const oldFocal = camera.getFocalPoint();
    const oldPos = camera.getPosition();

    // Translation offset
    const dx = worldPos[0] - oldFocal[0];
    const dy = worldPos[1] - oldFocal[1];
    const dz = worldPos[2] - oldFocal[2];

    camera.setFocalPoint(worldPos[0], worldPos[1], worldPos[2]);
    camera.setPosition(
      oldPos[0] + dx,
      oldPos[1] + dy,
      oldPos[2] + dz
    );

    this._renderer.resetCameraClippingRange();
    this._renderWindow.render();
  }

  /**
   * Reset to the default ecosystem view: canonical position, focal point,
   * and view-up vector.
   */
  resetView() {
    this._cancelAnimation();

    const camera = this._renderer.getActiveCamera();
    if (!camera) return;

    const fp = this._defaultFocalPoint;
    const pos = this._defaultPosition;
    const targetDist = vec3Distance(pos, fp);

    // Animate from current position to the default ecosystem view
    const startPos = camera.getPosition().slice();
    const startFocal = camera.getFocalPoint().slice();
    const startUp = camera.getViewUp().slice();

    const animate = () => {
      const curPos = camera.getPosition();
      const curFocal = camera.getFocalPoint();
      const curUp = camera.getViewUp();

      const newPos = [
        lerp(curPos[0], pos[0], LERP_FACTOR),
        lerp(curPos[1], pos[1], LERP_FACTOR),
        lerp(curPos[2], pos[2], LERP_FACTOR),
      ];
      const newFocal = [
        lerp(curFocal[0], fp[0], LERP_FACTOR),
        lerp(curFocal[1], fp[1], LERP_FACTOR),
        lerp(curFocal[2], fp[2], LERP_FACTOR),
      ];
      const newUp = [
        lerp(curUp[0], this._defaultViewUp[0], LERP_FACTOR),
        lerp(curUp[1], this._defaultViewUp[1], LERP_FACTOR),
        lerp(curUp[2], this._defaultViewUp[2], LERP_FACTOR),
      ];

      camera.setPosition(...newPos);
      camera.setFocalPoint(...newFocal);
      camera.setViewUp(...newUp);
      this._renderer.resetCameraClippingRange();
      this._renderWindow.render();

      // Check convergence on all three vectors
      const posDelta = vec3Distance(newPos, pos);
      const focalDelta = vec3Distance(newFocal, fp);

      if (posDelta < LERP_EPSILON && focalDelta < LERP_EPSILON) {
        // Snap to exact defaults
        camera.setPosition(...pos);
        camera.setFocalPoint(...fp);
        camera.setViewUp(...this._defaultViewUp);
        this._renderer.resetCameraClippingRange();
        this._renderWindow.render();
        this._animationId = null;
        return;
      }

      this._animationId = requestAnimationFrame(animate);
    };

    this._animationId = requestAnimationFrame(animate);
  }

  // ---------------------------------------------------------------------------
  // DOM indicator
  // ---------------------------------------------------------------------------

  /**
   * Update the #scale-indicator DOM element with the current scale name
   * and human-readable size range.
   *
   * @param {string} scaleName - The active scale level.
   */
  _updateScaleIndicator(scaleName) {
    const el = document.getElementById('scale-indicator');
    if (!el) return;

    const info = SCALE_INFO[scaleName];
    if (!info) {
      el.textContent = scaleName;
      return;
    }

    el.textContent = `${info.label} (${info.range})`;
    el.setAttribute('data-scale', scaleName);
  }

  // ---------------------------------------------------------------------------
  // Lifecycle
  // ---------------------------------------------------------------------------

  /**
   * Clean up subscriptions and cancel animations. Call when the viewer is destroyed.
   */
  destroy() {
    this._cancelAnimation();
    if (this._cameraSubscription) {
      this._cameraSubscription.unsubscribe();
      this._cameraSubscription = null;
    }
    this._listeners.clear();
  }
}
