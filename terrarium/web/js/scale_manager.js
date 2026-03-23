// ---------------------------------------------------------------------------
// ScaleManager — Powers-of-Ten camera-distance-driven scale detection
// ---------------------------------------------------------------------------
// Tracks the current biological scale level based on camera distance to the
// OrbitControls target. Manages dynamic near-plane adjustment, renderer
// show/hide orchestration, and data fetching for sub-ecosystem scales.
//
// Scale thresholds (matching CLI camera.rs):
//   Ecosystem  : distance > 15
//   Organism   : 3 < distance <= 15
//   Cellular   : 0.5 < distance <= 3
//   Molecular  : 0.08 < distance <= 0.5
//   Atomic     : distance <= 0.08

class ScaleManager {
  constructor(threeRenderer) {
    this.renderer = threeRenderer;
    this.currentScale = 'ecosystem';
    this.previousScale = 'ecosystem';
    this.transitionProgress = 1.0;
    this.selectedEntity = null; // { kind, index, worldPos }
    this._lastDistance = 30.0;

    // Distance thresholds — calibrated for web scene (terrain ~12 units across,
    // default camera at distance ~12, max 60). The CLI scene is ~10x smaller.
    this.THRESHOLDS = {
      organism:   5.0,   // close to a single plant
      cellular:   1.5,   // inside plant canopy
      molecular:  0.4,   // see molecules
      atomic:     0.1,   // individual atoms
    };

    // Near-plane per scale
    this.NEAR_PLANE = {
      ecosystem:  0.1,
      organism:   0.05,
      cellular:   0.01,
      molecular:  0.001,
      atomic:     0.0005,
    };

    // Scene environment per scale — ecosystem matches ThreeRenderer defaults exactly
    this.ENVIRONMENTS = {
      ecosystem:  { fogDensity: 0.018, bg: 0x57583d, ambientIntensity: 0.34 },
      organism:   { fogDensity: 0.014, bg: 0x4f5038, ambientIntensity: 0.38 },
      cellular:   { fogDensity: 0.006, bg: 0x1a2030, ambientIntensity: 0.50 },
      molecular:  { fogDensity: 0.0,   bg: 0x0b0f15, ambientIntensity: 0.65 },
      atomic:     { fogDensity: 0.0,   bg: 0x080a10, ambientIntensity: 0.70 },
    };

    // Scale labels for HUD
    this.SCALE_INFO = {
      ecosystem:  { label: 'Ecosystem',  range: '1\u201310 m' },
      organism:   { label: 'Organism',   range: '1 mm\u201310 cm' },
      cellular:   { label: 'Cellular',   range: '1\u2013100 \u00b5m' },
      molecular:  { label: 'Molecular',  range: '1 \u00c5\u201310 nm' },
      atomic:     { label: 'Atomic',     range: '1 pm\u20131 \u00c5' },
    };

    // Keyboard scale-jump distances (midpoint of each scale range)
    this.JUMP_DISTANCES = {
      ecosystem:  12.0,
      organism:   3.0,
      cellular:   0.8,
      molecular:  0.2,
      atomic:     0.06,
    };

    // Cached molecular/cellular/organism data
    this._cachedData = {};
    this._fetchingScale = null;

    // Ecosystem opacity (for crossfade)
    this.ecosystemOpacity = 1.0;
    this._targetEcosystemOpacity = 1.0;

    // Performance: track when environment has settled (no more per-frame work)
    this._lastAppliedOpacity = 1.0;
    this._envSettled = true;
    this._bgTarget = null;
  }

  /** Classify camera distance into a scale level string. */
  scaleFromDistance(distance) {
    if (distance <= this.THRESHOLDS.atomic)    return 'atomic';
    if (distance <= this.THRESHOLDS.molecular) return 'molecular';
    if (distance <= this.THRESHOLDS.cellular)  return 'cellular';
    if (distance <= this.THRESHOLDS.organism)  return 'organism';
    return 'ecosystem';
  }

  /** Call every frame before controls.update(). */
  update() {
    const camera = this.renderer.camera;
    const target = this.renderer.controls.target;
    const distance = camera.position.distanceTo(target);
    this._lastDistance = distance;

    // Only engage sub-ecosystem scales if there's a selected entity to drill into.
    // Without a selection, everything stays at ecosystem regardless of camera distance.
    const newScale = this.selectedEntity ? this.scaleFromDistance(distance) : 'ecosystem';

    // Dynamic near-plane (always active for z-fighting prevention)
    const nearForDist = this.NEAR_PLANE[this.scaleFromDistance(distance)] || 0.1;
    if (Math.abs(camera.near - nearForDist) > 0.00001) {
      camera.near = nearForDist;
      camera.updateProjectionMatrix();
    }

    // Detect scale transition
    if (newScale !== this.currentScale) {
      this.previousScale = this.currentScale;
      this.currentScale = newScale;
      this.transitionProgress = 0.0;
      this._onScaleChanged(this.previousScale, this.currentScale);
    }

    // Advance transition
    if (this.transitionProgress < 1.0) {
      this.transitionProgress = Math.min(1.0, this.transitionProgress + 0.033);
    }

    // Compute ecosystem opacity target — only fade when drilling into a selection
    if (!this.selectedEntity) {
      this._targetEcosystemOpacity = 1.0;
    } else {
      switch (this.currentScale) {
        case 'ecosystem': this._targetEcosystemOpacity = 1.0; break;
        case 'organism':  this._targetEcosystemOpacity = 0.4; break;
        default:          this._targetEcosystemOpacity = 0.0; break;
      }
    }
    // Lerp opacity
    this.ecosystemOpacity += (this._targetEcosystemOpacity - this.ecosystemOpacity) * 0.08;
    if (Math.abs(this.ecosystemOpacity - this._targetEcosystemOpacity) < 0.005) {
      this.ecosystemOpacity = this._targetEcosystemOpacity;
    }

    // Apply environment
    this._applyEnvironment();
  }

  /** Store which entity is selected for drill-down. */
  setSelectedEntity(ref) {
    if (!ref) {
      this.selectedEntity = null;
      this._cachedData = {};
      return;
    }
    // If selection changed, clear cache
    if (!this.selectedEntity ||
        this.selectedEntity.kind !== ref.kind ||
        this.selectedEntity.index !== ref.index) {
      this._cachedData = {};
    }
    this.selectedEntity = ref;
  }

  /** Animate camera to a target scale level. */
  jumpToScale(scaleName) {
    const dist = this.JUMP_DISTANCES[scaleName];
    if (dist == null) return;
    const target = this.renderer.controls.target;
    const camera = this.renderer.camera;
    const dir = camera.position.clone().sub(target).normalize();
    const dest = target.clone().add(dir.multiplyScalar(dist));
    this._animateCamera(dest, 40);
  }

  /** Smooth camera position animation over N frames. */
  _animateCamera(dest, frames) {
    const camera = this.renderer.camera;
    const start = camera.position.clone();
    let frame = 0;
    const step = () => {
      frame++;
      const t = Math.min(1.0, frame / frames);
      const ease = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; // ease in-out
      camera.position.lerpVectors(start, dest, ease);
      if (frame < frames) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  /** Apply scene environment settings for current scale.
   *  PERFORMANCE: Only does work when opacity or scale actually changes.
   *  At ecosystem scale (the common case), this is a no-op after settling. */
  _applyEnvironment() {
    // Skip entirely if at ecosystem with full opacity (common case)
    if (this.currentScale === 'ecosystem' && this.ecosystemOpacity >= 0.99 && this._envSettled) {
      return;
    }

    const env = this.ENVIRONMENTS[this.currentScale];
    if (!env) return;
    const scene = this.renderer.scene;

    // Fog — only adjust if not settled
    if (scene.fog) {
      const delta = env.fogDensity - scene.fog.density;
      if (Math.abs(delta) > 0.0001) {
        scene.fog.density += delta * 0.06;
      }
    }

    // Background color — reuse cached color object
    if (scene.background) {
      if (!this._bgTarget) this._bgTarget = new THREE.Color(env.bg);
      else this._bgTarget.set(env.bg);
      scene.background.lerp(this._bgTarget, 0.04);
    }

    // Ambient light
    if (this.renderer.ambientLight) {
      const a = this.renderer.ambientLight;
      const delta = env.ambientIntensity - a.intensity;
      if (Math.abs(delta) > 0.001) {
        a.intensity += delta * 0.06;
      }
    }

    // Ecosystem opacity crossfade — only when actually transitioning
    const opacityChanged = Math.abs(this.ecosystemOpacity - this._lastAppliedOpacity) > 0.005;
    if (opacityChanged) {
      this._lastAppliedOpacity = this.ecosystemOpacity;
      const r = this.renderer;
      if (r.shaderTerrain && r.shaderTerrain.setOpacity) {
        r.shaderTerrain.setOpacity(this.ecosystemOpacity);
      }
      if (r.plantRenderer && r.plantRenderer.setOpacity) {
        r.plantRenderer.setOpacity(this.ecosystemOpacity);
      }
    }

    // Check if we've settled (no more changes needed)
    this._envSettled = this.currentScale === 'ecosystem' && this.ecosystemOpacity >= 0.99;
  }

  /** Called when scale level changes. Orchestrates renderer visibility + data fetch. */
  _onScaleChanged(fromScale, toScale) {
    const r = this.renderer;

    // Show/hide scale-specific renderers
    const isSoil = this.selectedEntity?.kind === 'soil';
    if (r.molecularRenderer) {
      const show = toScale === 'molecular' || toScale === 'atomic';
      show ? r.molecularRenderer.show() : r.molecularRenderer.hide();
    }
    if (r.cellularRenderer) {
      // For soil: cellular renderer handles both organism (soil grains) and cellular (aggregate) scales
      const show = toScale === 'cellular' || (isSoil && toScale === 'organism');
      show ? r.cellularRenderer.show() : r.cellularRenderer.hide();
    }
    if (r.organismRenderer) {
      const show = toScale === 'organism' && !isSoil;
      show ? r.organismRenderer.show() : r.organismRenderer.hide();
    }

    // Fetch data for the new scale if we have a selected entity
    if (this.selectedEntity) {
      this._fetchScaleData(toScale);
    }

    // Dispatch custom event so the inspect panel can sync
    window.dispatchEvent(new CustomEvent('scaleChanged', {
      detail: { from: fromScale, to: toScale },
    }));
  }

  /** Fetch scale-specific data from the inspect API. */
  async _fetchScaleData(scale) {
    if (!this.selectedEntity) return;
    if (scale === 'ecosystem') return;

    // Check cache
    const cacheKey = `${this.selectedEntity.kind}_${this.selectedEntity.index}_${scale}`;
    if (this._cachedData[cacheKey]) {
      this._applyScaleData(scale, this._cachedData[cacheKey]);
      return;
    }

    if (this._fetchingScale === scale) return;
    this._fetchingScale = scale;

    try {
      const entity = this.selectedEntity;
      // Soil at organism/cellular zoom: fetch ecosystem-scale data which has composition[]
      const fetchScale = (entity.kind === 'soil' && (scale === 'organism' || scale === 'cellular'))
        ? 'ecosystem' : scale;
      let url = `/api/inspect?kind=${entity.kind}&scale=${fetchScale}`;
      if (entity.index != null) url += `&index=${entity.index}`;
      if (entity.x != null) url += `&x=${entity.x}`;
      if (entity.y != null) url += `&y=${entity.y}`;

      const resp = await fetch(url);
      if (!resp.ok) return;
      const data = await resp.json();
      this._cachedData[cacheKey] = data;
      // Only apply if we're still on this scale
      if (this.currentScale === scale) {
        this._applyScaleData(scale, data);
      }
    } catch (e) {
      console.warn('ScaleManager: fetch failed for', scale, e);
    } finally {
      this._fetchingScale = null;
    }
  }

  /** Apply fetched data to the appropriate renderer. */
  _applyScaleData(scale, data) {
    const r = this.renderer;
    const pos = this.selectedEntity?.worldPos;
    if (!pos) return;

    const isSoil = this.selectedEntity.kind === 'soil';

    switch (scale) {
      case 'organism':
        if (isSoil && r.cellularRenderer) {
          // Soil at organism zoom: show soil grains
          // Pass the backend-computed soil RGB if available from terrain visuals
          const soilRGB = this.selectedEntity.soilRGB || null;
          r.cellularRenderer.setSoilParticleField(data, pos, 'organism', soilRGB);
        } else if (r.organismRenderer && data.organism_components) {
          r.organismRenderer.setOrganismComponents(
            data.organism_components,
            this.selectedEntity.kind,
            pos
          );
        }
        break;
      case 'cellular':
        if (isSoil && r.cellularRenderer) {
          // Soil at cellular zoom: inside a soil aggregate
          const soilRGB2 = this.selectedEntity.soilRGB || null;
          r.cellularRenderer.setSoilParticleField(data, pos, 'cellular', soilRGB2);
        } else if (r.cellularRenderer && data.cellular_detail) {
          r.cellularRenderer.setCellularDetail(data.cellular_detail, pos);
        }
        break;
      case 'molecular':
      case 'atomic':
        if (r.molecularRenderer && data.molecular_detail) {
          r.molecularRenderer.setMolecularDetail(data.molecular_detail, pos);
        }
        break;
    }
  }

  /** Get HUD display info for current scale. */
  getHudInfo() {
    const info = this.SCALE_INFO[this.currentScale] || this.SCALE_INFO.ecosystem;
    return {
      label: info.label,
      range: info.range,
      distance: this._lastDistance.toFixed(2),
      hasSelection: !!this.selectedEntity,
    };
  }
}
