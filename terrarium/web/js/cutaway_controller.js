// ---------------------------------------------------------------------------
// CutawayController — Soil cross-section clipping plane for terrain cutaway
// ---------------------------------------------------------------------------
// Uses Three.js's native renderer.clippingPlanes to slice through the terrain,
// revealing geological layers (already colored by the terrain body shader),
// root positions, and chemistry gradients.
//
// Three.js clipping planes work by discarding fragments on the negative side
// of a plane equation (dot(normal, position) + constant < 0). This is handled
// in the GPU at the fragment shader level — zero geometry modification needed.
//
// When the clipping plane is active, all scene materials automatically clip.
// The cut face is visible because the terrain body mesh (which shows geological
// layers: bedrock -> subsoil -> topsoil) already renders with DoubleSide.
//
// Architecture:
//   - Creates a THREE.Plane as the clipping surface
//   - Toggles renderer.clippingPlanes and renderer.localClippingEnabled
//   - Manages material.side = DoubleSide on terrain meshes for visible cut faces
//   - Optionally renders a semi-transparent "cut face" mesh at the slice plane
//     showing soil layer colors, root positions, and chemistry data
//   - Animatable via setCutPosition() with optional lerp
//
// Usage:
//   const cutaway = new CutawayController(renderer3d);
//   cutaway.enable();                     // start cutting
//   cutaway.setCutPosition(0.5);          // cut at 50% through terrain (east-west)
//   cutaway.setAxis('z');                 // switch to north-south cut
//   cutaway.animateTo(0.8, 1.0);         // animate cut to 80% over 1 second
//   cutaway.disable();                    // remove clipping plane
//   // In render loop:
//   cutaway.update(dt);

class CutawayController {
  /**
   * @param {ThreeRenderer} threeRenderer - The app's ThreeRenderer instance
   *   (exposes .renderer, .scene, .shaderTerrain, .plantRenderer, etc.)
   * @param {Object} [options]
   * @param {string} [options.axis='x'] - Initial cut axis: 'x' (east-west) or 'z' (north-south)
   * @param {number} [options.position=0.5] - Initial cut position [0, 1] along the axis
   * @param {boolean} [options.showCutFace=true] - Render a visible plane at the cut surface
   * @param {number} [options.cutFaceOpacity=0.3] - Opacity of the cut face indicator
   */
  constructor(threeRenderer, options) {
    this._threeRenderer = threeRenderer;
    this._renderer = threeRenderer.renderer;
    this._scene = threeRenderer.scene;

    const opts = options || {};
    this._axis = opts.axis || 'x';
    this._position = opts.position !== undefined ? opts.position : 0.5;
    this._showCutFace = opts.showCutFace !== undefined ? opts.showCutFace : true;
    this._cutFaceOpacity = opts.cutFaceOpacity !== undefined ? opts.cutFaceOpacity : 0.3;

    this._enabled = false;
    this._animating = false;
    this._animTarget = 0.5;
    this._animSpeed = 0;

    // Terrain dimensions (updated when terrain data arrives)
    this._gridW = 0;
    this._gridH = 0;

    // The clipping plane: normal points in the positive axis direction.
    // Fragments with dot(normal, position) + constant < 0 are discarded.
    // To cut at x = cutX: normal = (1,0,0), constant = -cutX
    // This keeps everything with x > cutX and discards x < cutX.
    // For a "reveal left side" cut: normal = (-1,0,0), constant = cutX
    this._plane = new THREE.Plane(new THREE.Vector3(-1, 0, 0), 0);
    this._updatePlaneFromPosition();

    // Materials whose .side we modified (so we can restore on disable)
    this._modifiedMaterials = [];

    // Cut face indicator mesh (a flat plane at the cut location showing
    // what the terrain looks like at that slice — soil layers, tint)
    this._cutFaceMesh = null;
    this._cutFaceMaterial = null;
    if (this._showCutFace) {
      this._createCutFaceIndicator();
    }

    // Root visualization group (rendered on the cut face)
    this._rootGroup = new THREE.Group();
    this._rootGroup.visible = false;
    this._rootGroup.name = 'cutawayRoots';
    this._scene.add(this._rootGroup);
  }

  // --- Public API ---

  /** Enable the cutaway clipping plane. */
  enable() {
    if (this._enabled) return;
    this._enabled = true;

    // Detect terrain dimensions from the shader terrain renderer
    const st = this._threeRenderer.shaderTerrain;
    if (st) {
      this._gridW = st.gridW || 0;
      this._gridH = st.gridH || 0;
    }

    // Use local (per-material) clipping instead of global renderer clipping.
    // This lets the cut face indicator mesh remain unclipped — it has no
    // clippingPlanes set on its material, so it renders fully visible at the
    // slice location while terrain/plant materials get clipped.
    this._renderer.localClippingEnabled = true;

    // Apply the clipping plane + DoubleSide to all terrain/plant materials
    this._setDoubleSided(true);

    // Show cut face indicator
    if (this._cutFaceMesh) {
      this._cutFaceMesh.visible = true;
    }
    this._rootGroup.visible = true;

    this._updatePlaneFromPosition();
    this._updateCutFaceTransform();
  }

  /** Disable the cutaway — remove clipping plane, restore materials. */
  disable() {
    if (!this._enabled) return;
    this._enabled = false;
    this._animating = false;

    // Remove per-material clipping planes (handled in _setDoubleSided(false))

    // Restore original material sides
    this._setDoubleSided(false);

    // Hide cut face indicator
    if (this._cutFaceMesh) {
      this._cutFaceMesh.visible = false;
    }
    this._rootGroup.visible = false;
  }

  /** Check if cutaway is active. */
  get enabled() {
    return this._enabled;
  }

  /** Toggle cutaway on/off. Returns the new state. */
  toggle() {
    if (this._enabled) {
      this.disable();
    } else {
      this.enable();
    }
    return this._enabled;
  }

  /**
   * Set which axis to cut along.
   * @param {'x'|'z'} axis - 'x' for east-west cut, 'z' for north-south cut
   */
  setAxis(axis) {
    if (axis !== 'x' && axis !== 'z') return;
    this._axis = axis;
    this._updatePlaneFromPosition();
    this._updateCutFaceTransform();
  }

  /** Get the current cut axis. */
  get axis() {
    return this._axis;
  }

  /**
   * Set the cut position along the current axis.
   * @param {number} t - Normalized position [0, 1] where 0 is the start and 1 is the end
   */
  setCutPosition(t) {
    this._position = Math.max(0, Math.min(1, t));
    this._updatePlaneFromPosition();
    this._updateCutFaceTransform();
  }

  /** Get the current normalized cut position [0, 1]. */
  get position() {
    return this._position;
  }

  /**
   * Animate the cut position to a target value.
   * @param {number} targetT - Target normalized position [0, 1]
   * @param {number} duration - Animation duration in seconds
   */
  animateTo(targetT, duration) {
    this._animTarget = Math.max(0, Math.min(1, targetT));
    this._animSpeed = duration > 0 ? 1.0 / duration : 100;
    this._animating = true;
  }

  /**
   * Call every frame to advance animation.
   * @param {number} dt - Frame delta time in seconds
   */
  update(dt) {
    if (!this._enabled || !this._animating) return;

    const diff = this._animTarget - this._position;
    if (Math.abs(diff) < 0.001) {
      this._position = this._animTarget;
      this._animating = false;
    } else {
      // Ease-out: slow down as we approach target
      const step = diff * Math.min(1.0, this._animSpeed * dt * 3.0);
      this._position += step;
    }
    this._position = Math.max(0, Math.min(1, this._position));
    this._updatePlaneFromPosition();
    this._updateCutFaceTransform();
  }

  /**
   * Update root positions on the cut face from plant data.
   * @param {Array} plants - Array of { x, y, root_depth, root_spread, color } from backend
   */
  setRootData(plants) {
    // Clear existing roots
    while (this._rootGroup.children.length) {
      const child = this._rootGroup.children[0];
      this._rootGroup.remove(child);
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    }

    if (!plants || !plants.length || !this._enabled) return;

    const halfW = this._gridW / 2;
    const halfH = this._gridH / 2;
    const st = this._threeRenderer.shaderTerrain;

    for (const plant of plants) {
      // Determine if this plant is near the cut plane
      const worldX = plant.x - halfW + 0.5;
      const worldZ = plant.y - halfH + 0.5;
      const cutWorldPos = this._getWorldCutPosition();

      let distFromCut;
      if (this._axis === 'x') {
        distFromCut = Math.abs(worldX - cutWorldPos);
      } else {
        distFromCut = Math.abs(worldZ - cutWorldPos);
      }

      // Only show roots within ~2 cells of the cut plane
      const rootSpread = plant.root_spread || 1.5;
      if (distFromCut > rootSpread) continue;

      // Get surface height at this plant's position
      const surfaceY = st ? st.sampleSurfaceHeight(worldX, worldZ) : 0.5;

      // Root depth (from backend data, in world units)
      const rootDepth = plant.root_depth || 1.5;

      // Draw root as a tapered line going down from the surface
      const rootColor = plant.color || [0.55, 0.40, 0.25];
      const rootMaterial = new THREE.LineBasicMaterial({
        color: new THREE.Color(rootColor[0], rootColor[1], rootColor[2]),
        linewidth: 2,
      });

      // Root geometry: a branching structure going downward
      // Main taproot + 2-3 lateral roots
      const rootPoints = [];

      // Main taproot (straight down with slight wobble)
      const segments = 8;
      for (let s = 0; s <= segments; s++) {
        const t = s / segments;
        const wobbleX = Math.sin(t * 4.0 + plant.x) * 0.08 * (1 - t);
        const wobbleZ = Math.cos(t * 3.5 + plant.y) * 0.06 * (1 - t);
        rootPoints.push(new THREE.Vector3(
          worldX + wobbleX,
          surfaceY - t * rootDepth,
          worldZ + wobbleZ
        ));
      }

      const rootGeo = new THREE.BufferGeometry().setFromPoints(rootPoints);
      const rootLine = new THREE.Line(rootGeo, rootMaterial);
      this._rootGroup.add(rootLine);

      // Lateral roots (2 per plant, branching at 30-60% depth)
      for (let lr = 0; lr < 2; lr++) {
        const branchDepth = 0.3 + lr * 0.25;
        const branchAngle = (lr * 2 - 1) * (0.4 + Math.sin(plant.x + lr) * 0.2);
        const lateralPoints = [];
        const lateralSegments = 5;

        for (let s = 0; s <= lateralSegments; s++) {
          const t = s / lateralSegments;
          const lateralSpread = rootSpread * 0.5 * t;
          lateralPoints.push(new THREE.Vector3(
            worldX + lateralSpread * Math.cos(branchAngle),
            surfaceY - branchDepth * rootDepth - t * rootDepth * 0.3,
            worldZ + lateralSpread * Math.sin(branchAngle)
          ));
        }

        const lateralGeo = new THREE.BufferGeometry().setFromPoints(lateralPoints);
        const lateralLine = new THREE.Line(lateralGeo, rootMaterial);
        this._rootGroup.add(lateralLine);
      }
    }
  }

  /**
   * Update chemistry gradient visualization on the cut face.
   * @param {Object} chemistry - { type: 'nitrogen'|'moisture'|'ph', values: Float32Array }
   *   values is a 1D array of the gradient from surface to deep soil (top to bottom)
   */
  setChemistryGradient(chemistry) {
    if (!this._cutFaceMaterial || !this._cutFaceMaterial.uniforms) return;
    // The cut face shader can be extended to display chemistry overlays.
    // For now, we tint the cut face based on a simple gradient type.
    if (chemistry && chemistry.type) {
      let tint;
      switch (chemistry.type) {
        case 'nitrogen':  tint = new THREE.Color(0.2, 0.6, 0.3); break;
        case 'moisture':  tint = new THREE.Color(0.2, 0.4, 0.8); break;
        case 'ph':        tint = new THREE.Color(0.7, 0.3, 0.5); break;
        default:          tint = new THREE.Color(1, 1, 1); break;
      }
      this._cutFaceMaterial.uniforms.uChemistryTint.value.copy(tint);
      this._cutFaceMaterial.uniforms.uShowChemistry.value = 1.0;
    } else {
      this._cutFaceMaterial.uniforms.uShowChemistry.value = 0.0;
    }
  }

  /** Clean up GPU resources. */
  dispose() {
    this.disable();
    if (this._cutFaceMesh) {
      this._scene.remove(this._cutFaceMesh);
      if (this._cutFaceMesh.geometry) this._cutFaceMesh.geometry.dispose();
      if (this._cutFaceMaterial) this._cutFaceMaterial.dispose();
    }
    while (this._rootGroup.children.length) {
      const child = this._rootGroup.children[0];
      this._rootGroup.remove(child);
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    }
    this._scene.remove(this._rootGroup);
  }

  // --- Internal methods ---

  /** Convert normalized position [0,1] to world-space and update the THREE.Plane. */
  _updatePlaneFromPosition() {
    const halfW = this._gridW / 2 || 6;
    const halfH = this._gridH / 2 || 6;

    let worldPos;
    if (this._axis === 'x') {
      // Cut along X axis: plane normal = (-1, 0, 0), constant = cutX
      // This discards everything with x < cutX (reveals the right side)
      worldPos = -halfW + this._position * this._gridW;
      this._plane.normal.set(-1, 0, 0);
      this._plane.constant = worldPos;
    } else {
      // Cut along Z axis: plane normal = (0, 0, -1), constant = cutZ
      worldPos = -halfH + this._position * this._gridH;
      this._plane.normal.set(0, 0, -1);
      this._plane.constant = worldPos;
    }
  }

  /** Get the current world-space position of the cut. */
  _getWorldCutPosition() {
    if (this._axis === 'x') {
      const halfW = this._gridW / 2 || 6;
      return -halfW + this._position * this._gridW;
    } else {
      const halfH = this._gridH / 2 || 6;
      return -halfH + this._position * this._gridH;
    }
  }

  /**
   * Set DoubleSide + per-material clipping planes on terrain/plant materials.
   * Using local (per-material) clipping instead of global renderer.clippingPlanes
   * so the cut face indicator mesh remains unclipped — only terrain and plant
   * materials have the clipping plane applied.
   */
  _setDoubleSided(enable) {
    const st = this._threeRenderer.shaderTerrain;
    if (!st) return;

    const meshes = [st.surfaceMesh, st.bodyMesh, st.grassMesh, st.waterMesh];
    if (enable) {
      this._modifiedMaterials = [];
      for (const mesh of meshes) {
        if (mesh && mesh.material) {
          this._modifiedMaterials.push({
            material: mesh.material,
            originalSide: mesh.material.side,
            originalClipping: mesh.material.clippingPlanes,
          });
          mesh.material.side = THREE.DoubleSide;
          mesh.material.clippingPlanes = [this._plane];
        }
      }
      // Plant renderer materials (per-plant entries with trunkMat / canopyMat)
      const pr = this._threeRenderer.plantRenderer;
      if (pr && pr.plants) {
        for (const [, entry] of pr.plants) {
          for (const mat of [entry.trunkMat, entry.canopyMat]) {
            if (!mat) continue;
            this._modifiedMaterials.push({
              material: mat,
              originalSide: mat.side,
              originalClipping: mat.clippingPlanes,
            });
            mat.side = THREE.DoubleSide;
            mat.clippingPlanes = [this._plane];
          }
        }
      }
    } else {
      // Restore original side + clipping settings
      for (const entry of this._modifiedMaterials) {
        entry.material.side = entry.originalSide;
        entry.material.clippingPlanes = entry.originalClipping || null;
      }
      this._modifiedMaterials = [];
    }
  }

  /** Create the cut face indicator mesh: a vertical plane that shows soil layers. */
  _createCutFaceIndicator() {
    // The cut face is a tall, thin plane positioned at the cut location.
    // Its shader samples the terrain heightmap and soil color textures to
    // show what the soil looks like at this particular slice — matching
    // the geological layer colors from the terrain body shader.
    const geo = new THREE.PlaneGeometry(1, 1, 1, 32);

    this._cutFaceMaterial = new THREE.ShaderMaterial({
      vertexShader: CutawayController.CUT_FACE_VERT,
      fragmentShader: CutawayController.CUT_FACE_FRAG,
      uniforms: {
        uSurfaceY: { value: 2.0 },
        uFloorY: { value: -4.0 },
        uWidth: { value: 12.0 },
        uChemistryTint: { value: new THREE.Color(1, 1, 1) },
        uShowChemistry: { value: 0.0 },
        uOpacity: { value: this._cutFaceOpacity },
      },
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false,
      // No clippingPlanes set — this material is NOT clipped because we use
      // localClippingEnabled (per-material) instead of global renderer clipping.
      // Only materials with an explicit clippingPlanes array get clipped.
    });

    this._cutFaceMesh = new THREE.Mesh(geo, this._cutFaceMaterial);
    this._cutFaceMesh.visible = false;
    this._cutFaceMesh.renderOrder = 2; // render after terrain + water
    this._cutFaceMesh.name = 'cutawayFace';
    this._scene.add(this._cutFaceMesh);
  }

  /** Update the cut face mesh position and orientation to match the plane. */
  _updateCutFaceTransform() {
    if (!this._cutFaceMesh) return;

    const halfW = this._gridW / 2 || 6;
    const halfH = this._gridH / 2 || 6;
    const surfaceYApprox = 2.5; // approximate max terrain height
    const floorY = -4.0;
    const height = surfaceYApprox - floorY;
    const cutWorld = this._getWorldCutPosition();

    if (this._axis === 'x') {
      // Vertical plane perpendicular to X axis, spanning full Z width
      this._cutFaceMesh.scale.set(this._gridH || 12, height, 1);
      this._cutFaceMesh.position.set(cutWorld, floorY + height / 2, 0);
      this._cutFaceMesh.rotation.set(0, Math.PI / 2, 0);
    } else {
      // Vertical plane perpendicular to Z axis, spanning full X width
      this._cutFaceMesh.scale.set(this._gridW || 12, height, 1);
      this._cutFaceMesh.position.set(0, floorY + height / 2, cutWorld);
      this._cutFaceMesh.rotation.set(0, 0, 0);
    }

    // Update shader uniforms
    const mat = this._cutFaceMaterial;
    if (mat && mat.uniforms) {
      mat.uniforms.uSurfaceY.value = surfaceYApprox;
      mat.uniforms.uFloorY.value = floorY;
      mat.uniforms.uWidth.value = this._axis === 'x' ? (this._gridH || 12) : (this._gridW || 12);
    }
  }
}

// ---------------------------------------------------------------------------
// Cut face shaders — visualize soil layers on the slice plane
// ---------------------------------------------------------------------------
// These match the geological layer color logic from the terrain body shader
// (bedrock -> subsoil -> topsoil) so the cut face looks physically consistent
// with what you see on the terrain edge walls.

CutawayController.CUT_FACE_VERT = `
  varying vec2 vUv;
  varying vec3 vWorldPos;
  void main() {
    vUv = uv;
    vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

CutawayController.CUT_FACE_FRAG = `
  precision highp float;

  uniform float uSurfaceY;
  uniform float uFloorY;
  uniform float uWidth;
  uniform vec3 uChemistryTint;
  uniform float uShowChemistry;
  uniform float uOpacity;

  varying vec2 vUv;
  varying vec3 vWorldPos;

  void main() {
    // Normalized depth: 0 = floor (bottom), 1 = surface (top)
    float depth = clamp((vWorldPos.y - uFloorY) / (uSurfaceY - uFloorY), 0.0, 1.0);

    // Geological layer colors — emergent from mineral CPK averages.
    // These match the terrain body shader exactly:
    //   0.00 - 0.30: Bedrock (dark mineral silicate)
    //   0.30 - 0.65: Subsoil (transitional)
    //   0.65 - 1.00: Topsoil (organic-rich surface)
    vec3 bedrockColor = vec3(0.28, 0.24, 0.20);
    vec3 subsoilColor = vec3(0.42, 0.36, 0.28);
    vec3 surfaceColor = vec3(0.52, 0.44, 0.32);

    vec3 color;
    if (depth < 0.30) {
      float blend = smoothstep(0.0, 0.10, depth);
      color = mix(bedrockColor * 0.85, bedrockColor, blend);
    } else if (depth < 0.65) {
      float blend = smoothstep(0.30, 0.45, depth);
      color = mix(bedrockColor, subsoilColor, blend);
    } else {
      float blend = smoothstep(0.65, 0.82, depth);
      color = mix(subsoilColor, surfaceColor, blend);
    }

    // Sedimentary banding (subtle horizontal stripes for visual texture)
    float band = sin(vWorldPos.y * 18.0) * 0.025 + sin(vWorldPos.y * 7.3) * 0.015;
    color += band;

    // Chemistry gradient overlay
    if (uShowChemistry > 0.5) {
      // Gradient intensity: strongest near surface, fading at depth
      float chemIntensity = depth * depth * 0.4;
      color = mix(color, uChemistryTint, chemIntensity);
    }

    // Edge fade: soften the edges of the cut face
    float edgeFade = smoothstep(0.0, 0.03, vUv.x) * smoothstep(1.0, 0.97, vUv.x);

    gl_FragColor = vec4(color, uOpacity * edgeFade);
  }
`;
