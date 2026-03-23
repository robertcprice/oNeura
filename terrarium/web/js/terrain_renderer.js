/**
 * Shader-based terrain renderer for oNeura Terrarium.
 *
 * All visual properties emerge from the simulation:
 * - Soil color: Beer-Lambert molecular optics (backend)
 * - Moisture darkening: Lobell & Asner 2002 albedo reduction
 * - Water color: Pope & Fry 1997 spectral absorption
 * - Grass density: chlorophyll/organic content from soil color
 * - Wind animation: atmospheric simulation wind fields
 * - Geological layers: emergent from surface color toward mineral CPK base
 *
 * Zero hardcoded colors. 4 draw calls total (surface + body + grass + water).
 */

// ─── DataTexture Manager ──────────────────────────────────────────────

class TerrainDataTextures {
  constructor(gridW, gridH) {
    this.gridW = gridW;
    this.gridH = gridH;
    const size = gridW * gridH;

    // Texture 0: Heightmap (relief, soilTexture, moisture, waterMask)
    this.heightmapData = new Float32Array(size * 4);
    this.heightmapTex = new THREE.DataTexture(
      this.heightmapData, gridW, gridH,
      THREE.RGBAFormat, THREE.FloatType
    );
    this.heightmapTex.minFilter = THREE.LinearFilter;
    this.heightmapTex.magFilter = THREE.LinearFilter;
    this.heightmapTex.wrapS = THREE.ClampToEdgeWrapping;
    this.heightmapTex.wrapT = THREE.ClampToEdgeWrapping;
    this.heightmapTex.needsUpdate = true;

    // Texture 1: Soil color (emergent RGB from backend + canopy alpha)
    // Initialize with warm brown so terrain is visible before backend data arrives.
    this.soilColorData = new Float32Array(size * 4);
    for (let i = 0; i < size; i++) {
      this.soilColorData[i * 4 + 0] = 0.52;
      this.soilColorData[i * 4 + 1] = 0.42;
      this.soilColorData[i * 4 + 2] = 0.30;
      this.soilColorData[i * 4 + 3] = 0.0;
    }
    this.soilColorTex = new THREE.DataTexture(
      this.soilColorData, gridW, gridH,
      THREE.RGBAFormat, THREE.FloatType
    );
    this.soilColorTex.minFilter = THREE.LinearFilter;
    this.soilColorTex.magFilter = THREE.LinearFilter;
    this.soilColorTex.wrapS = THREE.ClampToEdgeWrapping;
    this.soilColorTex.wrapT = THREE.ClampToEdgeWrapping;
    this.soilColorTex.needsUpdate = true;

    // Texture 2: Atmosphere (temperature, humidity, windX, windY)
    this.atmosphereData = new Float32Array(size * 4);
    this.atmosphereTex = new THREE.DataTexture(
      this.atmosphereData, gridW, gridH,
      THREE.RGBAFormat, THREE.FloatType
    );
    this.atmosphereTex.minFilter = THREE.LinearFilter;
    this.atmosphereTex.magFilter = THREE.LinearFilter;
    this.atmosphereTex.wrapS = THREE.ClampToEdgeWrapping;
    this.atmosphereTex.wrapT = THREE.ClampToEdgeWrapping;
    this.atmosphereTex.needsUpdate = true;
  }

  updateHeightmap(terrainSurface, soilStructure, moisture, waterMask) {
    const n = this.gridW * this.gridH;
    for (let i = 0; i < n; i++) {
      this.heightmapData[i * 4 + 0] = terrainSurface?.[i] ?? 0.5;
      this.heightmapData[i * 4 + 1] = soilStructure?.[i] ?? 0.5;
      this.heightmapData[i * 4 + 2] = moisture?.[i] ?? 0.2;
      this.heightmapData[i * 4 + 3] = waterMask?.[i] ?? 0.0;
    }
    this.heightmapTex.needsUpdate = true;
  }

  updateSoilColor(terrainVisuals, canopy) {
    const n = this.gridW * this.gridH;
    for (let i = 0; i < n; i++) {
      const rgb = terrainVisuals?.[i]?.rgb;
      if (rgb) {
        // Backend sends 0-1 float RGB from molecular optics pipeline
        this.soilColorData[i * 4 + 0] = rgb[0];
        this.soilColorData[i * 4 + 1] = rgb[1];
        this.soilColorData[i * 4 + 2] = rgb[2];
      } else {
        this.soilColorData[i * 4 + 0] = 0.45;
        this.soilColorData[i * 4 + 1] = 0.38;
        this.soilColorData[i * 4 + 2] = 0.28;
      }
      this.soilColorData[i * 4 + 3] = canopy?.[i] ?? 0.0;
    }
    this.soilColorTex.needsUpdate = true;
  }

  updateAtmosphere(atmosphere) {
    if (!atmosphere) return;
    const n = this.gridW * this.gridH;
    const temp = atmosphere.temperature;
    const humid = atmosphere.humidity;
    const wx = atmosphere.wind_x;
    const wy = atmosphere.wind_y;
    for (let i = 0; i < n; i++) {
      this.atmosphereData[i * 4 + 0] = temp?.[i] ?? 22.0;
      this.atmosphereData[i * 4 + 1] = humid?.[i] ?? 0.3;
      this.atmosphereData[i * 4 + 2] = wx?.[i] ?? 0.0;
      this.atmosphereData[i * 4 + 3] = wy?.[i] ?? 0.0;
    }
    this.atmosphereTex.needsUpdate = true;
  }
}

// ─── Shader Terrain Renderer ──────────────────────────────────────────

class ShaderTerrainRenderer {
  constructor(scene) {
    this.scene = scene;
    this.dataTextures = null;
    this.surfaceMesh = null;
    this.bodyMesh = null;
    this.grassMesh = null;
    this.waterMesh = null;
    this.gridW = 0;
    this.gridH = 0;
    this.time = 0;

    // Keep CPU-side surface field for organism placement (sampleSurfaceHeight)
    this.surfaceField = null;
  }

  // Constants from simulation
  static VOXEL_HEIGHT = 0.6;
  static VOXEL_BASE_Y = -0.2;
  static SURFACE_LEVELS = 7.0;
  static FLOOR_Y = -4.0;

  /**
   * Initialize or reinitialize for a new grid size.
   */
  init(w, h) {
    if (this.gridW === w && this.gridH === h && this.dataTextures) return;
    this.cleanup();
    this.gridW = w;
    this.gridH = h;
    this.dataTextures = new TerrainDataTextures(w, h);
    this.createSurfaceMesh(w, h);
    this.createBodyMesh(w, h);
    this.createGrassInstances(w, h);
    this.createWaterMesh(w, h);
  }

  cleanup() {
    [this.surfaceMesh, this.bodyMesh, this.grassMesh, this.waterMesh].forEach(mesh => {
      if (mesh) {
        this.scene.remove(mesh);
        if (mesh.geometry) mesh.geometry.dispose();
        if (mesh.material) mesh.material.dispose();
      }
    });
    this.surfaceMesh = this.bodyMesh = this.grassMesh = this.waterMesh = null;
  }

  // ─── Surface Mesh ───────────────────────────────────────────────────

  createSurfaceMesh(w, h) {
    // Dense tessellation for continuous smooth terrain — 4 vertices per cell.
    // The GPU LINEAR texture filtering interpolates between heightmap data points,
    // and the high vertex count produces smooth rolling hills without cell artifacts.
    const geo = new THREE.PlaneGeometry(w, h, w * 4, h * 4);
    geo.rotateX(-Math.PI / 2);

    const mat = new THREE.ShaderMaterial({
      uniforms: {
        tHeightmap: { value: this.dataTextures.heightmapTex },
        tSoilColor: { value: this.dataTextures.soilColorTex },
        uVoxelHeight: { value: ShaderTerrainRenderer.VOXEL_HEIGHT },
        uVoxelBaseY: { value: ShaderTerrainRenderer.VOXEL_BASE_Y },
        uSurfaceLevels: { value: ShaderTerrainRenderer.SURFACE_LEVELS },
        uDaylight: { value: 0.5 },
        uSunDir: { value: new THREE.Vector3(0.4, 0.7, 0.3) },
      },
      vertexShader: ShaderTerrainRenderer.TERRAIN_VERT,
      fragmentShader: ShaderTerrainRenderer.TERRAIN_FRAG,
      side: THREE.DoubleSide,
    });

    this.surfaceMesh = new THREE.Mesh(geo, mat);
    this.surfaceMesh.receiveShadow = true;
    this.scene.add(this.surfaceMesh);
  }

  // ─── Body Mesh (Side Walls with Geological Layers) ──────────────────

  createBodyMesh(w, h) {
    // Build a ring of quads around the perimeter.
    // UV encodes the heightmap coordinate so the body shader can
    // sample the correct terrain height at each perimeter cell.
    const positions = [];
    const uvs = []; // UV = (cellX/w, cellY/h) — maps directly to heightmap
    const indices = [];
    const verticalDivisions = 10;

    const ox = -w / 2;
    const oz = -h / 2;

    const addWallStrip = (x0, z0, x1, z1, cellX0, cellY0, cellX1, cellY1) => {
      const baseIdx = positions.length / 3;
      const u0 = (cellX0 + 0.5) / w;
      const v0 = (cellY0 + 0.5) / h;
      const u1 = (cellX1 + 0.5) / w;
      const v1 = (cellY1 + 0.5) / h;
      for (let vi = 0; vi <= verticalDivisions; vi++) {
        const t = vi / verticalDivisions; // 0=floor, 1=surface
        positions.push(x0, t, z0);
        uvs.push(u0, v0); // heightmap UV for this cell
        positions.push(x1, t, z1);
        uvs.push(u1, v1);
      }
      for (let vi = 0; vi < verticalDivisions; vi++) {
        const a = baseIdx + vi * 2;
        const b = a + 1;
        const c = a + 2;
        const d = a + 3;
        indices.push(a, c, b, b, c, d);
      }
    };

    // South wall (z = oz, y=0)
    for (let x = 0; x < w; x++) {
      addWallStrip(x + ox, oz, x + 1 + ox, oz, x, 0, Math.min(x+1, w-1), 0);
    }
    // North wall (z = oz + h, y=h-1)
    for (let x = w; x > 0; x--) {
      addWallStrip(x + ox, h + oz, x - 1 + ox, h + oz, Math.min(x, w-1), h-1, Math.max(x-1, 0), h-1);
    }
    // West wall (x = ox, x=0)
    for (let y = 0; y < h; y++) {
      addWallStrip(ox, y + oz, ox, y + 1 + oz, 0, y, 0, Math.min(y+1, h-1));
    }
    // East wall (x = ox + w, x=w-1)
    for (let y = h; y > 0; y--) {
      addWallStrip(w + ox, y + oz, w + ox, y - 1 + oz, w-1, Math.min(y, h-1), w-1, Math.max(y-1, 0));
    }

    // Floor quad
    const fi = positions.length / 3;
    positions.push(ox, 0, oz);          uvs.push(0, 0);
    positions.push(w + ox, 0, oz);      uvs.push(1, 0);
    positions.push(w + ox, 0, h + oz);  uvs.push(1, 1);
    positions.push(ox, 0, h + oz);      uvs.push(0, 1);
    indices.push(fi, fi+1, fi+2, fi, fi+2, fi+3);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    geo.setIndex(indices);
    geo.computeVertexNormals();

    const mat = new THREE.ShaderMaterial({
      uniforms: {
        tHeightmap: { value: this.dataTextures.heightmapTex },
        tSoilColor: { value: this.dataTextures.soilColorTex },
        uVoxelHeight: { value: ShaderTerrainRenderer.VOXEL_HEIGHT },
        uVoxelBaseY: { value: ShaderTerrainRenderer.VOXEL_BASE_Y },
        uSurfaceLevels: { value: ShaderTerrainRenderer.SURFACE_LEVELS },
        uFloorY: { value: ShaderTerrainRenderer.FLOOR_Y },
      },
      vertexShader: ShaderTerrainRenderer.BODY_VERT,
      fragmentShader: ShaderTerrainRenderer.BODY_FRAG,
      side: THREE.DoubleSide,
    });

    this.bodyMesh = new THREE.Mesh(geo, mat);
    this.scene.add(this.bodyMesh);
  }

  // ─── Instanced Grass ────────────────────────────────────────────────

  createGrassInstances(w, h) {
    const bladesPerCell = 12; // dense ground cover
    const instanceCount = w * h * bladesPerCell;

    // Blade geometry: tapered quad with slight bend (6 vertices, 4 triangles)
    const bladeGeo = new THREE.BufferGeometry();
    const bladeVerts = new Float32Array([
      -0.5, 0.0, 0.0,   // base-left
       0.5, 0.0, 0.0,   // base-right
      -0.35, 0.4, 0.0,  // mid-left
       0.35, 0.4, 0.0,  // mid-right
      -0.15, 0.75, 0.0, // upper-left
       0.15, 0.75, 0.0, // upper-right
       0.0, 1.0, 0.0,   // tip
    ]);
    const bladeIdx = [0,1,2, 1,3,2, 2,3,4, 3,5,4, 4,5,6];
    bladeGeo.setAttribute('position', new THREE.Float32BufferAttribute(bladeVerts, 3));
    bladeGeo.setIndex(bladeIdx);

    // Per-instance attributes — jittered placement within each cell
    const cellData = new Float32Array(instanceCount * 2);
    const seedData = new Float32Array(instanceCount);
    const bladeData = new Float32Array(instanceCount);

    // Simple seeded random for deterministic but varied placement
    const seededRand = (x, y, b) => {
      let h = (x * 374761 + y * 668265 + b * 982451) & 0x7FFFFFFF;
      h = ((h >> 16) ^ h) * 0x45d9f3b;
      h = ((h >> 16) ^ h) * 0x45d9f3b;
      h = (h >> 16) ^ h;
      return (h & 0xFFFF) / 65535;
    };

    let idx = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        for (let b = 0; b < bladesPerCell; b++) {
          // Jitter position within cell for natural look
          cellData[idx * 2] = x + (seededRand(x, y, b * 2) - 0.5) * 0.15;
          cellData[idx * 2 + 1] = y + (seededRand(x, y, b * 2 + 1) - 0.5) * 0.15;
          seedData[idx] = seededRand(x, y, b + 100);
          bladeData[idx] = b;
          idx++;
        }
      }
    }

    const instancedGeo = new THREE.InstancedBufferGeometry();
    instancedGeo.index = bladeGeo.index;
    instancedGeo.setAttribute('position', bladeGeo.getAttribute('position'));
    instancedGeo.setAttribute('instanceCell', new THREE.InstancedBufferAttribute(cellData, 2));
    instancedGeo.setAttribute('instanceSeed', new THREE.InstancedBufferAttribute(seedData, 1));
    instancedGeo.setAttribute('instanceBlade', new THREE.InstancedBufferAttribute(bladeData, 1));
    instancedGeo.instanceCount = instanceCount;

    const mat = new THREE.ShaderMaterial({
      uniforms: {
        tHeightmap: { value: this.dataTextures.heightmapTex },
        tSoilColor: { value: this.dataTextures.soilColorTex },
        tAtmosphere: { value: this.dataTextures.atmosphereTex },
        uTime: { value: 0 },
        uGridW: { value: w },
        uGridH: { value: h },
        uVoxelHeight: { value: ShaderTerrainRenderer.VOXEL_HEIGHT },
        uVoxelBaseY: { value: ShaderTerrainRenderer.VOXEL_BASE_Y },
        uSurfaceLevels: { value: ShaderTerrainRenderer.SURFACE_LEVELS },
      },
      vertexShader: ShaderTerrainRenderer.GRASS_VERT,
      fragmentShader: ShaderTerrainRenderer.GRASS_FRAG,
      side: THREE.DoubleSide,
      transparent: false,
    });

    this.grassMesh = new THREE.Mesh(instancedGeo, mat);
    this.grassMesh.frustumCulled = false;
    this.scene.add(this.grassMesh);
  }

  // ─── Water Mesh ─────────────────────────────────────────────────────

  createWaterMesh(w, h) {
    // Higher tessellation for smooth water surface (matches terrain resolution)
    const geo = new THREE.PlaneGeometry(w, h, w * 3, h * 3);
    geo.rotateX(-Math.PI / 2);

    const mat = new THREE.ShaderMaterial({
      uniforms: {
        tHeightmap: { value: this.dataTextures.heightmapTex },
        tAtmosphere: { value: this.dataTextures.atmosphereTex },
        uTime: { value: 0 },
        uDaylight: { value: 0.5 },
        uSunDir: { value: new THREE.Vector3(0.4, 0.7, 0.3) },
        uVoxelHeight: { value: ShaderTerrainRenderer.VOXEL_HEIGHT },
        uVoxelBaseY: { value: ShaderTerrainRenderer.VOXEL_BASE_Y },
        uSurfaceLevels: { value: ShaderTerrainRenderer.SURFACE_LEVELS },
      },
      vertexShader: ShaderTerrainRenderer.WATER_VERT,
      fragmentShader: ShaderTerrainRenderer.WATER_FRAG,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
    });

    this.waterMesh = new THREE.Mesh(geo, mat);
    this.waterMesh.renderOrder = 1; // render after opaque
    this.scene.add(this.waterMesh);
  }

  // ─── Per-Frame Update ───────────────────────────────────────────────

  update(terrainSurface, terrainVisuals, moisture, waterMask, soilStructure, atmosphere, daylight, dt) {
    if (!this.dataTextures) return;

    this.time += dt || 0.016;

    // Update DataTextures (the only CPU work per frame: ~17 KB of typed array copies)
    this.dataTextures.updateHeightmap(terrainSurface, soilStructure, moisture, waterMask);
    this.dataTextures.updateSoilColor(terrainVisuals, null);
    this.dataTextures.updateAtmosphere(atmosphere);

    // Update shader uniforms — sun direction from backend solar position
    if (this.surfaceMesh) {
      this.surfaceMesh.material.uniforms.uDaylight.value = daylight || 0.5;
      // sunDirection: [east-west, north-south, vertical] from backend
      // Three.js: x=east-west, y=vertical, z=north-south
      if (typeof sunDirection !== 'undefined' && Array.isArray(sunDirection)) {
        this.surfaceMesh.material.uniforms.uSunDir.value.set(
          sunDirection[0], sunDirection[2], sunDirection[1]
        );
      }
    }
    if (this.grassMesh) {
      this.grassMesh.material.uniforms.uTime.value = this.time;
    }
    if (this.waterMesh) {
      this.waterMesh.material.uniforms.uTime.value = this.time;
      this.waterMesh.material.uniforms.uDaylight.value = daylight || 0.5;
      if (typeof sunDirection !== 'undefined' && Array.isArray(sunDirection)) {
        this.waterMesh.material.uniforms.uSunDir.value.set(
          sunDirection[0], sunDirection[2], sunDirection[1]
        );
      }
      // Water mesh always visible — the GPU shader handles per-vertex
      // visibility via vWaterPresence (discards fragments where no standing water).
      // This prevents flickering from CPU-side threshold oscillation.
      this.waterMesh.visible = true;
    }

    // Keep CPU-side surface field for organism placement
    this.surfaceField = terrainSurface;
  }

  /**
   * Sample surface height at a world position (for placing organisms).
   * This is the only CPU-side height computation — used by entity placement, not rendering.
   */
  sampleSurfaceHeight(wx, wz) {
    if (!this.surfaceField || !this.gridW) return 0;
    const fx = wx + this.gridW / 2 - 0.5;
    const fy = wz + this.gridH / 2 - 0.5;
    const ix = Math.floor(fx);
    const iy = Math.floor(fy);
    if (ix < 0 || ix >= this.gridW || iy < 0 || iy >= this.gridH) return 0;
    const relief = this.surfaceField[iy * this.gridW + ix] || 0;
    const normalized = Math.min(relief / 1.1, 1.0);
    const cH = ShaderTerrainRenderer.VOXEL_BASE_Y +
      (normalized * (ShaderTerrainRenderer.SURFACE_LEVELS - 1) + 1) * ShaderTerrainRenderer.VOXEL_HEIGHT;
    const qH = ShaderTerrainRenderer.VOXEL_BASE_Y +
      (Math.round(normalized * (ShaderTerrainRenderer.SURFACE_LEVELS - 1)) + 1) * ShaderTerrainRenderer.VOXEL_HEIGHT;
    return cH * 0.74 + qH * 0.26 + ShaderTerrainRenderer.VOXEL_HEIGHT * 0.06;
  }

  /**
   * Set opacity for all terrain meshes (for scale transition crossfade).
   * @param {number} opacity - 0.0 (invisible) to 1.0 (fully opaque)
   */
  setOpacity(opacity) {
    const meshes = [this.surfaceMesh, this.bodyMesh, this.grassMesh, this.waterMesh];
    for (const mesh of meshes) {
      if (!mesh || !mesh.material) continue;
      if (opacity >= 0.99) {
        mesh.material.transparent = false;
        mesh.material.opacity = 1.0;
        mesh.visible = true;
      } else if (opacity <= 0.01) {
        mesh.visible = false;
      } else {
        mesh.material.transparent = true;
        mesh.material.opacity = opacity;
        mesh.material.depthWrite = opacity > 0.5;
        mesh.visible = true;
      }
    }
  }
}

// ─── Inline Shaders (loaded from GLSL files at build time) ────────────
// These are set by the build system or inline in the HTML.

ShaderTerrainRenderer.TERRAIN_VERT = `
uniform sampler2D tHeightmap;
uniform float uVoxelHeight;
uniform float uVoxelBaseY;
uniform float uSurfaceLevels;
varying vec2 vTerrainUv;
varying float vRelief;
varying float vMoisture;
varying float vHeight;
varying vec3 vWorldPos;
void main() {
  vTerrainUv = uv;
  vec4 hm = texture2D(tHeightmap, uv);
  float relief = hm.r;
  vRelief = relief;
  vMoisture = hm.b;
  float normalized = clamp(relief / 1.1, 0.0, 1.0);
  float cH = uVoxelBaseY + (normalized * (uSurfaceLevels - 1.0) + 1.0) * uVoxelHeight;
  float qH = uVoxelBaseY + (floor(normalized * (uSurfaceLevels - 1.0) + 0.5) + 1.0) * uVoxelHeight;
  float shellHeight = cH * 0.74 + qH * 0.26 + uVoxelHeight * 0.06;
  vHeight = shellHeight;
  vec3 displaced = position;
  displaced.y = shellHeight;
  vWorldPos = (modelMatrix * vec4(displaced, 1.0)).xyz;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(displaced, 1.0);
}`;

ShaderTerrainRenderer.TERRAIN_FRAG = `
uniform sampler2D tSoilColor;
uniform sampler2D tHeightmap;
uniform float uDaylight;
uniform vec3 uSunDir;
varying vec2 vTerrainUv;
varying float vMoisture;
varying vec3 vWorldPos;
void main() {
  vec4 soil = texture2D(tSoilColor, vTerrainUv);
  vec4 hm = texture2D(tHeightmap, vTerrainUv);
  float waterMask = hm.a;
  vec3 color = soil.rgb;
  // Moisture darkening (Lobell & Asner 2002)
  color *= 1.0 - vMoisture * 0.35;
  // Standing water: visible ONLY when water exceeds soil pore capacity.
  // porosity = soil_structure * (1.08 - moisture * 0.58)  [van Genuchten model]
  // If water_mask > porosity: water pools on surface. Otherwise: only pore water
  // (invisible, but causes moisture darkening above via Lobell & Asner).
  float soilTexture = hm.g;
  float porosity = soilTexture * (1.08 - vMoisture * 0.58);
  // Standing water visible only when water exceeds porosity AND terrain is low.
  // The water_mask includes pore water for ecology; rendering filters for surface pools.
  float standingWater = max(0.0, waterMask - porosity);
  // Terrain depression: water pools in low-relief cells, drains from hilltops.
  float reliefNorm = clamp(hm.r / 1.1, 0.0, 1.0);
  standingWater *= 1.0 - smoothstep(0.25, 0.50, reliefNorm);
  if (standingWater > 0.001) {
    float depth = standingWater * 0.6;
    vec3 waterColor = vec3(
      0.18 * exp(-0.55 * depth),
      0.42 * exp(-0.12 * depth),
      0.72 * exp(-0.02 * depth)
    );
    float waterBlend = clamp(standingWater * 4.0, 0.0, 0.8);
    color = mix(color, waterColor, waterBlend);
  }
  // Derivative-based normal for lighting
  vec3 dx = dFdx(vWorldPos);
  vec3 dy = dFdy(vWorldPos);
  vec3 normal = normalize(cross(dx, dy));
  // Sun direction from backend solar position (compute_solar_state)
  // uSunDir = [east-west, vertical, north-south] in Three.js space
  vec3 lightDir = normalize(uSunDir);
  float diffuse = max(dot(normal, lightDir), 0.0);
  // Ambient scales with daylight (more sky contribution during day)
  float ambient = 0.30 + uDaylight * 0.20;
  color *= ambient + diffuse * (0.35 + uDaylight * 0.25);
  gl_FragColor = vec4(color, 1.0);
}`;

// Body vertex shader: UV now maps directly to heightmap cell coordinates.
// Top vertices displaced to terrain height, bottom pinned at floor.
ShaderTerrainRenderer.BODY_VERT = `
uniform sampler2D tHeightmap;
uniform float uVoxelHeight;
uniform float uVoxelBaseY;
uniform float uSurfaceLevels;
uniform float uFloorY;
varying vec2 vTerrainUv;
varying float vNormalizedDepth;
varying float vWorldY;
varying float vSurfaceY;
varying vec3 vWorldPos;
void main() {
  vTerrainUv = uv; // UV = (cellX/gridW, cellY/gridH) — direct heightmap lookup
  vec4 hm = texture2D(tHeightmap, uv);
  float relief = hm.r;
  float normalized = clamp(relief / 1.1, 0.0, 1.0);
  float cH = uVoxelBaseY + (normalized * (uSurfaceLevels - 1.0) + 1.0) * uVoxelHeight;
  float qH = uVoxelBaseY + (floor(normalized * (uSurfaceLevels - 1.0) + 0.5) + 1.0) * uVoxelHeight;
  float surfaceY = cH * 0.74 + qH * 0.26 + uVoxelHeight * 0.06;
  vSurfaceY = surfaceY;
  float verticalT = position.y;
  float worldY = mix(uFloorY, surfaceY, verticalT);
  vWorldY = worldY;
  vNormalizedDepth = verticalT;
  vec3 displaced = position;
  displaced.y = worldY;
  vWorldPos = (modelMatrix * vec4(displaced, 1.0)).xyz;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(displaced, 1.0);
}`;

ShaderTerrainRenderer.BODY_FRAG = `
uniform sampler2D tSoilColor;
uniform sampler2D tHeightmap;
varying vec2 vTerrainUv;
varying float vNormalizedDepth;
varying float vWorldY;
varying vec3 vWorldPos;
void main() {
  vec4 soil = texture2D(tSoilColor, vTerrainUv);
  vec4 hm = texture2D(tHeightmap, vTerrainUv);
  vec3 surfaceColor = soil.rgb;
  float soilTexture = hm.g; // 0=sand, 1=clay from simulation
  // Mineral base: derive from surface color by desaturating and darkening.
  // Deep rock has less organic matter → less color saturation.
  // This avoids hardcoded mineral colors — the base emerges from
  // removing the organic component of the surface soil.
  float surfaceLuminance = dot(surfaceColor, vec3(0.299, 0.587, 0.114));
  vec3 mineralBase = mix(surfaceColor, vec3(surfaceLuminance), 0.6) * 0.75;
  // Sandy parent material (low texture) is lighter than clayey (high texture)
  mineralBase *= 0.85 + (1.0 - soilTexture) * 0.25;
  vec3 bedrockColor = mix(surfaceColor, mineralBase * 0.7, 0.7);
  vec3 subsoilColor = mix(surfaceColor, mineralBase, 0.45);
  float t = vNormalizedDepth;
  vec3 color;
  if (t < 0.30) {
    float blend = smoothstep(0.0, 0.10, t);
    color = mix(bedrockColor * 0.85, bedrockColor, blend);
  } else if (t < 0.65) {
    float blend = smoothstep(0.30, 0.45, t);
    color = mix(bedrockColor, subsoilColor, blend);
  } else {
    float blend = smoothstep(0.65, 0.82, t);
    color = mix(subsoilColor, surfaceColor, blend);
  }
  color *= 0.72;
  float band = sin(vWorldY * 18.0) * 0.02 + sin(vWorldY * 7.3) * 0.015;
  color += band;
  gl_FragColor = vec4(color, 1.0);
}`;

// Grass: grows from the terrain surface. Density from organic matter (emergent).
// Height formula MUST match the terrain surface shader exactly.
// Uses modelViewMatrix so position matches the terrain mesh.
ShaderTerrainRenderer.GRASS_VERT = `
uniform sampler2D tHeightmap;
uniform sampler2D tSoilColor;
uniform sampler2D tAtmosphere;
uniform float uTime;
uniform float uGridW;
uniform float uGridH;
uniform float uVoxelHeight;
uniform float uVoxelBaseY;
uniform float uSurfaceLevels;
attribute vec2 instanceCell;
attribute float instanceSeed;
attribute float instanceBlade;
varying vec3 vGrassColor;
varying float vAlpha;
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

// Height formula — IDENTICAL to terrain surface shader
float terrainHeight(float relief) {
  float normalized = clamp(relief / 1.1, 0.0, 1.0);
  float cH = uVoxelBaseY + (normalized * (uSurfaceLevels - 1.0) + 1.0) * uVoxelHeight;
  float qH = uVoxelBaseY + (floor(normalized * (uSurfaceLevels - 1.0) + 0.5) + 1.0) * uVoxelHeight;
  return cH * 0.74 + qH * 0.26 + uVoxelHeight * 0.06;
}

void main() {
  // Per-blade random offset within cell
  float phase = instanceSeed + instanceBlade * 0.25;
  vec2 bladeOffset = vec2(
    (hash(vec2(phase, 0.13)) - 0.5) * 0.7,
    (hash(vec2(phase, 0.47)) - 0.5) * 0.7
  );

  // Sample heightmap at ACTUAL blade position (not just cell center)
  // This ensures blades sit exactly on the terrain surface
  vec2 bladeUv = (instanceCell + 0.5 + bladeOffset) / vec2(uGridW, uGridH);
  bladeUv = clamp(bladeUv, vec2(0.0), vec2(1.0));
  vec4 hm = texture2D(tHeightmap, bladeUv);
  float relief = hm.r;
  float moisture = hm.b;

  // Compute terrain height at blade position — same formula as surface mesh
  float surfaceY = terrainHeight(relief);

  // Check if underwater — grass doesn't grow where water sources deposit water
  float waterMask = hm.a;
  float isUnderwater = step(0.82, waterMask);

  // Green signal from emergent soil color (organic matter / chlorophyll)
  vec2 cellUv = (instanceCell + 0.5) / vec2(uGridW, uGridH);
  vec4 soil = texture2D(tSoilColor, cellUv);
  // Green signal from soil organic content — even brown soil grows grass
  // (organic matter > 0 means some plant growth is possible)
  float organicSignal = clamp(soil.g * 2.0 + soil.r * 0.5, 0.0, 1.0);
  float greenSignal = clamp(
    (soil.g - soil.r) * 2.4 + (soil.g - soil.b) * 1.3 + organicSignal * 0.6, 0.0, 1.0);

  // Most terrain should have grass — only bare rock/sand or underwater is bare
  float densityRoll = hash(vec2(instanceSeed * 7.3, instanceBlade * 3.1));
  // Low threshold: grass grows almost everywhere with organic content
  float visible = step(0.05, organicSignal) * step(densityRoll, 0.6 + greenSignal * 0.4) * (1.0 - isUnderwater);

  // World position of blade base — ON the terrain surface
  float ox = -uGridW * 0.5;
  float oz = -uGridH * 0.5;
  vec3 basePos = vec3(
    instanceCell.x + 0.5 + ox + bladeOffset.x,
    surfaceY,  // exactly on surface, no offset
    instanceCell.y + 0.5 + oz + bladeOffset.y
  );

  // Blade height: short grass, not tall sticks
  float heightVariation = 0.4 + hash(vec2(instanceSeed, 0.77)) * 0.6;
  float bladeHeight = visible * (0.04 + organicSignal * 0.12) * heightVariation;

  // Wind from atmospheric simulation
  vec4 atmo = texture2D(tAtmosphere, cellUv);
  float windX = atmo.b;
  float windY = atmo.a;
  float windStrength = length(vec2(windX, windY));

  // Blade deformation: position.y is [0,1] from base to tip
  float tipFactor = clamp(position.y, 0.0, 1.0);
  float windBend = tipFactor * tipFactor * windStrength * 0.5;
  float sway = sin(uTime * 2.2 + basePos.x * 1.5 + basePos.z * 1.1 + instanceSeed * 6.28)
             * tipFactor * 0.03 * (1.0 + windStrength * 0.4);

  vec3 worldPos = basePos;
  worldPos.y += tipFactor * bladeHeight;
  worldPos.x += windBend * sign(windX + 0.001) + sway;
  worldPos.z += windBend * sign(windY + 0.001) + sway * 0.6;

  // Blade width — narrow grass blades
  worldPos.x += position.x * 0.012 * (1.0 - tipFactor * 0.6);
  worldPos.z += position.z * 0.004;

  // Color emerges from soil color (chlorophyll/organic content)
  vec3 baseGreen = vec3(soil.r * 0.65, soil.g * 1.15, soil.b * 0.55);
  baseGreen = clamp(baseGreen, 0.0, 1.0);
  vGrassColor = mix(baseGreen * 0.85, baseGreen * 1.1, tipFactor);
  vGrassColor *= 1.0 - moisture * 0.15;
  vAlpha = visible;

  // Use modelViewMatrix to match terrain mesh coordinate space
  gl_Position = projectionMatrix * modelViewMatrix * vec4(worldPos, 1.0);
}`;

ShaderTerrainRenderer.GRASS_FRAG = `
varying vec3 vGrassColor;
varying float vAlpha;
void main() {
  if (vAlpha < 0.5) discard;
  gl_FragColor = vec4(vGrassColor, 1.0);
}`;

// Water: per-cell visibility from actual water source deposits.
// Water_mask channel (heightmap.a) is high (>0.45) ONLY where water sources
// have deposited water via deposit_2d. Lower values are dissolved pore water.
// The water surface sits just above terrain height at deposited cells.
ShaderTerrainRenderer.WATER_VERT = `
uniform sampler2D tHeightmap;
uniform sampler2D tAtmosphere;
uniform float uTime;
uniform float uVoxelHeight;
uniform float uVoxelBaseY;
uniform float uSurfaceLevels;
varying vec2 vUv;
varying float vWaterPresence;
varying float vWaterDepth;
varying vec3 vWorldPos;

float terrainH(float relief) {
  float n = clamp(relief / 1.1, 0.0, 1.0);
  float cH = uVoxelBaseY + (n * (uSurfaceLevels - 1.0) + 1.0) * uVoxelHeight;
  float qH = uVoxelBaseY + (floor(n * (uSurfaceLevels - 1.0) + 0.5) + 1.0) * uVoxelHeight;
  return cH * 0.74 + qH * 0.26 + uVoxelHeight * 0.06;
}

void main() {
  vUv = uv;
  vec4 hm = texture2D(tHeightmap, uv);
  float relief = hm.r;
  float waterMask = hm.a;

  // Only cells with deposited water from actual sources show water
  // waterMask > 0.45 = from deposit_2d (water source proximity)
  // waterMask < 0.45 = dissolved pore water (not visible)
  // Only cells with very high water_mask (direct water source deposits)
  // show visible standing water. Substrate dissolved water is invisible.
  // Standing water: physical — visible when water exceeds soil pore capacity.
  // porosity = soilTexture * (1.08 - moisture * 0.58)
  // Standing water appears only when water_mask > porosity (physical threshold).
  float soilTexture = hm.g;
  float moisture = hm.b;
  float porosity = soilTexture * (1.08 - moisture * 0.58);
  // Standing water: porosity + depression filter (rendering-side only).
  float standingWater = max(0.0, waterMask - porosity);
  float reliefNorm = clamp(relief / 1.1, 0.0, 1.0);
  standingWater *= 1.0 - smoothstep(0.25, 0.50, reliefNorm);
  vWaterPresence = smoothstep(0.005, 0.04, standingWater);
  float surfaceY = terrainH(relief);

  // Water depth: directly from standing water amount (physical, not hardcoded)
  vWaterDepth = standingWater * 0.6;
  float waterY = surfaceY + vWaterDepth * vWaterPresence + 0.02;

  // Wind-driven ripples
  vec4 atmo = texture2D(tAtmosphere, uv);
  float windStrength = length(vec2(atmo.b, atmo.a));
  float ripple = sin(position.x*12.0+uTime*3.0)*cos(position.z*9.0+uTime*2.0)
               * 0.008 * vWaterPresence * (0.3 + windStrength * 0.7);

  vec3 displaced = position;
  displaced.y = waterY + ripple;
  vWorldPos = (modelMatrix * vec4(displaced, 1.0)).xyz;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(displaced, 1.0);
}`;

// Water fragment: Pope & Fry 1997 spectral absorption.
// Per-cell visibility — only renders where actual water sources exist.
ShaderTerrainRenderer.WATER_FRAG = `
uniform float uTime;
uniform float uDaylight;
uniform vec3 uSunDir;
varying vec2 vUv;
varying float vWaterPresence;
varying float vWaterDepth;
varying vec3 vWorldPos;
void main() {
  if (vWaterPresence < 0.02) discard;

  // Pope & Fry 1997: spectral absorption in pure water.
  // Beer-Lambert: I = I0 * exp(-a * depth) where a is the absorption coefficient.
  // Shallow water is nearly transparent. Deep water absorbs red first, then green.
  float depth = clamp(vWaterDepth, 0.005, 1.5);
  vec3 waterColor = vec3(
    0.22 * exp(-0.55 * depth),  // red absorbed first
    0.48 * exp(-0.12 * depth),  // green persists longer
    0.78 * exp(-0.02 * depth)   // blue persists longest
  );

  // Opacity from Beer-Lambert: 1 - transmittance = 1 - exp(-extinction * depth).
  // Shallow water is almost invisible. Deep water is opaque.
  // Extinction ~2.5 /m for turbid natural water (Kirk 1994).
  float transmittance = exp(-2.5 * depth);
  float opacity = vWaterPresence * (1.0 - transmittance);

  // Wind-driven caustic refraction patterns on the bottom
  float caustic = sin(vWorldPos.x * 14.0 + uTime * 1.2)
                * sin(vWorldPos.z * 11.0 + uTime * 0.9)
                * 0.04 * uDaylight * (1.0 - opacity * 0.5);
  waterColor += caustic;

  // Sun specular: Fresnel reflection on water surface
  vec3 lightDir = normalize(uSunDir);
  float specAngle = max(lightDir.y, 0.0); // y = up in Three.js
  float specular = pow(specAngle, 12.0) * 0.12 * uDaylight * vWaterPresence;
  waterColor += vec3(specular);

  gl_FragColor = vec4(waterColor, opacity);
}`;
