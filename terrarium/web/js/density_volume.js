// ---------------------------------------------------------------------------
// DensityVolume — ray-marching volume renderer for chemistry concentration fields
// ---------------------------------------------------------------------------
// Renders spatial concentration gradients as smooth volumetric clouds rather
// than per-cell flat colors. Uses a 3D DataTexture sampled from the backend
// snapshot's chemistry data, with ray-marching through a box volume.
//
// This creates the organic, diffusion-like appearance seen in scientific
// molecular visualization (VMD Gaussian density isosurfaces).
//
// Usage:
//   densityVolume.setConcentrationField(name, data, gridW, gridH, color)
//   densityVolume.show() / densityVolume.hide()
//   densityVolume.animate(dt)

const VOLUME_VERT = `
  varying vec3 vWorldPos;
  varying vec3 vLocalPos;
  void main() {
    vLocalPos = position;
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPos = worldPos.xyz;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const VOLUME_FRAG = `
  precision highp float;
  precision highp sampler3D;

  varying vec3 vWorldPos;
  varying vec3 vLocalPos;

  uniform sampler3D tDensity;
  uniform vec3 uColor;
  uniform float uOpacity;
  uniform float uThreshold;
  uniform vec3 uCameraPos;
  uniform vec3 uVolumeMin;
  uniform vec3 uVolumeMax;
  uniform float uTime;

  // Ray-box intersection (slab method)
  vec2 intersectBox(vec3 rayOrigin, vec3 rayDir, vec3 boxMin, vec3 boxMax) {
    vec3 invDir = 1.0 / rayDir;
    vec3 t0 = (boxMin - rayOrigin) * invDir;
    vec3 t1 = (boxMax - rayOrigin) * invDir;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    float tNear = max(max(tmin.x, tmin.y), tmin.z);
    float tFar = min(min(tmax.x, tmax.y), tmax.z);
    return vec2(tNear, tFar);
  }

  void main() {
    // Ray from camera through this fragment
    vec3 rayDir = normalize(vWorldPos - uCameraPos);
    vec3 rayOrigin = uCameraPos;

    // Intersect ray with volume box (in local coords: -0.5 to 0.5)
    vec2 tRange = intersectBox(rayOrigin, rayDir, uVolumeMin, uVolumeMax);
    if (tRange.x > tRange.y) discard;
    tRange.x = max(tRange.x, 0.0);

    // Ray march through volume
    float stepSize = 0.02;
    int maxSteps = 64;
    vec4 accum = vec4(0.0);

    for (int i = 0; i < 64; i++) {
      if (float(i) * stepSize > (tRange.y - tRange.x)) break;

      float t = tRange.x + float(i) * stepSize;
      vec3 samplePos = rayOrigin + rayDir * t;

      // Convert world pos to texture coords (0-1)
      vec3 texCoord = (samplePos - uVolumeMin) / (uVolumeMax - uVolumeMin);
      if (any(lessThan(texCoord, vec3(0.0))) || any(greaterThan(texCoord, vec3(1.0)))) continue;

      // Sample density
      float density = texture(tDensity, texCoord).r;

      // Apply threshold and smooth falloff
      float alpha = smoothstep(uThreshold, uThreshold + 0.2, density) * uOpacity * stepSize * 3.0;

      // Accumulate color with front-to-back compositing
      vec3 sampleColor = uColor * (0.6 + density * 0.4);
      accum.rgb += sampleColor * alpha * (1.0 - accum.a);
      accum.a += alpha * (1.0 - accum.a);

      // Early termination
      if (accum.a > 0.95) break;
    }

    if (accum.a < 0.01) discard;
    gl_FragColor = vec4(accum.rgb, accum.a);
  }
`;

// Fallback for WebGL 1 (no sampler3D): use a 2D texture atlas
const VOLUME_FRAG_WEBGL1 = `
  precision highp float;

  varying vec3 vWorldPos;
  varying vec3 vLocalPos;

  uniform sampler2D tDensityAtlas;
  uniform vec3 uColor;
  uniform float uOpacity;
  uniform float uThreshold;
  uniform vec3 uCameraPos;
  uniform vec3 uVolumeMin;
  uniform vec3 uVolumeMax;
  uniform float uGridW;
  uniform float uGridH;
  uniform float uGridD;
  uniform float uTime;

  // Sample from 2D atlas (slices laid out horizontally)
  float sampleDensity(vec3 texCoord) {
    float z = texCoord.z * uGridD;
    float zFloor = floor(z);
    float zFrac = z - zFloor;

    // Two adjacent slices
    float slice0 = clamp(zFloor, 0.0, uGridD - 1.0);
    float slice1 = clamp(zFloor + 1.0, 0.0, uGridD - 1.0);

    // Atlas UV: each slice occupies 1/gridD of the atlas width
    vec2 uv0 = vec2((slice0 + texCoord.x) / uGridD, texCoord.y);
    vec2 uv1 = vec2((slice1 + texCoord.x) / uGridD, texCoord.y);

    float d0 = texture2D(tDensityAtlas, uv0).r;
    float d1 = texture2D(tDensityAtlas, uv1).r;

    return mix(d0, d1, zFrac);
  }

  // Ray-box intersection
  vec2 intersectBox(vec3 rayOrigin, vec3 rayDir, vec3 boxMin, vec3 boxMax) {
    vec3 invDir = 1.0 / rayDir;
    vec3 t0 = (boxMin - rayOrigin) * invDir;
    vec3 t1 = (boxMax - rayOrigin) * invDir;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    float tNear = max(max(tmin.x, tmin.y), tmin.z);
    float tFar = min(min(tmax.x, tmax.y), tmax.z);
    return vec2(tNear, tFar);
  }

  void main() {
    vec3 rayDir = normalize(vWorldPos - uCameraPos);
    vec3 rayOrigin = uCameraPos;

    vec2 tRange = intersectBox(rayOrigin, rayDir, uVolumeMin, uVolumeMax);
    if (tRange.x > tRange.y) discard;
    tRange.x = max(tRange.x, 0.0);

    float stepSize = 0.03;
    vec4 accum = vec4(0.0);

    for (int i = 0; i < 48; i++) {
      if (float(i) * stepSize > (tRange.y - tRange.x)) break;

      float t = tRange.x + float(i) * stepSize;
      vec3 samplePos = rayOrigin + rayDir * t;

      vec3 texCoord = (samplePos - uVolumeMin) / (uVolumeMax - uVolumeMin);
      if (any(lessThan(texCoord, vec3(0.0))) || any(greaterThan(texCoord, vec3(1.0)))) continue;

      float density = sampleDensity(texCoord);
      float alpha = smoothstep(uThreshold, uThreshold + 0.15, density) * uOpacity * stepSize * 3.0;

      vec3 sampleColor = uColor * (0.6 + density * 0.4);
      accum.rgb += sampleColor * alpha * (1.0 - accum.a);
      accum.a += alpha * (1.0 - accum.a);

      if (accum.a > 0.95) break;
    }

    if (accum.a < 0.01) discard;
    gl_FragColor = vec4(accum.rgb, accum.a);
  }
`;


class DensityVolume {
  constructor(scene) {
    this.scene = scene;
    this.group = new THREE.Group();
    this.group.visible = false;
    this.group.name = 'densityVolume';
    this.scene.add(this.group);

    this._volumeMesh = null;
    this._material = null;
    this._time = 0;

    // Check WebGL2 support for 3D textures
    this._useWebGL2 = false; // Default to WebGL1 atlas for compatibility
  }

  /**
   * Set a concentration field from per-cell data.
   * @param {string} name - field name (e.g., 'glucose', 'oxygen')
   * @param {Float32Array|number[]} data - concentration values, length = w * h (2D) or w * h * d (3D)
   * @param {number} w - grid width
   * @param {number} h - grid height
   * @param {number} d - grid depth (default 1 for 2D fields)
   * @param {number[]} color - [r,g,b] 0-1
   * @param {number[]} volumeMin - [x,y,z] world-space minimum corner
   * @param {number[]} volumeMax - [x,y,z] world-space maximum corner
   */
  setConcentrationField(name, data, w, h, d, color, volumeMin, volumeMax) {
    this._clear();

    d = d || 1;
    const totalCells = w * h * d;

    // For 2D fields, extrude into 3D by repeating with vertical falloff
    let volumeData;
    if (d <= 1) {
      d = 4; // Create 4 depth layers
      volumeData = new Float32Array(w * h * d);
      for (let z = 0; z < d; z++) {
        const depthFactor = 1.0 - (z / d) * 0.6; // fade with depth
        for (let i = 0; i < w * h; i++) {
          volumeData[z * w * h + i] = (data[i] || 0) * depthFactor;
        }
      }
    } else {
      volumeData = new Float32Array(data);
    }

    // Normalize to 0-1
    let maxVal = 0;
    for (let i = 0; i < volumeData.length; i++) {
      maxVal = Math.max(maxVal, volumeData[i]);
    }
    if (maxVal > 0) {
      for (let i = 0; i < volumeData.length; i++) {
        volumeData[i] /= maxVal;
      }
    }

    // Gaussian blur for smoothness (simple 3x3x3 box blur)
    const blurred = new Float32Array(volumeData.length);
    for (let z = 0; z < d; z++) {
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          let sum = 0, count = 0;
          for (let dz = -1; dz <= 1; dz++) {
            for (let dy = -1; dy <= 1; dy++) {
              for (let dx = -1; dx <= 1; dx++) {
                const nx = x + dx, ny = y + dy, nz = z + dz;
                if (nx >= 0 && nx < w && ny >= 0 && ny < h && nz >= 0 && nz < d) {
                  sum += volumeData[nz * w * h + ny * w + nx];
                  count++;
                }
              }
            }
          }
          blurred[z * w * h + y * w + x] = sum / count;
        }
      }
    }

    // Create 2D atlas texture (WebGL1 compatible)
    // Layout: d slices side by side horizontally
    const atlasW = w * d;
    const atlasH = h;
    const atlasData = new Float32Array(atlasW * atlasH);
    for (let z = 0; z < d; z++) {
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          atlasData[y * atlasW + z * w + x] = blurred[z * w * h + y * w + x];
        }
      }
    }

    const tex = new THREE.DataTexture(atlasData, atlasW, atlasH, THREE.LuminanceFormat, THREE.FloatType);
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.needsUpdate = true;

    const vMin = new THREE.Vector3(volumeMin[0], volumeMin[1], volumeMin[2]);
    const vMax = new THREE.Vector3(volumeMax[0], volumeMax[1], volumeMax[2]);
    const vSize = new THREE.Vector3().subVectors(vMax, vMin);
    const vCenter = new THREE.Vector3().addVectors(vMin, vMax).multiplyScalar(0.5);

    this._material = new THREE.ShaderMaterial({
      vertexShader: VOLUME_VERT,
      fragmentShader: VOLUME_FRAG_WEBGL1,
      uniforms: {
        tDensityAtlas: { value: tex },
        uColor: { value: new THREE.Color(color[0], color[1], color[2]) },
        uOpacity: { value: 0.6 },
        uThreshold: { value: 0.15 },
        uCameraPos: { value: new THREE.Vector3() },
        uVolumeMin: { value: vMin },
        uVolumeMax: { value: vMax },
        uGridW: { value: w },
        uGridH: { value: h },
        uGridD: { value: d },
        uTime: { value: 0 },
      },
      side: THREE.BackSide, // render back faces so ray enters from behind
      transparent: true,
      depthWrite: false,
    });

    // Volume box mesh
    const boxGeo = new THREE.BoxGeometry(vSize.x, vSize.y, vSize.z);
    this._volumeMesh = new THREE.Mesh(boxGeo, this._material);
    this._volumeMesh.position.copy(vCenter);
    this.group.add(this._volumeMesh);

    this.group.visible = true;
  }

  /**
   * Convenience: set from the terrain's moisture/chemistry arrays.
   * Pass the per-cell field data and the terrain dimensions.
   */
  setFromTerrainField(fieldData, gridW, gridH, color, terrainBounds) {
    const vMin = terrainBounds
      ? [terrainBounds.minX, terrainBounds.minY - 0.5, terrainBounds.minZ]
      : [-gridW / 2, -0.5, -gridH / 2];
    const vMax = terrainBounds
      ? [terrainBounds.maxX, terrainBounds.maxY + 1.0, terrainBounds.maxZ]
      : [gridW / 2, 1.5, gridH / 2];

    this.setConcentrationField('terrain_field', fieldData, gridW, gridH, 1, color, vMin, vMax);
  }

  animate(dt, camera) {
    if (!this.group.visible || !this._material) return;
    this._time += dt;
    this._material.uniforms.uTime.value = this._time;
    if (camera) {
      this._material.uniforms.uCameraPos.value.copy(camera.position);
    }
  }

  show() { this.group.visible = true; }
  hide() { this.group.visible = false; }

  _clear() {
    while (this.group.children.length) {
      const child = this.group.children[0];
      this.group.remove(child);
      if (child.geometry) child.geometry.dispose();
      if (child.material) {
        if (child.material.uniforms?.tDensityAtlas?.value) {
          child.material.uniforms.tDensityAtlas.value.dispose();
        }
        child.material.dispose();
      }
    }
    this._volumeMesh = null;
    this._material = null;
  }

  dispose() {
    this._clear();
    this.scene.remove(this.group);
  }
}
