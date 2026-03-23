/**
 * ShaderTerrain — Full-screen GLSL terrain renderer.
 *
 * Raymarched procedural terrain with grass, trees, water, and atmospheric
 * lighting. Runs on a separate WebGL canvas behind the vtk.js layer.
 * Fed by simulation data from the Rust backend.
 *
 * Inspired by Shadertoy terrain shaders (dd2cWh style).
 */

const VERT_SHADER = `
attribute vec2 position;
void main() {
  gl_Position = vec4(position, 0.0, 1.0);
}
`;

const FRAG_SHADER = `
precision highp float;

uniform float iTime;
uniform vec2 iResolution;
uniform vec3 uCameraPos;
uniform vec3 uCameraTarget;
uniform float uDaylight;
uniform vec3 uSunDir;
uniform float uMoisture;
uniform float uCloudCover;

// Terrain data textures (8x8 grid)
uniform sampler2D uHeightMap;    // terrain_surface
uniform sampler2D uColorMap;     // terrain_visuals rgb
uniform sampler2D uMoistureMap;  // moisture per cell

// Plant positions (up to 8 plants)
uniform vec4 uPlants[8];        // xyz = position, w = height
uniform vec3 uPlantColors[8];   // leaf RGB
uniform int uPlantCount;

// ===== Noise functions =====

float hash(float n) { return fract(sin(n) * 43758.5453123); }
float hash2(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

float noise(vec3 x) {
  vec3 p = floor(x);
  vec3 f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  float n = p.x + p.y * 57.0 + p.z * 113.0;
  return mix(
    mix(mix(hash(n), hash(n + 1.0), f.x),
        mix(hash(n + 57.0), hash(n + 58.0), f.x), f.y),
    mix(mix(hash(n + 113.0), hash(n + 114.0), f.x),
        mix(hash(n + 170.0), hash(n + 171.0), f.x), f.y),
    f.z);
}

float noise2(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float a = hash2(i);
  float b = hash2(i + vec2(1.0, 0.0));
  float c = hash2(i + vec2(0.0, 1.0));
  float d = hash2(i + vec2(1.0, 1.0));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
  float v = 0.0;
  float a = 0.5;
  vec2 shift = vec2(100.0);
  for (int i = 0; i < 3; i++) {
    v += a * noise2(p);
    p = p * 2.0 + shift;
    a *= 0.5;
  }
  return v;
}

// ===== Terrain height =====

// Smooth bicubic-ish sampling to hide grid artifacts on 8x8 textures
vec4 texSmooth(sampler2D tex, vec2 uv) {
  vec2 res = vec2(8.0);
  vec2 st = uv * res - 0.5;
  vec2 iuv = floor(st);
  vec2 fuv = fract(st);
  // Smoothstep interpolation instead of bilinear — removes grid artifacts
  fuv = fuv * fuv * (3.0 - 2.0 * fuv);
  vec2 uv00 = (iuv + 0.5) / res;
  vec2 uv10 = (iuv + vec2(1.0, 0.0) + 0.5) / res;
  vec2 uv01 = (iuv + vec2(0.0, 1.0) + 0.5) / res;
  vec2 uv11 = (iuv + vec2(1.0, 1.0) + 0.5) / res;
  return mix(
    mix(texture2D(tex, uv00), texture2D(tex, uv10), fuv.x),
    mix(texture2D(tex, uv01), texture2D(tex, uv11), fuv.x),
    fuv.y);
}

float terrainHeight(vec2 p) {
  vec2 uv = p / 8.0;
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    float edge = max(0.0, 1.0 - length(uv - 0.5) * 1.5);
    return edge * 0.3 + noise2(p * 0.5) * 0.08;
  }

  float baseH = texSmooth(uHeightMap, uv).r;
  float detail = noise2(p * 3.0) * 0.05 + noise2(p * 10.0) * 0.012;

  return (baseH - 0.3) * 3.5 + detail;
}

vec3 terrainColor(vec2 p) {
  vec2 uv = p / 8.0;
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    // Outside: natural earth with fbm variation
    float n = fbm(p * 1.5);
    return vec3(0.4 + n*0.1, 0.32 + n*0.08, 0.2 + n*0.05);
  }
  vec3 col = texSmooth(uColorMap, uv).rgb;
  // If data not loaded yet, use warm earth default
  if (col.r < 0.01 && col.g < 0.01 && col.b < 0.01) {
    col = vec3(0.55, 0.42, 0.28);
  }
  // Aggressively break up grid with multi-scale noise
  float n1 = noise2(p * 3.0);
  float n2 = noise2(p * 8.0);
  float n3 = noise2(p * 18.0);
  // Mix backend color with procedural variation to mask the 8x8 grid
  vec3 procCol = vec3(0.55 + n1*0.12, 0.42 + n1*0.08, 0.28 + n1*0.06);
  col = mix(col, procCol, 0.35); // blend 35% procedural to break grid
  col *= 0.85 + n2 * 0.18 + n3 * 0.06;
  col.r += n2 * 0.035;
  col.b -= n3 * 0.02;
  return col;
}

float terrainMoisture(vec2 p) {
  vec2 uv = p / 8.0;
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return 0.2;
  return texSmooth(uMoistureMap, uv).r;
}

// ===== Terrain normal =====

vec3 terrainNormal(vec2 p) {
  float eps = 0.02;
  float h = terrainHeight(p);
  float hx = terrainHeight(p + vec2(eps, 0.0));
  float hy = terrainHeight(p + vec2(0.0, eps));
  return normalize(vec3(h - hx, eps, h - hy));
}

// ===== Grass =====

float grassDensity(vec2 p) {
  float m = terrainMoisture(p);
  // Multi-scale noise for clumpy, natural grass distribution
  float clump = noise2(p * 5.0); // large patches
  float detail = noise2(p * 20.0); // individual blade variation
  float blade = noise2(p * 50.0); // fine detail
  // Flat areas get more grass
  vec3 norm = terrainNormal(p);
  float slope = norm.y;
  // Grass needs actual moisture to grow
  float density = smoothstep(0.15, 0.5, m) * slope;
  density *= smoothstep(0.3, 0.55, clump); // clumpy patches
  density *= 0.5 + detail * 0.5;
  return clamp(density, 0.0, 1.0);
}

vec3 grassColor(vec2 p, float density) {
  float n = noise2(p * 30.0);
  float n2 = noise2(p * 60.0);
  float m = terrainMoisture(p);
  // Rich saturated green palette
  vec3 lush = vec3(0.08, 0.38, 0.04);      // deep forest green
  vec3 bright = vec3(0.18, 0.52, 0.08);    // bright spring green
  vec3 yellow = vec3(0.38, 0.40, 0.1);     // dry meadow
  vec3 dark = vec3(0.05, 0.22, 0.03);      // shadow green
  // Moisture drives lush/dry balance
  float moist = clamp(m * 2.0, 0.0, 1.0);
  vec3 base = mix(yellow, lush, moist);
  base = mix(base, bright, n * 0.35);
  base = mix(base, dark, (1.0 - n2) * 0.15);
  base *= 0.95 + n2 * 0.2;
  return base;
}

// ===== Trees (SDF) =====

float sdSphere(vec3 p, float r) { return length(p) - r; }
float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
  vec3 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - r;
}

// Smooth minimum for organic blending (from user's GLKITTY reference)
float smin(float a, float b, float k) {
  float h = clamp(0.5 + 0.5*(b-a)/k, 0.0, 1.0);
  return mix(b, a, h) - k*h*(1.0-h);
}

float treeSDF(vec3 p, vec3 treePos, float treeH) {
  vec3 lp = p - treePos;

  // Trunk — tapered capsule
  float trunkH = treeH * 0.55;
  float trunkR = treeH * 0.035;
  float trunk = sdCapsule(lp, vec3(0, -0.05, 0), vec3(0, trunkH, 0), trunkR);

  // Main canopy — noise-displaced sphere
  vec3 c0 = vec3(0, trunkH + treeH * 0.22, 0);
  float r0 = treeH * 0.35;
  float canopy = sdSphere(lp - c0, r0);
  // Use cheaper noise for displacement
  float disp = noise(lp * 3.5 / treeH) * 0.6 + noise(lp * 7.0 / treeH) * 0.2;
  canopy += disp * r0 * 0.5 - r0 * 0.15;

  // One secondary lobe for asymmetry
  vec3 c1 = c0 + vec3(r0*0.35, r0*0.1, r0*0.25);
  float lobe1 = sdSphere(lp - c1, r0 * 0.55);
  lobe1 += noise(lp * 4.0 / treeH + 17.0) * r0 * 0.35;

  // Blend with smooth minimum
  float fullCanopy = smin(canopy, lobe1, r0 * 0.35);

  return min(trunk, fullCanopy);
}

float treeField(vec3 p) {
  float d = 100.0;
  for (int i = 0; i < 8; i++) {
    if (i >= uPlantCount) break;
    vec3 tpos = uPlants[i].xyz;
    float th = uPlants[i].w;
    // Quick bounding sphere rejection — skip full SDF if too far
    float boundDist = length(p.xz - tpos.xz) - th * 0.6;
    if (boundDist > 2.0) continue;
    tpos.y = terrainHeight(tpos.xz);
    d = min(d, treeSDF(p, tpos, th));
  }
  return d;
}

vec3 treeColor(vec3 p, vec3 n) {
  for (int i = 0; i < 8; i++) {
    if (i >= uPlantCount) break;
    vec3 tpos = uPlants[i].xyz;
    float th = uPlants[i].w;
    tpos.y = terrainHeight(tpos.xz);
    float d = treeSDF(p, tpos, th);
    if (d < 0.15) {
      vec3 lp = p - tpos;
      float trunkH = th * 0.55;

      if (lp.y < trunkH * 0.7 && length(lp.xz) < th * 0.06) {
        // Bark — warm brown with vertical grain
        float grain = noise(vec3(lp.x * 20.0, lp.y * 3.0, lp.z * 20.0));
        return vec3(0.35 + grain*0.1, 0.22 + grain*0.06, 0.12 + grain*0.04);
      }

      // Leaf color with variation
      vec3 leafCol = uPlantColors[i];
      // If no data, use rich green
      if (leafCol.r < 0.01 && leafCol.g < 0.01) leafCol = vec3(0.15, 0.4, 0.08);
      // Noise-based leaf variation — light/dark patches
      float leafNoise = noise(p * 8.0);
      float leafDetail = noise(p * 20.0);
      leafCol *= 0.7 + leafNoise * 0.4 + leafDetail * 0.15;
      // Subsurface scattering — leaves glow when backlit
      float sss = max(dot(-n, uSunDir), 0.0);
      leafCol += vec3(0.1, 0.15, 0.02) * sss * sss * uDaylight;
      return leafCol;
    }
  }
  return vec3(0.15, 0.35, 0.08);
}

// ===== Raymarching =====
// Returns: hitT in out_hit.x, material in out_hit.y (0=sky,1=terrain,2=tree)
// hitPos in out_pos

vec2 raymarch(vec3 ro, vec3 rd, out vec3 out_pos) {
  float t = 0.0;
  out_pos = ro;

  for (int i = 0; i < 64; i++) {
    vec3 p = ro + rd * t;
    float terrainH = terrainHeight(p.xz);
    float groundDist = p.y - terrainH;

    if (groundDist < 0.02) {
      out_pos = p;
      return vec2(t, 1.0); // terrain
    }

    // Only check trees when close to terrain level (perf optimization)
    float treeDist = 100.0;
    if (groundDist < 3.0) {
      treeDist = treeField(p);
      if (treeDist < 0.03) {
        out_pos = p;
        return vec2(t, 2.0); // tree
      }
    }

    float d = min(groundDist, treeDist);
    t += max(d * 0.6, 0.02);
    if (t > 40.0) break;
  }

  return vec2(t, 0.0); // sky
}

// ===== Sky =====

vec3 sky(vec3 rd) {
  float sunAmount = max(dot(rd, uSunDir), 0.0);

  // Rayleigh scattering — deeper blue at zenith, paler at horizon
  vec3 zenith = vec3(0.2, 0.4, 0.85) * uDaylight;
  vec3 horizonCol = vec3(0.6, 0.7, 0.85) * uDaylight;
  float horizonFade = pow(1.0 - max(rd.y, 0.0), 3.0);
  vec3 skyCol = mix(zenith, horizonCol, horizonFade);

  // Sun disc and glow
  skyCol += vec3(1.0, 0.85, 0.5) * pow(sunAmount, 6.0) * 0.6;
  skyCol += vec3(1.0, 0.95, 0.85) * pow(sunAmount, 128.0) * 3.0;
  // Warm haze around sun
  skyCol += vec3(0.3, 0.15, 0.05) * pow(sunAmount, 3.0) * 0.4;

  // Clouds — single layer for performance
  if (rd.y > 0.02 && uCloudCover > 0.1) {
    vec2 cUV = rd.xz / rd.y * 2.5 + iTime * 0.015;
    float cloud = noise2(cUV) * 0.6 + noise2(cUV * 2.5) * 0.3;
    cloud = smoothstep(0.85 - uCloudCover * 0.4, 1.0, cloud);
    // Clouds lit by sun
    vec3 cloudCol = mix(vec3(0.85, 0.85, 0.9), vec3(1.0, 0.98, 0.95), sunAmount) * uDaylight;
    // Cloud shadow on underside
    cloudCol *= 0.8 + 0.2 * max(rd.y, 0.0);
    skyCol = mix(skyCol, cloudCol, cloud * 0.8);
  }

  // Night
  if (uDaylight < 0.15) {
    vec3 nightSky = vec3(0.015, 0.015, 0.05);
    // Stars
    float star = smoothstep(0.98, 0.99, noise2(rd.xz * 500.0));
    nightSky += vec3(0.8, 0.85, 1.0) * star * (1.0 - uDaylight / 0.15);
    skyCol = mix(nightSky, skyCol, clamp(uDaylight / 0.15, 0.0, 1.0));
  }

  return skyCol;
}

// ===== Main =====

void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * iResolution) / iResolution.y;

  // Camera
  vec3 ro = uCameraPos;
  vec3 ta = uCameraTarget;
  vec3 ww = normalize(ta - ro);
  vec3 uu = normalize(cross(ww, vec3(0, 1, 0)));
  vec3 vv = cross(uu, ww);
  vec3 rd = normalize(uv.x * uu + uv.y * vv + 1.5 * ww);

  // Raymarch
  vec3 hitPos;
  vec2 hit = raymarch(ro, rd, hitPos);
  float hitT = hit.x;
  float material = hit.y;

  vec3 col;

  if (material < 0.5) {
    // Sky
    col = sky(rd);
  } else {
    vec3 p = hitPos;
    vec3 n;

    if (material < 1.5) {
      // Terrain
      n = terrainNormal(p.xz);

      vec3 baseCol = terrainColor(p.xz);

      // Grass overlay — strong green coverage
      float gd = grassDensity(p.xz);
      vec3 gc = grassColor(p.xz, gd);
      // Grass dominates on flat moist areas
      baseCol = mix(baseCol, gc, gd * 0.9);
      // Subtle moss only where actually moist
      float moss = smoothstep(0.3, 0.6, terrainMoisture(p.xz)) * 0.08;
      baseCol = mix(baseCol, vec3(0.25, 0.35, 0.12), moss * (1.0 - gd));

      col = baseCol;
    } else {
      // Tree — approximate normal from noise gradient (cheaper than 3 extra SDF evals)
      float eps = 0.05;
      float nx = noise(p * 3.0 + vec3(eps,0,0)) - noise(p * 3.0 - vec3(eps,0,0));
      float ny = noise(p * 3.0 + vec3(0,eps,0)) - noise(p * 3.0 - vec3(0,eps,0));
      float nz = noise(p * 3.0 + vec3(0,0,eps)) - noise(p * 3.0 - vec3(0,0,eps));
      n = normalize(vec3(nx, ny + 0.5, nz)); // bias upward for leaves
      col = treeColor(p, n);
    }

    // Lighting — rich multi-source
    float diff = max(dot(n, uSunDir), 0.0) * uDaylight;

    // Sky ambient — hemisphere lighting (blue from above, warm from below)
    float skyAmt = 0.5 + 0.5 * n.y;
    vec3 ambCol = mix(vec3(0.1, 0.08, 0.06), vec3(0.2, 0.28, 0.4), skyAmt) * uDaylight;
    ambCol += vec3(0.05); // minimum base

    // Soft shadow (terrain only, skip trees for performance)
    float shadow = 1.0;
    if (material < 1.5) {
      vec3 shadowP = p + n * 0.1;
      float shadowT = 0.1;
      for (int j = 0; j < 8; j++) {
        vec3 sp = shadowP + uSunDir * shadowT;
        float sh = sp.y - terrainHeight(sp.xz);
        shadow = min(shadow, 8.0 * sh / shadowT);
        shadowT += 0.25;
        if (shadow < 0.01) break;
      }
      shadow = clamp(shadow, 0.15, 1.0);
    }

    // Bounce light from terrain (warm)
    float bounce = max(-n.y, 0.0) * 0.15;
    vec3 bounceCol = vec3(0.4, 0.3, 0.2) * bounce * uDaylight;

    // Back/rim light
    float rim = pow(1.0 - max(dot(n, normalize(uCameraPos - p)), 0.0), 3.0);
    vec3 rimCol = vec3(0.2, 0.25, 0.3) * rim * 0.3 * uDaylight;

    // Combine — make sun strong so terrain has clear light/shadow
    vec3 sunCol = vec3(1.1, 1.0, 0.85);
    col = col * (ambCol + sunCol * diff * shadow * 1.2 + bounceCol + rimCol);

    // Atmospheric fog — warm near horizon
    float fog = 1.0 - exp(-hitT * 0.025);
    vec3 fogCol = mix(sky(rd), vec3(0.7, 0.72, 0.75) * uDaylight, 0.3);
    col = mix(col, fogCol, fog);
  }

  // Increase contrast, not brightness
  col *= 1.1;

  // Simple Reinhard with white point
  float whitePoint = 2.0;
  col = col * (1.0 + col / (whitePoint * whitePoint)) / (1.0 + col);

  // Warm tint
  col.r *= 1.04;
  col.b *= 0.94;

  // Boost saturation
  float lum = dot(col, vec3(0.299, 0.587, 0.114));
  col = mix(vec3(lum), col, 1.3); // 30% more saturation

  // Contrast curve (S-curve)
  col = smoothstep(-0.02, 1.02, col);

  // Gamma
  col = pow(col, vec3(0.48));

  // Subtle vignette
  vec2 vUV = gl_FragCoord.xy / iResolution;
  float vig = vUV.x * vUV.y * (1.0 - vUV.x) * (1.0 - vUV.y);
  vig = clamp(pow(vig * 16.0, 0.2), 0.0, 1.0);
  col *= 0.7 + 0.3 * vig;

  gl_FragColor = vec4(col, 1.0);
}
`;

export class ShaderTerrain {
  constructor(container) {
    // Primary full-screen canvas — this IS the renderer
    this.canvas = document.createElement('canvas');
    this.canvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;z-index:0;cursor:grab;';
    container.style.position = 'relative';
    container.appendChild(this.canvas);

    this.gl = this.canvas.getContext('webgl', { antialias: true, alpha: false });
    if (!this.gl) {
      console.error('ShaderTerrain: WebGL not available');
      return;
    }

    this._initShader();
    this._initGeometry();
    this._initTextures();

    // Uniforms state
    this.time = 0;
    this.daylight = 0.6;
    this.sunDir = [0.5, 0.7, 0.3];
    this.moisture = 0.3;
    this.cloudCover = 0.3;
    this.plants = [];
    this.plantColors = [];

    // Orbit camera state (Y-up coordinate system for shader)
    this.camAngleX = 0.6;   // horizontal orbit angle
    this.camAngleY = 0.45;  // vertical angle (0=horizon, PI/2=top-down)
    this.camDist = 10.0;    // distance from target
    this.targetCamDist = 10.0;
    this.camTarget = [4.0, 0.3, 4.0]; // center of terrain (Y-up)
    this._updateCameraFromOrbit();

    // Mouse interaction
    this._isDragging = false;
    this._lastMouse = [0, 0];
    this._setupMouseControls();

    // Start render loop
    this._animFrame = null;
    this._startTime = performance.now();
    this._animate();
  }

  _setupMouseControls() {
    this._dragTotal = 0;

    this.canvas.addEventListener('mousedown', (e) => {
      if (e.button === 0) {
        this._isDragging = true;
        this._dragTotal = 0;
        this._lastMouse = [e.clientX, e.clientY];
        this.canvas.style.cursor = 'grabbing';
        e.preventDefault();
      }
    });
    window.addEventListener('mouseup', () => {
      if (this._isDragging) {
        this._isDragging = false;
        this.canvas.style.cursor = 'grab';
      }
    });
    window.addEventListener('mousemove', (e) => {
      if (!this._isDragging) return;
      const dx = e.clientX - this._lastMouse[0];
      const dy = e.clientY - this._lastMouse[1];
      this._dragTotal += Math.abs(dx) + Math.abs(dy);
      this._lastMouse = [e.clientX, e.clientY];
      this.camAngleX -= dx * 0.008;
      this.camAngleY = Math.max(0.05, Math.min(1.45, this.camAngleY + dy * 0.008));
      this._updateCameraFromOrbit();
    });
    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.targetCamDist *= (1 + e.deltaY * 0.002);
      this.targetCamDist = Math.max(2, Math.min(25, this.targetCamDist));
    }, { passive: false });
    // Touch support
    this.canvas.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        this._isDragging = true;
        this._dragTotal = 0;
        this._lastMouse = [e.touches[0].clientX, e.touches[0].clientY];
        e.preventDefault();
      }
    }, { passive: false });
    this.canvas.addEventListener('touchend', () => { this._isDragging = false; });
    this.canvas.addEventListener('touchmove', (e) => {
      if (!this._isDragging || e.touches.length !== 1) return;
      e.preventDefault();
      const dx = e.touches[0].clientX - this._lastMouse[0];
      const dy = e.touches[0].clientY - this._lastMouse[1];
      this._dragTotal += Math.abs(dx) + Math.abs(dy);
      this._lastMouse = [e.touches[0].clientX, e.touches[0].clientY];
      this.camAngleX -= dx * 0.008;
      this.camAngleY = Math.max(0.05, Math.min(1.45, this.camAngleY + dy * 0.008));
      this._updateCameraFromOrbit();
    }, { passive: false });
  }

  /** Returns true if the last mouse interaction was a drag, false if it was a click */
  wasDrag() { return this._dragTotal > 5; }

  _updateCameraFromOrbit() {
    // Smooth zoom
    this.camDist += (this.targetCamDist - this.camDist) * 0.1;
    // Spherical to cartesian (Y-up)
    const x = this.camTarget[0] + this.camDist * Math.cos(this.camAngleY) * Math.sin(this.camAngleX);
    const y = this.camTarget[1] + this.camDist * Math.sin(this.camAngleY);
    const z = this.camTarget[2] + this.camDist * Math.cos(this.camAngleY) * Math.cos(this.camAngleX);
    this.cameraPos = [x, y, z];
    this.cameraTarget = [...this.camTarget];
  }

  _initShader() {
    const gl = this.gl;

    const vs = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vs, VERT_SHADER);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      console.error('Vertex shader:', gl.getShaderInfoLog(vs));
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fs, FRAG_SHADER);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      console.error('Fragment shader:', gl.getShaderInfoLog(fs));
      // Show the error line
      const log = gl.getShaderInfoLog(fs);
      console.error(log);
    }

    this.program = gl.createProgram();
    gl.attachShader(this.program, vs);
    gl.attachShader(this.program, fs);
    gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      console.error('Program link:', gl.getProgramInfoLog(this.program));
    }

    // Cache uniform locations
    this.uniforms = {};
    const names = [
      'iTime', 'iResolution', 'uCameraPos', 'uCameraTarget',
      'uDaylight', 'uSunDir', 'uMoisture', 'uCloudCover',
      'uHeightMap', 'uColorMap', 'uMoistureMap', 'uPlantCount',
    ];
    for (const n of names) {
      this.uniforms[n] = gl.getUniformLocation(this.program, n);
    }
    // Plant uniforms
    for (let i = 0; i < 8; i++) {
      this.uniforms[`uPlants[${i}]`] = gl.getUniformLocation(this.program, `uPlants[${i}]`);
      this.uniforms[`uPlantColors[${i}]`] = gl.getUniformLocation(this.program, `uPlantColors[${i}]`);
    }
  }

  _initGeometry() {
    const gl = this.gl;
    // Full-screen quad
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1, 1, -1, -1, 1, 1, 1,
    ]), gl.STATIC_DRAW);
    this.quadBuffer = buffer;
  }

  _initTextures() {
    const gl = this.gl;
    // Create 8x8 data textures for terrain
    this.heightTex = this._createDataTexture(8, 8);
    this.colorTex = this._createDataTexture(8, 8);
    this.moistureTex = this._createDataTexture(8, 8);
  }

  _createDataTexture(w, h) {
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const data = new Uint8Array(w * h * 4);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    return tex;
  }

  // === Data updates ===

  updateFromFrame(meta) {
    if (!meta) return;
    const gl = this.gl;
    if (!gl) return;

    const w = 8, h = 8;

    // Height map
    if (meta.terrain_surface) {
      const data = new Uint8Array(w * h * 4);
      for (let i = 0; i < w * h; i++) {
        const v = Math.round((meta.terrain_surface[i] || 0) * 255);
        data[i * 4] = v;
        data[i * 4 + 1] = v;
        data[i * 4 + 2] = v;
        data[i * 4 + 3] = 255;
      }
      gl.bindTexture(gl.TEXTURE_2D, this.heightTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    }

    // Color map
    if (meta.terrain_visuals) {
      const data = new Uint8Array(w * h * 4);
      for (let i = 0; i < w * h; i++) {
        const v = meta.terrain_visuals[i];
        if (v && v.rgb) {
          data[i * 4] = Math.round(v.rgb[0] * 255);
          data[i * 4 + 1] = Math.round(v.rgb[1] * 255);
          data[i * 4 + 2] = Math.round(v.rgb[2] * 255);
        } else {
          data[i * 4] = 180; data[i * 4 + 1] = 140; data[i * 4 + 2] = 90;
        }
        data[i * 4 + 3] = 255;
      }
      gl.bindTexture(gl.TEXTURE_2D, this.colorTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    }

    // Moisture map
    if (meta.moisture) {
      const data = new Uint8Array(w * h * 4);
      for (let i = 0; i < w * h; i++) {
        const v = Math.round((meta.moisture[i] || 0) * 255);
        data[i * 4] = v; data[i * 4 + 1] = v; data[i * 4 + 2] = v; data[i * 4 + 3] = 255;
      }
      gl.bindTexture(gl.TEXTURE_2D, this.moistureTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    }

    // Atmosphere
    this.daylight = meta.daylight ?? 0.6;
    this.sunDir = meta.sun_direction ? [...meta.sun_direction] : [0.5, 0.7, 0.3];
    this.cloudCover = meta.cloud_cover ?? 0.3;

    if (meta.atmosphere && meta.atmosphere.humidity) {
      const hum = meta.atmosphere.humidity;
      let s = 0;
      for (let i = 0; i < hum.length; i++) s += hum[i];
      this.moisture = s / hum.length;
    }
  }

  updatePlants(entitiesMsg) {
    if (!entitiesMsg) return;
    const plants = entitiesMsg.full_plants || [];
    const visuals = entitiesMsg.plant_visuals || [];
    this.plants = [];
    this.plantColors = [];

    for (let i = 0; i < Math.min(8, plants.length); i++) {
      const p = plants[i];
      const h = Math.max(1.2, (p.height_mm || 10) / 10.0);
      // Note: in shader Y is up, so plant pos = (x+0.5, 0, y+0.5)
      this.plants.push([p.x + 0.5, 0, p.y + 0.5, h]);

      const vis = visuals[i];
      if (vis && vis.leaf_rgb) {
        this.plantColors.push(vis.leaf_rgb);
      } else {
        this.plantColors.push([0.15, 0.4, 0.1]);
      }
    }
  }

  // Camera is managed by orbit controls, not externally

  // === Render loop ===

  _animate() {
    this._updateCameraFromOrbit();
    this._render();
    this._animFrame = requestAnimationFrame(() => this._animate());
  }

  _render() {
    const gl = this.gl;
    if (!gl) return;

    // Resize canvas — render at HALF resolution for performance
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5) * 0.6;
    const w = Math.round(this.canvas.clientWidth * dpr);
    const h = Math.round(this.canvas.clientHeight * dpr);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
      gl.viewport(0, 0, w, h);
    }

    this.time = (performance.now() - this._startTime) / 1000;

    gl.useProgram(this.program);

    // Set uniforms
    gl.uniform1f(this.uniforms.iTime, this.time);
    gl.uniform2f(this.uniforms.iResolution, w, h);
    gl.uniform3fv(this.uniforms.uCameraPos, this.cameraPos);
    gl.uniform3fv(this.uniforms.uCameraTarget, this.cameraTarget);
    gl.uniform1f(this.uniforms.uDaylight, this.daylight);
    gl.uniform3fv(this.uniforms.uSunDir, this.sunDir);
    gl.uniform1f(this.uniforms.uMoisture, this.moisture);
    gl.uniform1f(this.uniforms.uCloudCover, this.cloudCover);
    gl.uniform1i(this.uniforms.uPlantCount, this.plants.length);

    // Plant uniforms
    for (let i = 0; i < 8; i++) {
      if (i < this.plants.length) {
        gl.uniform4fv(this.uniforms[`uPlants[${i}]`], this.plants[i]);
        gl.uniform3fv(this.uniforms[`uPlantColors[${i}]`], this.plantColors[i]);
      }
    }

    // Bind textures
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.heightTex);
    gl.uniform1i(this.uniforms.uHeightMap, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.colorTex);
    gl.uniform1i(this.uniforms.uColorMap, 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.moistureTex);
    gl.uniform1i(this.uniforms.uMoistureMap, 2);

    // Draw full-screen quad
    const posLoc = gl.getAttribLocation(this.program, 'position');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  destroy() {
    if (this._animFrame) cancelAnimationFrame(this._animFrame);
    if (this.canvas.parentElement) this.canvas.parentElement.removeChild(this.canvas);
  }
}
