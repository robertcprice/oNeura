// microcosm.js — Raymarched grass terrain adapted from Shadertoy dd2cWh
// Fullscreen raymarching shader: FBM terrain + SDF grass blades
// Backend soil chemistry drives grass color, moisture, lighting.

(function () {
  'use strict';

  var RECONNECT_MS = 2000;
  var renderer, scene, orthoCamera, perspCamera, controls, clock;
  var quadMesh, rmMat; // raymarching material
  var ws, gridW = 0, gridH = 0;
  var texSoil, texMoist, texWater;

  // ═══════════════════════════════════════════════════════════════
  // Raymarching shader — adapted from Shadertoy dd2cWh (MonterMan)
  // ═══════════════════════════════════════════════════════════════

  var VERT = /* glsl */ `
    void main() {
      gl_Position = vec4(position.xy, 0.0, 1.0);
    }
  `;

  var FRAG = /* glsl */ `
    precision highp float;

    uniform vec2 uResolution;
    uniform float uTime;
    uniform vec3 uCameraPos;
    uniform vec3 uCameraTarget;
    uniform vec3 uSunDir;
    uniform float uDaylight;
    uniform vec3 uSunColor;
    uniform sampler2D uSoilColor;
    uniform sampler2D uMoisture;
    uniform sampler2D uWater;

    // terrain patch bounds for UV mapping (larger to match 20x scale)
    const float PATCH_SIZE = 60.0;
    const float PATCH_HALF = 30.0;

    // ──────── Hashing (from Shadertoy) ────────
    float hash12(vec2 p) {
      vec3 p3 = fract(vec3(p.xyx) * .1031);
      p3 += dot(p3, p3.yzx + 33.33);
      return fract((p3.x + p3.y) * p3.z);
    }
    vec4 hash42(vec2 p) {
      vec4 p4 = fract(vec4(p.xyxy) * vec4(.1031, .1030, .0973, .1099));
      p4 += dot(p4, p4.wzxy + 33.33);
      return fract((p4.xxyz + p4.yzzw) * p4.zywx);
    }

    // ──────── Noise / FBM ────────
    float noise(vec2 p) {
      vec2 ip = floor(p);
      vec2 fp = fract(p);
      float a = hash12(ip);
      float b = hash12(ip + vec2(1, 0));
      float c = hash12(ip + vec2(0, 1));
      float d = hash12(ip + vec2(1, 1));
      vec2 t = smoothstep(0.0, 1.0, fp);
      return mix(mix(a, b, t.x), mix(c, d, t.x), t.y);
    }

    mat2 rot45 = mat2(0.7071, -0.7071, 0.7071, 0.7071);

    float fbm(vec2 p, int octaves) {
      float value = 0.0, amplitude = 0.5;
      for (int i = 0; i < 10; ++i) {
        if (i >= octaves) break;
        value += amplitude * noise(p);
        p = rot45 * p * 2.0 + 100.0;
        amplitude *= 0.5;
      }
      return value;
    }

    // ──────── Terrain (same scale as Shadertoy dd2cWh) ────────
    float calcTerrainHeight(vec2 p) {
      return 20.0 * fbm(0.02 * p, 10);
    }

    float approxTerrainHeight(vec2 p) {
      return 20.0 * (fbm(0.02 * p, 2) + 0.25);
    }

    vec3 calcTerrainNormal(vec2 p, float h) {
      vec2 e = vec2(0.0, 0.001);
      vec3 fw = vec3(0.0, calcTerrainHeight(p + e.xy) - h, e.y);
      vec3 rt = vec3(e.y, calcTerrainHeight(p + e.yx) - h, 0.0);
      return normalize(cross(fw, rt));
    }

    vec3 approxTerrainNormal(vec2 p) {
      vec2 e = vec2(0.0, 0.001);
      float h0 = approxTerrainHeight(p);
      vec3 fw = vec3(0.0, approxTerrainHeight(p + e.xy) - h0, e.y);
      vec3 rt = vec3(e.y, approxTerrainHeight(p + e.yx) - h0, 0.0);
      return normalize(cross(fw, rt));
    }

    // ──────── SDF helpers ────────
    float sdCircle(vec2 p, float r) { return length(p) - r; }

    float sdGrassBlade2d(vec2 p) {
      float dist = sdCircle(p - vec2(1.7, -1.3), 2.0);
      dist = max(dist, -sdCircle(p - vec2(1.7, -1.0), 1.8));
      dist = max(dist, p.y + 1.0);
      dist = max(dist, -p.x + 1.7);
      return dist;
    }

    float sdGrassBlade(vec3 p, float thickness) {
      p -= vec3(0.0, 1.0, 0.0);
      float dist2d = max(0.0, sdGrassBlade2d(p.xy));
      return sqrt(dist2d * dist2d + p.z * p.z) - thickness;
    }

    // ──────── Repeat ────────
    vec2 opRepeat(vec2 p, vec2 period, out vec2 id) {
      id = floor((p + 0.5 * period) / period);
      return mod(p + 0.5 * period, period) - 0.5 * period;
    }

    // ──────── Scene distance function ────────
    float map(vec3 p, inout float terrainH) {
      float distToTerrain = 0.5 * (p.y - terrainH);

      float guard = 1.1;
      if (distToTerrain > guard) {
        return distToTerrain - (guard - 1.0);
      }

      terrainH = calcTerrainHeight(p.xz);
      distToTerrain = 0.5 * (p.y - terrainH);

      vec2 grassId;
      float repeatPeriod = 0.25;
      p.xz = opRepeat(p.xz, vec2(repeatPeriod), grassId);

      float dist = 1e31;

      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          vec3 neighborP = p - vec3(float(dx), 0, float(dy)) * repeatPeriod;
          vec2 neighborId = grassId + vec2(float(dx), float(dy));
          float tH = terrainH;
          neighborP.y -= tH;

          vec4 rand = hash42(neighborId);
          float c = cos(rand.z * 6.28), s = sin(rand.z * 6.28);
          neighborP.xz = mat2(c, s, -s, c) * neighborP.xz;
          neighborP.xz += (rand.xy - 0.5) * repeatPeriod;

          dist = min(dist, sdGrassBlade(neighborP / sqrt(rand.w), 0.002));
        }
      }

      dist = min(dist, distToTerrain);
      return dist;
    }

    // ──────── Camera ────────
    mat3 calcCameraMat(vec3 from, vec3 at) {
      vec3 z = normalize(at - from);
      vec3 x = normalize(cross(vec3(0, 1, 0), z));
      vec3 y = cross(z, x);
      return mat3(x, y, z);
    }

    // ──────── Sky ────────
    vec3 getSkyCol(vec3 rd) {
      vec3 daySky = vec3(0.55, 0.65, 0.80);
      vec3 nightSky = vec3(0.02, 0.02, 0.05);
      vec3 skyBase = mix(nightSky, daySky, uDaylight);
      return mix(skyBase, 0.5 * skyBase, rd.y);
    }

    // ──────── UV for backend data sampling ────────
    vec2 terrainUV(vec2 p) {
      return clamp((p + PATCH_HALF) / PATCH_SIZE, 0.0, 1.0);
    }

    // ──────── Main ────────
    void main() {
      vec2 uv = gl_FragCoord.xy / uResolution;
      uv = 2.0 * uv - 1.0;
      uv.x *= uResolution.x / uResolution.y;

      vec2 pixelSize = 2.0 / uResolution;

      vec3 ro = uCameraPos;
      vec3 at = uCameraTarget;
      mat3 camMat = calcCameraMat(ro, at);
      float filmDist = 1.3;
      vec3 rd = normalize(camMat * vec3(uv, filmDist));

      vec3 skyCol = 1.1 * getSkyCol(rd);

      vec3 col = skyCol;
      bool hit = false;
      float t = 0.0;

      for (int i = 0; i < 400 && t < 500.0; ++i) {
        if (rd.y > 0.0 && t > 10.0) break;

        vec3 p = ro + t * rd;
        float terrainH = approxTerrainHeight(p.xz);
        float dist = map(p, terrainH);

        vec2 projPixel = pixelSize / filmDist * t;
        if (dist < 0.25 * projPixel.x) {
          hit = true;
          break;
        }
        t += dist;
      }

      if (hit) {
        vec3 p = ro + t * rd;
        vec2 sUv = terrainUV(p.xz);

        // Backend emergent soil color
        vec3 soilCol = texture2D(uSoilColor, sUv).rgb;
        float moisture = texture2D(uMoisture, sUv).r;
        float water = texture2D(uWater, sUv).r;

        // Grass color: blend between dry/lush based on moisture
        vec3 dryGrass = vec3(0.32, 0.30, 0.15);
        vec3 lushGrass = vec3(0.22, 0.38, 0.10);
        vec3 grassCol = mix(dryGrass, lushGrass, smoothstep(0.2, 0.6, moisture));

        // Subtle FBM color variation
        float colBlend = smoothstep(0.4, 0.3, fbm(0.2 * p.xz, 2));
        grassCol = mix(grassCol, grassCol * 0.85, colBlend);

        float terrainH = calcTerrainHeight(p.xz);
        float heightFromTerrain = p.y - terrainH;

        // Below grass → soil color; above → grass color
        float grassBlend = smoothstep(0.0, 0.15, heightFromTerrain);
        vec3 surfaceCol = mix(soilCol * 0.6, grassCol, grassBlend);

        // Water areas: suppress grass, show soil/water
        if (water > 0.5) {
          float waterBlend = smoothstep(0.5, 1.0, water);
          surfaceCol = mix(surfaceCol, soilCol * 0.3 + vec3(0.02, 0.04, 0.06), waterBlend * 0.7);
        }

        vec3 n = calcTerrainNormal(p.xz, terrainH);
        float directLight = max(0.0, dot(n, uSunDir));
        float ao = 0.3 + 0.7 * pow(max(heightFromTerrain, 0.001), 0.5);

        vec3 sampleCol = surfaceCol * (uSunColor * directLight + 0.12 * skyCol) * ao;

        // Fresnel rim (iq suggestion from original shader)
        vec3 fresnelCol = vec3(0.18, 0.20, 0.08) * ao;
        float fresnel = pow(clamp(1.0 + dot(approxTerrainNormal(p.xz), rd), 0.0, 1.0), 3.0);
        col = 0.74 * sampleCol + 0.82 * fresnelCol * fresnel;

        // Wet sheen
        if (moisture > 0.4) {
          vec3 H = normalize(uSunDir - rd);
          float spec = pow(max(dot(n, H), 0.0), 64.0);
          col += uSunColor * spec * moisture * uDaylight * 0.15;
        }
      } else {
        t = 1e5;
      }

      // Fog
      vec3 fogExp = -0.25 * vec3(0.008, 0.010, 0.008);
      col = mix(skyCol, col, exp(fogExp * t));

      // Night shift
      float night = smoothstep(0.3, 0.0, uDaylight);
      col = mix(col, col * vec3(0.5, 0.55, 0.85), night * 0.5);

      // Tone curve
      col = smoothstep(0.0, 1.0, col);

      gl_FragColor = vec4(col, 1.0);
    }
  `;

  // ═══════════════════════════════════════════════════════════════
  // Three.js setup
  // ═══════════════════════════════════════════════════════════════

  function init() {
    var el = document.getElementById('canvas-container');

    renderer = new THREE.WebGLRenderer({ antialias: false });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);

    // Orthographic camera renders the fullscreen quad
    scene = new THREE.Scene();
    orthoCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    // Perspective camera for OrbitControls interaction only
    perspCamera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 1000);
    // Camera ~5 units above terrain, looking slightly down and forward (matches Shadertoy)
    perspCamera.position.set(0.0, 18.0, -5.0);

    controls = new THREE.OrbitControls(perspCamera, renderer.domElement);
    controls.target.set(0.0, 14.0, 8.0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.minDistance = 2.0;
    controls.maxDistance = 100;
    controls.update();

    clock = new THREE.Clock();

    // Textures (placeholders until backend data arrives)
    texSoil  = makeTex3(8, 8, 0.30, 0.22, 0.13);
    texMoist = makeTex1(8, 8, 0.3);
    texWater = makeTex1(8, 8, 0.0);

    // Raymarching material
    rmMat = new THREE.ShaderMaterial({
      uniforms: {
        uResolution:   { value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
        uTime:         { value: 0 },
        uCameraPos:    { value: perspCamera.position.clone() },
        uCameraTarget: { value: controls.target.clone() },
        uSunDir:       { value: new THREE.Vector3(-0.5, 0.9, 0.0).normalize() },
        uDaylight:     { value: 0.8 },
        uSunColor:     { value: new THREE.Color(1.0, 0.95, 0.85) },
        uSoilColor:    { value: texSoil },
        uMoisture:     { value: texMoist },
        uWater:        { value: texWater },
      },
      vertexShader: VERT,
      fragmentShader: FRAG,
      depthWrite: false,
      depthTest: false,
    });

    // Fullscreen quad
    quadMesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), rmMat);
    quadMesh.frustumCulled = false;
    scene.add(quadMesh);

    window.addEventListener('resize', onResize);
    fetchSnapshot();
    connectWS();
    animate();
  }

  // ═══════════════════════════════════════════════════════════════
  // Textures
  // ═══════════════════════════════════════════════════════════════

  function makeTex3(w, h, r, g, b) {
    var d = new Float32Array(w * h * 4);
    for (var i = 0; i < w * h; i++) { d[i*4]=r; d[i*4+1]=g; d[i*4+2]=b; d[i*4+3]=1; }
    var t = new THREE.DataTexture(d, w, h, THREE.RGBAFormat, THREE.FloatType);
    t.magFilter = THREE.LinearFilter; t.minFilter = THREE.LinearFilter;
    t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping; t.needsUpdate = true; return t;
  }
  function makeTex1(w, h, v) { return makeTex3(w, h, v, v, v); }

  // ═══════════════════════════════════════════════════════════════
  // WebSocket
  // ═══════════════════════════════════════════════════════════════

  function connectWS() {
    var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(proto + '//' + location.host + '/ws');
    ws.binaryType = 'arraybuffer';
    ws.onopen = function() {
      document.getElementById('status').classList.add('hidden');
      ws.send(JSON.stringify({ cmd: "view", mode: "Terrain" }));
    };
    ws.onmessage = function(ev) {
      if (ev.data instanceof ArrayBuffer) parseBinaryFrame(ev.data);
    };
    ws.onclose = function() {
      document.getElementById('status').textContent = 'reconnecting...';
      document.getElementById('status').classList.remove('hidden');
      setTimeout(connectWS, RECONNECT_MS);
    };
    ws.onerror = function() { ws.close(); };
  }

  function parseBinaryFrame(buf) {
    var b = new Uint8Array(buf);
    if (b.length < 6 || b[0] !== 0x46 || b[1] !== 0x52) return;
    var w = b[2], h = b[3], pe = 4 + w * h * 2;
    if (b.length < pe + 1 || b[pe] !== 0) return;
    var meta;
    try { meta = JSON.parse(new TextDecoder().decode(b.slice(pe + 1))); } catch(_) { return; }
    meta.width = w; meta.height = h;
    onFrame(meta);
  }

  // ═══════════════════════════════════════════════════════════════
  // Frame handler
  // ═══════════════════════════════════════════════════════════════

  function onFrame(f) {
    var w = f.width, h = f.height;

    // Rebuild textures if grid changed
    if (w !== gridW || h !== gridH) {
      gridW = w; gridH = h;
      texSoil  = makeTex3(w, h, 0.30, 0.22, 0.13);
      texMoist = makeTex1(w, h, 0.3);
      texWater = makeTex1(w, h, 0.0);
      rmMat.uniforms.uSoilColor.value = texSoil;
      rmMat.uniforms.uMoisture.value = texMoist;
      rmMat.uniforms.uWater.value = texWater;
    }

    if (f.terrain_visuals && f.terrain_visuals.length === w * h) {
      var d = texSoil.image.data;
      for (var i = 0; i < w * h; i++) {
        var c = f.terrain_visuals[i].rgb;
        d[i*4] = c[0]; d[i*4+1] = c[1]; d[i*4+2] = c[2];
      }
      texSoil.needsUpdate = true;
    }
    if (f.moisture && f.moisture.length === w * h) {
      var d = texMoist.image.data;
      for (var i = 0; i < w * h; i++) d[i*4] = d[i*4+1] = d[i*4+2] = f.moisture[i];
      texMoist.needsUpdate = true;
    }
    if (f.water_mask && f.water_mask.length === w * h) {
      var d = texWater.image.data;
      for (var i = 0; i < w * h; i++) d[i*4] = d[i*4+1] = d[i*4+2] = f.water_mask[i];
      texWater.needsUpdate = true;
    }

    // Sun + daylight
    if (f.sun_direction) {
      rmMat.uniforms.uSunDir.value.set(f.sun_direction[0], f.sun_direction[1], f.sun_direction[2]);
    }
    if (f.daylight !== undefined) {
      rmMat.uniforms.uDaylight.value = f.daylight;
    }
    if (f.sun_elevation_rad !== undefined) {
      var warm = Math.max(0, 1.0 - f.sun_elevation_rad * 1.2);
      rmMat.uniforms.uSunColor.value.setRGB(1.0, 0.82 + warm * 0.12, 0.68 + warm * 0.18);
    }

    updateHUD(f);
  }

  // ═══════════════════════════════════════════════════════════════
  // HUD
  // ═══════════════════════════════════════════════════════════════

  function updateHUD(f) {
    if (f.time_label) setText('time-label', f.time_label);
    if (f.atmosphere) {
      var a = f.atmosphere;
      setText('hud-temp', arrMean(a.temperature_c).toFixed(1));
      setText('hud-humidity', (arrMean(a.humidity) * 100).toFixed(0));
      var wx = arrMean(a.wind_x), wz = arrMean(a.wind_z);
      setText('hud-wind', Math.sqrt(wx*wx + wz*wz).toFixed(2));
    }
    if (f.moisture) setText('hud-moisture', (arrMean(f.moisture) * 100).toFixed(0));
    if (f.daylight !== undefined) {
      var dl = f.daylight;
      setText('hud-daylight', dl > 0.8 ? 'day' : dl > 0.4 ? 'dusk' : dl > 0.05 ? 'twilight' : 'night');
    }
  }

  function fetchSnapshot() {
    function go() { fetch('/api/snapshot').then(function(r){return r.json();}).then(updateChemHUD).catch(function(){}); }
    go(); setInterval(go, 5000);
  }
  function updateChemHUD(s) {
    var L = [];
    function a(l,v) { if(v!=null) L.push('<span class="'+(v>0.001?'active':'')+'">'+ l+' '+v.toFixed(3)+'</span>'); }
    a('glucose',s.mean_soil_glucose); a('O\u2082',s.mean_soil_oxygen);
    a('NH\u2084\u207A',s.mean_soil_ammonium); a('NO\u2083\u207B',s.mean_soil_nitrate);
    a('Fe\u00B2\u207A',s.mean_soil_aqueous_iron); a('DSi',s.mean_soil_dissolved_silicate);
    a('HCO\u2083\u207B',s.mean_soil_bicarbonate); a('Ca\u00B2\u207A',s.mean_soil_exchangeable_calcium);
    document.getElementById('chem-details').innerHTML = L.join('<br>');
  }

  // ═══════════════════════════════════════════════════════════════
  // Render loop
  // ═══════════════════════════════════════════════════════════════

  function animate() {
    requestAnimationFrame(animate);
    controls.update();

    // Feed camera to shader
    rmMat.uniforms.uCameraPos.value.copy(perspCamera.position);
    rmMat.uniforms.uCameraTarget.value.copy(controls.target);
    rmMat.uniforms.uTime.value = clock.getElapsedTime();

    // Render fullscreen quad with ortho camera
    renderer.render(scene, orthoCamera);
  }

  // ═══════════════════════════════════════════════════════════════
  // Utilities
  // ═══════════════════════════════════════════════════════════════

  function onResize() {
    var w = window.innerWidth, h = window.innerHeight;
    renderer.setSize(w, h);
    perspCamera.aspect = w / h;
    perspCamera.updateProjectionMatrix();
    rmMat.uniforms.uResolution.value.set(w, h);
  }
  function arrMean(a) { if(!a||!a.length) return 0; var s=0; for(var i=0;i<a.length;i++) s+=a[i]; return s/a.length; }
  function setText(id, t) { var e = document.getElementById(id); if(e) e.textContent = t; }

  init();
})();
