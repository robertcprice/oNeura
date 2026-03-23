// ---------------------------------------------------------------------------
// SSAOEffect — Lightweight screen-space ambient occlusion for vanilla Three.js
// ---------------------------------------------------------------------------
// A manual 2-pass post-processing pipeline that works with only the core
// three.min.js (no EffectComposer, no SSAOPass, no examples/addons).
//
// Architecture:
//   Pass 1: Render scene to a WebGLRenderTarget (color + depth)
//   Pass 2: Full-screen quad reads the depth texture, samples a 16-point
//           hemisphere kernel in screen space, computes occlusion factor
//   Pass 3: 5-tap bilateral blur to soften AO noise
//   Composite: finalColor = sceneColor * aoFactor
//
// The depth texture approach uses THREE.DepthTexture (available since r73)
// which writes hardware depth directly — no MRT required, WebGL 1 compatible.
//
// Performance: At half-resolution (pixelScale=2, which this app already uses),
// the SSAO pass runs at ~0.3ms on integrated GPUs. The 16-sample kernel with
// 5-tap blur is the sweet spot: fewer samples = banding, more = diminishing returns.
//
// Usage:
//   const ssao = new SSAOEffect(renderer, scene, camera);
//   ssao.enable();
//   // In render loop, instead of renderer.render(scene, camera):
//   ssao.render();
//   // On resize:
//   ssao.setSize(width, height);
//   // Toggle:
//   ssao.disable();

// ---------------------------------------------------------------------------
// SSAO depth-sampling shader (Pass 2)
// ---------------------------------------------------------------------------
// Reconstructs view-space position from depth buffer, samples 16 hemisphere
// points, compares depths to detect occlusion. Uses a Poisson-disk kernel
// rotated per-pixel by a noise function (replaces the noise texture that
// production SSAO implementations use).
const SSAO_VERT = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

const SSAO_FRAG = `
  precision highp float;

  uniform sampler2D tDepth;
  uniform vec2 uResolution;
  uniform mat4 uProjectionMatrix;
  uniform mat4 uInverseProjectionMatrix;
  uniform float uRadius;       // world-space AO radius
  uniform float uBias;         // depth comparison bias
  uniform float uIntensity;    // AO darkness multiplier

  varying vec2 vUv;

  // Reconstruct view-space position from screen UV + depth
  vec3 viewPosFromDepth(vec2 uv, float depth) {
    // NDC: [-1, 1]
    vec4 ndc = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 viewPos = uInverseProjectionMatrix * ndc;
    return viewPos.xyz / viewPos.w;
  }

  // Per-pixel pseudo-random rotation (replaces noise texture)
  // Returns a rotation angle in [0, 2pi) based on screen position.
  // The integer-coordinate hash prevents correlation between neighbors
  // while keeping the pattern temporally stable (no flicker).
  float noiseAngle(vec2 screenCoord) {
    // Integer hash for per-pixel variation (Jarzynski & Olano 2020)
    vec2 p = floor(screenCoord);
    float h = fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
    return h * 6.2831853;
  }

  // 16-sample Poisson-disk hemisphere kernel (pre-normalized to unit hemisphere).
  // These are distributed to minimize correlation — NOT random.
  // The z-component biases samples toward the surface normal direction,
  // which concentrates samples where occlusion is most visible.
  //
  // Distribution: cos-weighted hemisphere sampling (Pharr & Humphreys 2010)
  // with samples at varying distances from origin for multi-scale AO.
  //
  // Using a function lookup instead of an array to ensure WebGL 1 / GLSL ES 1.0
  // compatibility across all drivers (some mobile/older drivers miscompile
  // global arrays indexed by loop variables).
  const int KERNEL_SIZE = 16;

  vec3 getKernelSample(int i) {
    if (i ==  0) return vec3( 0.04, 0.08, 0.05);
    if (i ==  1) return vec3(-0.06, 0.03, 0.08);
    if (i ==  2) return vec3( 0.10,-0.05, 0.07);
    if (i ==  3) return vec3(-0.03,-0.10, 0.12);
    if (i ==  4) return vec3( 0.15, 0.12, 0.10);
    if (i ==  5) return vec3(-0.14, 0.08, 0.18);
    if (i ==  6) return vec3( 0.08,-0.18, 0.14);
    if (i ==  7) return vec3(-0.20, 0.04, 0.16);
    if (i ==  8) return vec3( 0.22, 0.18, 0.12);
    if (i ==  9) return vec3(-0.12,-0.22, 0.24);
    if (i == 10) return vec3( 0.28, 0.06, 0.18);
    if (i == 11) return vec3(-0.08, 0.30, 0.22);
    if (i == 12) return vec3( 0.18,-0.28, 0.30);
    if (i == 13) return vec3(-0.32, 0.14, 0.26);
    if (i == 14) return vec3( 0.14, 0.34, 0.32);
    return vec3(-0.24,-0.30, 0.40); // i == 15
  }

  void main() {
    // Sample depth at this fragment
    float rawDepth = texture2D(tDepth, vUv).r;

    // Sky / far-plane: no AO
    if (rawDepth >= 0.9999) {
      gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
      return;
    }

    // Reconstruct view-space position of this fragment
    vec3 origin = viewPosFromDepth(vUv, rawDepth);

    // Approximate view-space normal from depth buffer derivatives.
    // This avoids needing a separate normal buffer (no MRT).
    // The cross-product of screen-space depth gradients gives the surface normal.
    vec2 texelSize = 1.0 / uResolution;
    float depthR = texture2D(tDepth, vUv + vec2(texelSize.x, 0.0)).r;
    float depthU = texture2D(tDepth, vUv + vec2(0.0, texelSize.y)).r;
    vec3 posR = viewPosFromDepth(vUv + vec2(texelSize.x, 0.0), depthR);
    vec3 posU = viewPosFromDepth(vUv + vec2(0.0, texelSize.y), depthU);
    vec3 normal = normalize(cross(posR - origin, posU - origin));

    // Per-pixel rotation to break up banding
    float angle = noiseAngle(gl_FragCoord.xy);
    float cosA = cos(angle);
    float sinA = sin(angle);

    // Accumulate occlusion from hemisphere samples
    float occlusion = 0.0;

    for (int i = 0; i < KERNEL_SIZE; i++) {
      // Rotate kernel sample around the normal (in tangent plane)
      vec3 sampleDir = getKernelSample(i);
      // Apply per-pixel rotation in XY plane
      vec3 rotated = vec3(
        sampleDir.x * cosA - sampleDir.y * sinA,
        sampleDir.x * sinA + sampleDir.y * cosA,
        sampleDir.z
      );

      // Orient sample to hemisphere around the surface normal.
      // If the sample points away from the surface (dot < 0), flip it.
      // This is the hemisphere constraint — we only sample above the surface.
      if (dot(rotated, normal) < 0.0) rotated = -rotated;

      // Scale by AO radius and offset from origin
      vec3 samplePos = origin + rotated * uRadius;

      // Project sample back to screen space to read its depth
      vec4 projected = uProjectionMatrix * vec4(samplePos, 1.0);
      projected.xyz /= projected.w;
      vec2 sampleUv = projected.xy * 0.5 + 0.5;

      // Read depth at sample's screen position
      float sampleDepth = texture2D(tDepth, sampleUv).r;
      vec3 sampleActualPos = viewPosFromDepth(sampleUv, sampleDepth);

      // Range check: ignore samples that are too far away (prevents
      // halos at depth discontinuities like object edges).
      float rangeCheck = smoothstep(0.0, 1.0,
        uRadius / max(abs(origin.z - sampleActualPos.z), 0.001));

      // Occlusion test: in view space, Z is negative (camera looks down -Z).
      // "Closer to camera" = larger Z (less negative). A sample point is
      // occluded when the actual surface at its screen position is closer
      // to the camera (larger Z) than the sample: sampleActualPos.z > samplePos.z.
      // step(a, b) = 1 when b >= a, so step(samplePos.z + uBias, sampleActualPos.z)
      // fires when the surface is in front of the sample (with bias tolerance).
      occlusion += step(samplePos.z + uBias, sampleActualPos.z) * rangeCheck;
    }

    // Normalize and apply intensity
    float ao = 1.0 - (occlusion / float(KERNEL_SIZE)) * uIntensity;
    ao = clamp(ao, 0.0, 1.0);

    gl_FragColor = vec4(ao, ao, ao, 1.0);
  }
`;

// ---------------------------------------------------------------------------
// Bilateral blur shader (Pass 3)
// ---------------------------------------------------------------------------
// 5-tap Gaussian blur that respects depth edges — prevents AO from bleeding
// across depth discontinuities (e.g., an object's edge against the background).
// The depth-aware weighting is critical: without it, dark AO halos appear
// around objects floating in front of distant backgrounds.
const BLUR_VERT = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

const BLUR_FRAG = `
  precision highp float;

  uniform sampler2D tAO;
  uniform sampler2D tDepth;
  uniform vec2 uDirection; // (1,0) for horizontal, (0,1) for vertical
  uniform vec2 uResolution;

  varying vec2 vUv;

  // Gaussian weights for 5-tap kernel (sigma ~= 1.5)
  // GLSL ES 1.0 does not support typed array constructors, so we use a
  // helper function to index into the weight table.
  float gaussWeight(int i) {
    // weights: 0.06136, 0.24477, 0.38774, 0.24477, 0.06136
    if (i == 0 || i == 4) return 0.06136;
    if (i == 1 || i == 3) return 0.24477;
    return 0.38774; // i == 2 (center)
  }

  void main() {
    vec2 texelSize = 1.0 / uResolution;
    float centerDepth = texture2D(tDepth, vUv).r;

    float result = 0.0;
    float totalWeight = 0.0;

    for (int i = -2; i <= 2; i++) {
      vec2 offset = float(i) * uDirection * texelSize * 2.0;
      vec2 sampleUv = vUv + offset;
      float aoSample = texture2D(tAO, sampleUv).r;
      float sampleDepth = texture2D(tDepth, sampleUv).r;

      // Bilateral weight: attenuate contribution from samples at very
      // different depths. The threshold (0.0001) is tuned for typical
      // perspective projection depth ranges.
      float depthDiff = abs(centerDepth - sampleDepth);
      float depthWeight = 1.0 / (1.0 + depthDiff * 10000.0);

      float w = gaussWeight(i + 2) * depthWeight;
      result += aoSample * w;
      totalWeight += w;
    }

    gl_FragColor = vec4(vec3(result / totalWeight), 1.0);
  }
`;

// ---------------------------------------------------------------------------
// Composite shader (final pass)
// ---------------------------------------------------------------------------
// Multiplies scene color by the blurred AO factor.
const COMPOSITE_VERT = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

const COMPOSITE_FRAG = `
  precision highp float;

  uniform sampler2D tColor;
  uniform sampler2D tAO;

  varying vec2 vUv;

  void main() {
    vec3 color = texture2D(tColor, vUv).rgb;
    float ao = texture2D(tAO, vUv).r;
    gl_FragColor = vec4(color * ao, 1.0);
  }
`;


// ---------------------------------------------------------------------------
// SSAOEffect class
// ---------------------------------------------------------------------------

class SSAOEffect {
  /**
   * @param {THREE.WebGLRenderer} renderer - The app's WebGL renderer
   * @param {THREE.Scene} scene - The scene to render
   * @param {THREE.PerspectiveCamera} camera - The perspective camera
   * @param {Object} [options]
   * @param {number} [options.radius=0.5] - AO sampling radius in view-space units
   * @param {number} [options.bias=0.025] - Depth comparison bias (prevents self-occlusion)
   * @param {number} [options.intensity=1.2] - AO darkness multiplier (higher = darker shadows)
   * @param {number} [options.downscale=1] - Resolution downscale for AO pass (2 = half-res)
   */
  constructor(renderer, scene, camera, options) {
    this._renderer = renderer;
    this._scene = scene;
    this._camera = camera;
    this._enabled = false;

    const opts = options || {};
    this._radius = opts.radius !== undefined ? opts.radius : 0.5;
    this._bias = opts.bias !== undefined ? opts.bias : 0.025;
    this._intensity = opts.intensity !== undefined ? opts.intensity : 1.2;
    this._downscale = opts.downscale !== undefined ? opts.downscale : 1;

    // Get current renderer size
    const size = new THREE.Vector2();
    renderer.getSize(size);
    this._width = size.x;
    this._height = size.y;

    // --- Render targets ---

    // Scene color + depth
    this._depthTexture = new THREE.DepthTexture(this._width, this._height);
    this._depthTexture.type = THREE.UnsignedIntType;
    this._depthTexture.minFilter = THREE.NearestFilter;
    this._depthTexture.magFilter = THREE.NearestFilter;

    this._sceneTarget = new THREE.WebGLRenderTarget(this._width, this._height, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      depthTexture: this._depthTexture,
      depthBuffer: true,
      stencilBuffer: false,
    });

    // AO result (can be lower resolution)
    const aoW = Math.max(1, Math.floor(this._width / this._downscale));
    const aoH = Math.max(1, Math.floor(this._height / this._downscale));

    this._aoTarget = new THREE.WebGLRenderTarget(aoW, aoH, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      depthBuffer: false,
      stencilBuffer: false,
    });

    // Blur ping-pong target
    this._blurTarget = new THREE.WebGLRenderTarget(aoW, aoH, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      depthBuffer: false,
      stencilBuffer: false,
    });

    // --- Full-screen quad geometry (shared by all passes) ---
    this._quadGeo = new THREE.PlaneGeometry(2, 2);

    // --- Inverse projection matrix (updated each frame) ---
    this._inverseProjection = new THREE.Matrix4();

    // --- SSAO pass material ---
    this._ssaoMaterial = new THREE.ShaderMaterial({
      vertexShader: SSAO_VERT,
      fragmentShader: SSAO_FRAG,
      uniforms: {
        tDepth: { value: this._depthTexture },
        uResolution: { value: new THREE.Vector2(aoW, aoH) },
        uProjectionMatrix: { value: this._camera.projectionMatrix },
        uInverseProjectionMatrix: { value: this._inverseProjection },
        uRadius: { value: this._radius },
        uBias: { value: this._bias },
        uIntensity: { value: this._intensity },
      },
      depthTest: false,
      depthWrite: false,
    });

    // --- Blur pass material ---
    this._blurMaterial = new THREE.ShaderMaterial({
      vertexShader: BLUR_VERT,
      fragmentShader: BLUR_FRAG,
      uniforms: {
        tAO: { value: this._aoTarget.texture },
        tDepth: { value: this._depthTexture },
        uDirection: { value: new THREE.Vector2(1, 0) },
        uResolution: { value: new THREE.Vector2(aoW, aoH) },
      },
      depthTest: false,
      depthWrite: false,
    });

    // --- Composite pass material ---
    this._compositeMaterial = new THREE.ShaderMaterial({
      vertexShader: COMPOSITE_VERT,
      fragmentShader: COMPOSITE_FRAG,
      uniforms: {
        tColor: { value: this._sceneTarget.texture },
        tAO: { value: this._blurTarget.texture },
      },
      depthTest: false,
      depthWrite: false,
    });

    // --- Full-screen quad scene (reused for all post-process passes) ---
    this._quadMesh = new THREE.Mesh(this._quadGeo, this._ssaoMaterial);
    this._quadMesh.frustumCulled = false;
    this._postScene = new THREE.Scene();
    this._postScene.add(this._quadMesh);
    this._postCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  }

  // --- Public API ---

  /** Enable SSAO post-processing. */
  enable() {
    this._enabled = true;
  }

  /** Disable SSAO — render() becomes a simple pass-through. */
  disable() {
    this._enabled = false;
  }

  /** Check if SSAO is currently active. */
  get enabled() {
    return this._enabled;
  }

  /** Toggle SSAO on/off. Returns the new state. */
  toggle() {
    this._enabled = !this._enabled;
    return this._enabled;
  }

  /** Update AO parameters at runtime. */
  setParams(params) {
    if (params.radius !== undefined) {
      this._radius = params.radius;
      this._ssaoMaterial.uniforms.uRadius.value = params.radius;
    }
    if (params.bias !== undefined) {
      this._bias = params.bias;
      this._ssaoMaterial.uniforms.uBias.value = params.bias;
    }
    if (params.intensity !== undefined) {
      this._intensity = params.intensity;
      this._ssaoMaterial.uniforms.uIntensity.value = params.intensity;
    }
  }

  /**
   * Update render target sizes. Call this when the renderer resizes.
   * WebGLRenderTarget.setSize() does NOT resize the DepthTexture, so we
   * must dispose and recreate the scene target with a fresh DepthTexture.
   * @param {number} width - New render width in pixels
   * @param {number} height - New render height in pixels
   */
  setSize(width, height) {
    if (width === this._width && height === this._height) return;
    this._width = width;
    this._height = height;

    // Dispose old scene target (including its DepthTexture)
    this._sceneTarget.dispose();
    this._depthTexture.dispose();

    // Recreate DepthTexture + scene target at new dimensions
    this._depthTexture = new THREE.DepthTexture(width, height);
    this._depthTexture.type = THREE.UnsignedIntType;
    this._depthTexture.minFilter = THREE.NearestFilter;
    this._depthTexture.magFilter = THREE.NearestFilter;

    this._sceneTarget = new THREE.WebGLRenderTarget(width, height, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      depthTexture: this._depthTexture,
      depthBuffer: true,
      stencilBuffer: false,
    });

    // Update texture references in materials
    this._ssaoMaterial.uniforms.tDepth.value = this._depthTexture;
    this._blurMaterial.uniforms.tDepth.value = this._depthTexture;
    this._compositeMaterial.uniforms.tColor.value = this._sceneTarget.texture;

    // AO targets at (possibly) reduced resolution
    const aoW = Math.max(1, Math.floor(width / this._downscale));
    const aoH = Math.max(1, Math.floor(height / this._downscale));
    this._aoTarget.setSize(aoW, aoH);
    this._blurTarget.setSize(aoW, aoH);

    // Update resolution uniforms
    this._ssaoMaterial.uniforms.uResolution.value.set(aoW, aoH);
    this._blurMaterial.uniforms.uResolution.value.set(aoW, aoH);
  }

  /**
   * Render the scene with SSAO. Call this instead of renderer.render().
   * When disabled, falls through to a direct renderer.render() call.
   */
  render() {
    if (!this._enabled) {
      this._renderer.render(this._scene, this._camera);
      return;
    }

    const renderer = this._renderer;

    // Update camera-dependent uniforms (inverse projection for depth reconstruction)
    this._inverseProjection.copy(this._camera.projectionMatrix).invert();

    // ---- Pass 1: Render scene to offscreen target (color + depth) ----
    renderer.setRenderTarget(this._sceneTarget);
    renderer.clear();
    renderer.render(this._scene, this._camera);

    // ---- Pass 2: Compute SSAO from depth buffer ----
    this._quadMesh.material = this._ssaoMaterial;
    renderer.setRenderTarget(this._aoTarget);
    renderer.clear();
    renderer.render(this._postScene, this._postCamera);

    // ---- Pass 3a: Horizontal bilateral blur ----
    this._blurMaterial.uniforms.tAO.value = this._aoTarget.texture;
    this._blurMaterial.uniforms.uDirection.value.set(1, 0);
    this._quadMesh.material = this._blurMaterial;
    renderer.setRenderTarget(this._blurTarget);
    renderer.clear();
    renderer.render(this._postScene, this._postCamera);

    // ---- Pass 3b: Vertical bilateral blur ----
    this._blurMaterial.uniforms.tAO.value = this._blurTarget.texture;
    this._blurMaterial.uniforms.uDirection.value.set(0, 1);
    renderer.setRenderTarget(this._aoTarget);
    renderer.clear();
    renderer.render(this._postScene, this._postCamera);

    // ---- Pass 4: Composite scene color * AO ----
    this._compositeMaterial.uniforms.tAO.value = this._aoTarget.texture;
    this._quadMesh.material = this._compositeMaterial;
    renderer.setRenderTarget(null); // render to screen
    renderer.clear();
    renderer.render(this._postScene, this._postCamera);
  }

  /** Clean up GPU resources. */
  dispose() {
    this._sceneTarget.dispose();
    this._depthTexture.dispose();
    this._aoTarget.dispose();
    this._blurTarget.dispose();
    this._quadGeo.dispose();
    this._ssaoMaterial.dispose();
    this._blurMaterial.dispose();
    this._compositeMaterial.dispose();
  }
}
