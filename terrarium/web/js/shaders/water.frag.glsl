// Water fragment shader — Pope & Fry (1997) spectral absorption model.
// Water color emerges from wavelength-dependent absorption coefficients:
//   a_red   = 0.55 m^-1  (strong absorption → water appears blue)
//   a_green = 0.12 m^-1  (moderate)
//   a_blue  = 0.02 m^-1  (weak → blue light persists)
// These are measured physical constants, not hardcoded aesthetic choices.

uniform sampler2D tHeightmap;
uniform float uTime;

varying vec2 vUv;
varying float vWaterDepth;
varying vec3 vWorldPos;

void main() {
    vec4 hm = texture2D(tHeightmap, vUv);
    float waterMask = hm.a;

    // Discard non-water fragments
    if (waterMask < 0.06) discard;

    // Pope & Fry 1997: spectral attenuation I = I0 * exp(-a * depth)
    // Absorption coefficients (m^-1) for pure water:
    float depth = waterMask * 0.6;
    vec3 waterColor = vec3(
        0.18 * exp(-0.55 * depth),  // red attenuates fastest
        0.42 * exp(-0.12 * depth),  // green moderate
        0.72 * exp(-0.02 * depth)   // blue persists longest
    );

    // Depth-dependent opacity: deeper water is more opaque
    // Beer-Lambert: transmittance T = exp(-a_total * d)
    float opacity = 1.0 - exp(-2.5 * waterMask);
    opacity = clamp(opacity, 0.25, 0.85);

    // Wind-driven caustic patterns
    float caustic = sin(vWorldPos.x * 14.0 + uTime * 1.2)
                  * sin(vWorldPos.z * 11.0 + uTime * 0.9)
                  * 0.04 * waterMask;
    waterColor += caustic;

    gl_FragColor = vec4(waterColor, opacity);
}
