// Plant trunk/branch fragment shader.
// Color driven by backend stem_rgb (emergent from Eyring TST molecular optics).
// Bark texture from procedural noise.
// NOTE: This file must stay in sync with PLANT_TRUNK_FRAG in plant_renderer.js.

uniform vec3 uStemColor;
uniform float uDaylight;
uniform vec3 uSunDir;

varying vec3 vWorldPos;
varying vec3 vNormal;
varying float vBranchDepth;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    float grain = sin(vWorldPos.y * 40.0 + vWorldPos.x * 3.0) * 0.03;
    float noise = hash(floor(vWorldPos.xz * 20.0)) * 0.06;
    float barkVariation = 1.0 + grain + noise - 0.04;

    vec3 color = uStemColor * barkVariation;
    color = mix(color, color * 1.15, vBranchDepth * 0.3);

    vec3 sunDir = normalize(uSunDir);
    float diffuse = max(dot(vNormal, sunDir), 0.0);
    float ambient = 0.35 + uDaylight * 0.15;
    color *= ambient + diffuse * uDaylight * 0.55;

    gl_FragColor = vec4(color, 1.0);
}
