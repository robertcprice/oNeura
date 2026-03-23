// Plant canopy volume vertex shader.
// Organic shape from 3-octave noise displacement on icosahedron.
// Per-vertex wind animation driven by atmospheric simulation.
// NOTE: This file must stay in sync with PLANT_CANOPY_VERT in plant_renderer.js.

uniform float uTime;
uniform float uWindX;
uniform float uWindZ;
uniform float uWindStrength;
uniform float uGrowthScale;
uniform float uCanopyRadius;

attribute float branchDepth;

varying vec3 vWorldPos;
varying vec3 vNormal;
varying float vBranchDepth;
varying float vNoiseDisplacement;

float hash3(vec3 p) {
    p = fract(p * vec3(443.897, 441.423, 437.195));
    p += dot(p, p.yzx + 19.19);
    return fract((p.x + p.y) * p.z);
}

float noise3(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash3(i);
    float b = hash3(i + vec3(1, 0, 0));
    float c = hash3(i + vec3(0, 1, 0));
    float d = hash3(i + vec3(1, 1, 0));
    float e = hash3(i + vec3(0, 0, 1));
    float f1 = hash3(i + vec3(1, 0, 1));
    float g = hash3(i + vec3(0, 1, 1));
    float h = hash3(i + vec3(1, 1, 1));
    float x1 = mix(a, b, f.x);
    float x2 = mix(c, d, f.x);
    float x3 = mix(e, f1, f.x);
    float x4 = mix(g, h, f.x);
    float y1 = mix(x1, x2, f.y);
    float y2 = mix(x3, x4, f.y);
    return mix(y1, y2, f.z);
}

void main() {
    vec3 dir = normalize(position);
    // Multi-octave noise for organic canopy shape.
    // Low freq: large lobes (like major branch clusters).
    // High freq: small leaf-cluster bumps.
    float noiseVal = noise3(dir * 2.5 + 0.5) * 0.40
                   + noise3(dir * 5.0 + 1.3) * 0.20
                   + noise3(dir * 10.0 + 2.7) * 0.08;
    vNoiseDisplacement = noiseVal;

    // Displacement: push vertices outward along normal for bumpy shape
    vec3 pos = position + normal * noiseVal * uCanopyRadius * 0.40;
    pos *= uGrowthScale;
    vBranchDepth = branchDepth;

    float swayFactor = 0.3 + branchDepth * 0.7;
    float gust = sin(uTime * 1.8 + pos.x * 1.5 + pos.z * 1.1);
    float sway = gust * swayFactor * uWindStrength * 0.10;
    float flutter = sin(uTime * 4.5 + pos.x * 8.0 + pos.y * 6.0 + pos.z * 7.0)
                  * branchDepth * 0.015 * (1.0 + uWindStrength * 0.5);

    pos.x += sway * sign(uWindX + 0.001) + flutter;
    pos.z += sway * sign(uWindZ + 0.001) + flutter * 0.7;
    pos.y += flutter * 0.3 - swayFactor * uWindStrength * 0.02;

    vec4 worldPos = modelMatrix * vec4(pos, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * viewMatrix * worldPos;
}
