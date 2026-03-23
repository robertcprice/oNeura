// Plant trunk/branch vertex shader.
// Wind sway driven by atmospheric simulation (not procedural).
// branchDepth attribute: 0.0 at trunk base, 1.0 at branch tips.
// Quadratic sway: tips move exponentially more than trunk.
// NOTE: This file must stay in sync with PLANT_TRUNK_VERT in plant_renderer.js.

uniform float uTime;
uniform float uWindX;
uniform float uWindZ;
uniform float uWindStrength;
uniform float uGrowthScale;

attribute float branchDepth;

varying vec3 vWorldPos;
varying vec3 vNormal;
varying float vBranchDepth;

void main() {
    vec3 pos = position;
    vBranchDepth = branchDepth;
    pos *= uGrowthScale;

    float swayFactor = branchDepth * branchDepth;
    float gust = sin(uTime * 1.8 + pos.x * 2.0 + pos.z * 1.4 + branchDepth * 3.0);
    float sway = gust * swayFactor * uWindStrength * 0.12;
    float tremor = sin(uTime * 5.2 + pos.x * 7.0 + pos.z * 5.3) * swayFactor * 0.008;

    pos.x += sway * sign(uWindX + 0.001) + tremor;
    pos.z += sway * sign(uWindZ + 0.001) + tremor * 0.6;
    pos.y -= swayFactor * uWindStrength * 0.03;

    vec4 worldPos = modelMatrix * vec4(pos, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * viewMatrix * worldPos;
}
