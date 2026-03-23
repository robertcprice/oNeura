// Water surface vertex shader.
// Height from terrain heightmap + water mask depth.
// Ripple animation from physical wind data.

uniform sampler2D tHeightmap;
uniform sampler2D tAtmosphere;
uniform float uTime;
uniform float uVoxelHeight;
uniform float uVoxelBaseY;
uniform float uSurfaceLevels;

varying vec2 vUv;
varying float vWaterDepth;
varying vec3 vWorldPos;

void main() {
    vUv = uv;

    vec4 hm = texture2D(tHeightmap, uv);
    float relief = hm.r;
    float waterMask = hm.a;
    vWaterDepth = waterMask;

    // Surface height
    float normalized = clamp(relief / 1.1, 0.0, 1.0);
    float cH = uVoxelBaseY + (normalized * (uSurfaceLevels - 1.0) + 1.0) * uVoxelHeight;
    float qH = uVoxelBaseY + (floor(normalized * (uSurfaceLevels - 1.0) + 0.5) + 1.0) * uVoxelHeight;
    float surfaceY = cH * 0.74 + qH * 0.26 + uVoxelHeight * 0.06;

    // Water sits above terrain, depth from simulation water_mask
    float waterHeight = surfaceY + waterMask * 0.22 + 0.03;

    // Wind-driven ripples (from atmospheric simulation, not procedural)
    vec4 atmo = texture2D(tAtmosphere, uv);
    float windStrength = length(vec2(atmo.b, atmo.a));
    float ripple = sin(position.x * 8.0 + uTime * 2.0 + windStrength * 3.0)
                 * cos(position.z * 6.0 + uTime * 1.5)
                 * 0.012 * waterMask * (0.3 + windStrength * 0.7);

    vec3 displaced = position;
    displaced.y = waterHeight + ripple;
    vWorldPos = (modelMatrix * vec4(displaced, 1.0)).xyz;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(displaced, 1.0);
}
