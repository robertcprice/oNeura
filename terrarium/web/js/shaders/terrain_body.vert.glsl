// Terrain body (side walls) vertex shader.
// Top vertices displaced by heightmap, bottom pinned at floor.
// Shows geological strata: bedrock → subsoil → topsoil.

uniform sampler2D tHeightmap;
uniform float uVoxelHeight;
uniform float uVoxelBaseY;
uniform float uSurfaceLevels;
uniform float uFloorY;

varying vec2 vTerrainUv;
varying float vNormalizedDepth; // 0=floor, 1=surface
varying float vWorldY;
varying float vSurfaceY;
varying vec3 vWorldPos;

void main() {
    vTerrainUv = uv;

    vec4 hm = texture2D(tHeightmap, vec2(uv.x, 0.5));
    float relief = hm.r;
    float normalized = clamp(relief / 1.1, 0.0, 1.0);
    float cH = uVoxelBaseY + (normalized * (uSurfaceLevels - 1.0) + 1.0) * uVoxelHeight;
    float qH = uVoxelBaseY + (floor(normalized * (uSurfaceLevels - 1.0) + 0.5) + 1.0) * uVoxelHeight;
    float surfaceY = cH * 0.74 + qH * 0.26 + uVoxelHeight * 0.06;
    vSurfaceY = surfaceY;

    // position.y: 0 = bottom of wall, 1 = top (surface)
    float verticalT = position.y;
    float worldY = mix(uFloorY, surfaceY, verticalT);
    vWorldY = worldY;
    vNormalizedDepth = verticalT;

    vec3 displaced = position;
    displaced.y = worldY;
    vWorldPos = (modelMatrix * vec4(displaced, 1.0)).xyz;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(displaced, 1.0);
}
