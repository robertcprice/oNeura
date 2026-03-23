// Terrain surface vertex shader — heightmap displacement from emergent simulation data.
// All values from molecular optics pipeline, zero hardcoded.

uniform sampler2D tHeightmap;   // R=relief, G=soilTexture, B=moisture, A=waterMask
uniform float uVoxelHeight;     // 0.6
uniform float uVoxelBaseY;      // -0.2
uniform float uSurfaceLevels;   // 7.0

varying vec2 vTerrainUv;
varying float vRelief;
varying float vMoisture;
varying float vHeight;
varying vec3 vWorldPos;

void main() {
    vTerrainUv = uv;

    vec4 hm = texture2D(tHeightmap, uv);
    float relief = hm.r;
    vRelief = relief;
    vMoisture = hm.b;

    // Height: blend continuous + quantized (matches simulation formula)
    float normalized = clamp(relief / 1.1, 0.0, 1.0);
    float cH = uVoxelBaseY + (normalized * (uSurfaceLevels - 1.0) + 1.0) * uVoxelHeight;
    float qH = uVoxelBaseY + (floor(normalized * (uSurfaceLevels - 1.0) + 0.5) + 1.0) * uVoxelHeight;
    float shellHeight = cH * 0.74 + qH * 0.26 + uVoxelHeight * 0.06;
    vHeight = shellHeight;

    vec3 displaced = position;
    displaced.y = shellHeight;
    vWorldPos = (modelMatrix * vec4(displaced, 1.0)).xyz;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(displaced, 1.0);
}
