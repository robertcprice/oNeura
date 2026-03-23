// Terrain surface fragment shader — emergent soil color from molecular optics.
// Color comes entirely from the backend Beer-Lambert/CPK/Kubelka-Munk pipeline.
// Moisture darkening is physical: wet soil absorbs more light (Lobell & Asner 2002).

uniform sampler2D tSoilColor;   // RGB = emergent soil color from backend, A = canopy
uniform sampler2D tHeightmap;   // R=relief, G=soilTexture, B=moisture, A=waterMask
uniform float uDaylight;
uniform vec3 uSunDir;

varying vec2 vTerrainUv;
varying float vRelief;
varying float vMoisture;
varying float vHeight;
varying vec3 vWorldPos;

void main() {
    // Sample emergent soil color (computed by backend from Beer-Lambert molecular optics)
    vec4 soil = texture2D(tSoilColor, vTerrainUv);
    vec3 color = soil.rgb;

    // Moisture darkening — physical: wet surfaces have lower albedo
    // Lobell & Asner 2002: soil reflectance decreases ~30-50% when saturated
    color *= 1.0 - vMoisture * 0.35;

    // Derivative-based normal for lighting
    vec3 dFdxPos = dFdx(vWorldPos);
    vec3 dFdyPos = dFdy(vWorldPos);
    vec3 normal = normalize(cross(dFdxPos, dFdyPos));

    // Sun direction from backend compute_solar_state()
    vec3 lightDir = normalize(uSunDir);
    float diffuse = max(dot(normal, lightDir), 0.0);
    float ambient = 0.18 + uDaylight * 0.22;
    color *= ambient + diffuse * (0.40 + uDaylight * 0.25);

    gl_FragColor = vec4(color, 1.0);
}
