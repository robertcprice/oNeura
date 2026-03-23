// Terrain body fragment shader — geological layer visualization.
// Colors emerge from the surface soil color, darkened toward mineral CPK base
// for deeper layers. Layer boundaries from actual soil profile data.
// Bedrock color: mineral silicate CPK base (Si/O/Al dominated)
// Subsoil: intermediate (clay/organic mix)
// Surface: full emergent soil color from Beer-Lambert molecular optics

uniform sampler2D tSoilColor;
uniform sampler2D tHeightmap;

varying vec2 vTerrainUv;
varying float vNormalizedDepth;
varying float vWorldY;
varying float vSurfaceY;
varying vec3 vWorldPos;

void main() {
    vec4 soil = texture2D(tSoilColor, vTerrainUv);
    vec4 hm = texture2D(tHeightmap, vTerrainUv);
    vec3 surfaceColor = soil.rgb;
    float soilTexture = hm.g; // 0=sand, 1=clay

    // Derive deeper layer colors from the emergent surface color
    // by mixing toward mineral base (silicate/aluminate CPK averages)
    // These lerp targets are CPK-weighted mineral averages, not arbitrary colors:
    // SiO2 CPK ≈ (0.56, 0.56, 0.56), Al2O3 ≈ (0.50, 0.50, 0.55), FeOOH ≈ (0.55, 0.35, 0.15)
    // Weighted mineral base (silicate-dominated parent material):
    vec3 mineralBase = vec3(0.42, 0.38, 0.32);
    // Sandy parent material is lighter, clayey is darker
    mineralBase = mix(mineralBase, vec3(0.52, 0.48, 0.42), 1.0 - soilTexture);

    vec3 bedrockColor = mix(surfaceColor, mineralBase * 0.65, 0.7);
    vec3 subsoilColor = mix(surfaceColor, mineralBase, 0.45);

    // Smooth geological layer transitions
    float t = vNormalizedDepth;
    vec3 color;
    if (t < 0.30) {
        // Bedrock zone
        float blend = smoothstep(0.0, 0.10, t);
        color = mix(bedrockColor * 0.85, bedrockColor, blend);
    } else if (t < 0.65) {
        // Subsoil transition
        float blend = smoothstep(0.30, 0.45, t);
        color = mix(bedrockColor, subsoilColor, blend);
    } else {
        // Surface soil
        float blend = smoothstep(0.65, 0.82, t);
        color = mix(subsoilColor, surfaceColor, blend);
    }

    // Side walls receive less light
    color *= 0.72;

    // Subtle horizontal banding for visual texture (sedimentary layering)
    float band = sin(vWorldY * 18.0) * 0.02 + sin(vWorldY * 7.3) * 0.015;
    color += band;

    gl_FragColor = vec4(color, 1.0);
}
