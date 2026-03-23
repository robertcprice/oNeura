// Plant canopy volume fragment shader.
// Color driven by backend leaf_rgb (emergent from molecular optics / photosynthesis).
// Subsurface scattering for thin-leaf translucency.
// Rim light for sky-edge definition.
// NOTE: This file must stay in sync with PLANT_CANOPY_FRAG in plant_renderer.js.

uniform vec3 uLeafColor;
uniform float uDaylight;
uniform vec3 uSunDir;

varying vec3 vWorldPos;
varying vec3 vNormal;
varying float vBranchDepth;
varying float vNoiseDisplacement;

void main() {
    vec3 sunDir = normalize(uSunDir);
    float NdotL = dot(vNormal, sunDir);

    float diffuse = max(NdotL, 0.0);
    float subsurface = max(-NdotL, 0.0) * 0.35;
    float ambient = 0.30 + uDaylight * 0.20;

    vec3 baseColor = uLeafColor;
    float variation = vNoiseDisplacement * 1.5;
    baseColor *= 0.88 + variation * 0.24;

    vec3 subsurfaceColor = baseColor * vec3(1.1, 1.3, 0.7);

    vec3 color = baseColor * (ambient + diffuse * uDaylight * 0.55)
               + subsurfaceColor * subsurface * uDaylight;

    // Depth variation: outer surface brighter, inner darker (self-shadowing)
    color *= 0.82 + vBranchDepth * 0.18;

    // Rim light: edges of canopy catch sky light (Fresnel-like)
    float rimFactor = 1.0 - max(dot(vNormal, normalize(vec3(0.0, 0.3, 1.0))), 0.0);
    color += vec3(0.04, 0.06, 0.08) * rimFactor * rimFactor * uDaylight;

    gl_FragColor = vec4(color, 1.0);
}
