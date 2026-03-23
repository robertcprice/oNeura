// Instanced grass fragment shader.
// Color from vertex shader (emergent, not hardcoded).

varying vec3 vGrassColor;
varying float vAlpha;

void main() {
    if (vAlpha < 0.5) discard;
    gl_FragColor = vec4(vGrassColor, 1.0);
}
