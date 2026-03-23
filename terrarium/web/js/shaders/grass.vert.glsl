// Instanced grass blade vertex shader.
// Density driven by emergent organic signal (chlorophyll/organic from molecular optics).
// Wind animation from atmospheric simulation (not procedural).
// GPU-culled: blades in non-green or underwater cells collapse to zero height.
// NOTE: This file must stay in sync with the inline GRASS_VERT in terrain_renderer.js.

uniform sampler2D tHeightmap;
uniform sampler2D tSoilColor;
uniform sampler2D tAtmosphere;
uniform float uTime;
uniform float uGridW;
uniform float uGridH;
uniform float uVoxelHeight;
uniform float uVoxelBaseY;
uniform float uSurfaceLevels;

attribute vec2 instanceCell;
attribute float instanceSeed;
attribute float instanceBlade;

varying vec3 vGrassColor;
varying float vAlpha;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

// Height formula — IDENTICAL to terrain surface shader
float terrainHeight(float relief) {
  float normalized = clamp(relief / 1.1, 0.0, 1.0);
  float cH = uVoxelBaseY + (normalized * (uSurfaceLevels - 1.0) + 1.0) * uVoxelHeight;
  float qH = uVoxelBaseY + (floor(normalized * (uSurfaceLevels - 1.0) + 0.5) + 1.0) * uVoxelHeight;
  return cH * 0.74 + qH * 0.26 + uVoxelHeight * 0.06;
}

void main() {
  // Per-blade random offset within cell
  float phase = instanceSeed + instanceBlade * 0.25;
  vec2 bladeOffset = vec2(
    (hash(vec2(phase, 0.13)) - 0.5) * 0.7,
    (hash(vec2(phase, 0.47)) - 0.5) * 0.7
  );

  // Sample heightmap at ACTUAL blade position (not just cell center)
  vec2 bladeUv = (instanceCell + 0.5 + bladeOffset) / vec2(uGridW, uGridH);
  bladeUv = clamp(bladeUv, vec2(0.0), vec2(1.0));
  vec4 hm = texture2D(tHeightmap, bladeUv);
  float relief = hm.r;
  float moisture = hm.b;

  // Compute terrain height at blade position — same formula as surface mesh
  float surfaceY = terrainHeight(relief);

  // Check if underwater — grass doesn't grow where water sources deposit water
  float waterMask = hm.a;
  float isUnderwater = step(0.82, waterMask);

  // Green signal from emergent soil color (organic matter / chlorophyll)
  vec2 cellUv = (instanceCell + 0.5) / vec2(uGridW, uGridH);
  vec4 soil = texture2D(tSoilColor, cellUv);
  float organicSignal = clamp(soil.g * 2.0 + soil.r * 0.5, 0.0, 1.0);
  float greenSignal = clamp(
    (soil.g - soil.r) * 2.4 + (soil.g - soil.b) * 1.3 + organicSignal * 0.6, 0.0, 1.0);

  // Most terrain should have grass — only bare rock/sand or underwater is bare
  float densityRoll = hash(vec2(instanceSeed * 7.3, instanceBlade * 3.1));
  float visible = step(0.05, organicSignal) * step(densityRoll, 0.6 + greenSignal * 0.4) * (1.0 - isUnderwater);

  // World position of blade base — ON the terrain surface
  float ox = -uGridW * 0.5;
  float oz = -uGridH * 0.5;
  vec3 basePos = vec3(
    instanceCell.x + 0.5 + ox + bladeOffset.x,
    surfaceY,
    instanceCell.y + 0.5 + oz + bladeOffset.y
  );

  // Blade height: short grass, not tall sticks
  float heightVariation = 0.4 + hash(vec2(instanceSeed, 0.77)) * 0.6;
  float bladeHeight = visible * (0.04 + organicSignal * 0.12) * heightVariation;

  // Wind from atmospheric simulation
  vec4 atmo = texture2D(tAtmosphere, cellUv);
  float windX = atmo.b;
  float windY = atmo.a;
  float windStrength = length(vec2(windX, windY));

  // Blade deformation: position.y is [0,1] from base to tip
  float tipFactor = clamp(position.y, 0.0, 1.0);
  float windBend = tipFactor * tipFactor * windStrength * 0.5;
  float sway = sin(uTime * 2.2 + basePos.x * 1.5 + basePos.z * 1.1 + instanceSeed * 6.28)
             * tipFactor * 0.03 * (1.0 + windStrength * 0.4);

  vec3 worldPos = basePos;
  worldPos.y += tipFactor * bladeHeight;
  worldPos.x += windBend * sign(windX + 0.001) + sway;
  worldPos.z += windBend * sign(windY + 0.001) + sway * 0.6;

  // Blade width — narrow grass blades
  worldPos.x += position.x * 0.012 * (1.0 - tipFactor * 0.6);
  worldPos.z += position.z * 0.004;

  // Color emerges from soil color (chlorophyll/organic content)
  vec3 baseGreen = vec3(soil.r * 0.65, soil.g * 1.15, soil.b * 0.55);
  baseGreen = clamp(baseGreen, 0.0, 1.0);
  vGrassColor = mix(baseGreen * 0.85, baseGreen * 1.1, tipFactor);
  vGrassColor *= 1.0 - moisture * 0.15;
  vAlpha = visible;

  gl_Position = projectionMatrix * modelViewMatrix * vec4(worldPos, 1.0);
}
