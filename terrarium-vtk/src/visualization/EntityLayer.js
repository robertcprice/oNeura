/**
 * EntityLayer — vtk.js rendering layer for biological entities.
 *
 * Architecture: One vtkActor per entity category (plants-trunks, plants-canopies,
 * flies, fruits, seeds, earthworms, nematodes). Each actor holds merged PolyData
 * built via vtkAppendPolyData so the entire category is a single draw call.
 *
 * All colors originate from the Rust backend. When visual data is missing,
 * a neutral gray [0.5, 0.5, 0.5] is used as the fallback.
 */
import '@kitware/vtk.js/Rendering/Profiles/Geometry';

import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkSphereSource from '@kitware/vtk.js/Filters/Sources/SphereSource';
import vtkConeSource from '@kitware/vtk.js/Filters/Sources/ConeSource';
import vtkCylinderSource from '@kitware/vtk.js/Filters/Sources/CylinderSource';
import vtkAppendPolyData from '@kitware/vtk.js/Filters/General/AppendPolyData';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const FALLBACK_RGB = [0.5, 0.5, 0.5];
const PICK_TOLERANCE = 1.5; // world units — generous for clicking near entities

// Higher resolution for smoother geometry
const SPHERE_THETA = 16;
const SPHERE_PHI = 16;
const CYLINDER_RES = 12;
const CONE_RES = 12;

// Height conversion: backend sends height_mm. At early growth plants are
// only 3-5mm. We amplify massively so they're prominent at ecosystem scale.
// A 4mm plant becomes 1.2 units, a mature 50mm plant becomes 5 units.
const MM_TO_WORLD = 1.0 / 10.0;
const MIN_PLANT_HEIGHT = 1.2; // minimum visible height — plants must be prominent

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Ensure an [r, g, b] array is in 0-1 range for vtk.js.
 * Backend sends 0-1 floats. If values > 1, assume 0-255 and normalize.
 * Returns FALLBACK_RGB when input is missing or invalid.
 */
function normalizeRgb(rgb) {
  if (!rgb || !Array.isArray(rgb) || rgb.length < 3) return FALLBACK_RGB;
  // Auto-detect: if any value > 1, assume 0-255
  if (rgb[0] > 1 || rgb[1] > 1 || rgb[2] > 1) {
    return [rgb[0] / 255, rgb[1] / 255, rgb[2] / 255];
  }
  return [rgb[0], rgb[1], rgb[2]];
}

/**
 * Create a simple actor with a mapper attached. The actor's diffuse color
 * is set to `rgb` (already normalized 0-1).
 */
function makeColoredActor(polyData, rgb) {
  const mapper = vtkMapper.newInstance();
  mapper.setInputData(polyData);

  const actor = vtkActor.newInstance();
  actor.setMapper(mapper);
  const prop = actor.getProperty();
  prop.setColor(...rgb);
  prop.setAmbient(0.5);
  prop.setDiffuse(0.6);
  prop.setSpecular(0.15);
  prop.setSpecularPower(30);
  return actor;
}

/**
 * Build merged PolyData from an array of source output data objects using
 * vtkAppendPolyData. Returns null when the array is empty.
 */
function mergePolyDatas(polyDatas) {
  if (polyDatas.length === 0) return null;
  if (polyDatas.length === 1) return polyDatas[0];

  const appender = vtkAppendPolyData.newInstance();
  appender.setInputData(polyDatas[0]);
  for (let i = 1; i < polyDatas.length; i++) {
    appender.addInputData(polyDatas[i]);
  }
  appender.update();
  return appender.getOutputData();
}

/**
 * Generate sphere PolyData at a given center and radius.
 */
function makeSpherePolyData(cx, cy, cz, radius) {
  const src = vtkSphereSource.newInstance({
    center: [cx, cy, cz],
    radius,
    thetaResolution: SPHERE_THETA,
    phiResolution: SPHERE_PHI,
  });
  src.update();
  return src.getOutputData();
}

/**
 * Generate cylinder PolyData at a given center, height, and radius.
 * vtk.js CylinderSource default direction is [1,0,0], so we set it
 * to [0,0,1] (up) for vertical trunks.
 */
function makeCylinderPolyData(cx, cy, cz, height, radius) {
  const src = vtkCylinderSource.newInstance({
    center: [cx, cy, cz],
    height,
    radius,
    resolution: CYLINDER_RES,
    direction: [0, 0, 1],
    capping: true,
  });
  src.update();
  return src.getOutputData();
}

/**
 * Generate cone PolyData at a given center, height, and radius.
 * Direction points upward ([0, 0, 1]).
 */
function makeConePolyData(cx, cy, cz, height, radius) {
  const src = vtkConeSource.newInstance({
    center: [cx, cy, cz],
    height,
    radius,
    resolution: CONE_RES,
    direction: [0, 0, 1],
    capping: true,
  });
  src.update();
  return src.getOutputData();
}

// ---------------------------------------------------------------------------
// EntityLayer
// ---------------------------------------------------------------------------

export class EntityLayer {
  /**
   * @param {vtkRenderer} renderer - The vtk.js renderer to add actors to.
   */
  constructor(renderer) {
    this._renderer = renderer;

    // Active actors grouped by category for easy teardown
    this._actors = [];

    // Spatial index for picking: array of { kind, index, x, y, z, data }
    this._pickIndex = [];
  }

  // =========================================================================
  // Public API
  // =========================================================================

  /**
   * Receive the full entities message from the backend and rebuild all
   * entity geometry. This replaces the previous frame completely.
   *
   * @param {Object} msg — entity payload from backend WebSocket
   */
  update(msg) {
    this._removeAllActors();
    this._pickIndex = [];

    if (!msg) return;

    this._buildPlants(msg);
    this._buildFlies(msg);
    this._buildFruits(msg);
    this._buildSeeds(msg);
    this._buildEarthworms(msg);
    this._buildNematodes(msg);
  }

  /**
   * Find the nearest entity to a world-space position.
   *
   * @param {number[]} worldPos — [x, y, z]
   * @returns {{ kind: string, index: number, x: number, y: number, z: number, data: Object } | null}
   */
  findEntityAt(worldPos) {
    if (!worldPos || this._pickIndex.length === 0) return null;

    const wx = worldPos[0];
    const wy = worldPos[1];
    const wz = worldPos[2];

    let bestDist = Infinity;
    let bestEntry = null;

    for (let i = 0; i < this._pickIndex.length; i++) {
      const e = this._pickIndex[i];
      const dx = e.x - wx;
      const dy = e.y - wy;
      // Use 2D distance (XY plane) — click lands on terrain, entity center may be above
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < bestDist) {
        bestDist = dist;
        bestEntry = e;
      }
    }

    if (bestDist > PICK_TOLERANCE) return null;
    return bestEntry;
  }

  /**
   * Remove all actors from the renderer and release references.
   */
  destroy() {
    this._removeAllActors();
    this._pickIndex = [];
  }

  // =========================================================================
  // Internal — actor management
  // =========================================================================

  _addActor(actor) {
    this._renderer.addActor(actor);
    this._actors.push(actor);
  }

  _removeAllActors() {
    for (const actor of this._actors) {
      this._renderer.removeActor(actor);
      actor.delete();
    }
    this._actors = [];
  }

  // =========================================================================
  // Internal — Plants
  // =========================================================================

  _buildPlants(msg) {
    const plants = msg.full_plants;
    const visuals = msg.plant_visuals;
    if (!plants || plants.length === 0) return;

    const trunkPolys = [];
    const canopyPolys = [];
    // We will accumulate per-entity average colors and use the first valid
    // visual as the actor color. For a single merged actor per category all
    // entities share one color. In practice the backend supplies per-plant
    // visuals; when they differ significantly a future version can use
    // per-vertex coloring. For now a weighted average keeps it simple.
    let stemR = 0, stemG = 0, stemB = 0, stemCount = 0;
    let leafR = 0, leafG = 0, leafB = 0, leafCount = 0;

    for (let i = 0; i < plants.length; i++) {
      const p = plants[i];
      const vis = visuals && visuals[i] ? visuals[i] : null;

      // Position — offset to cell center (+0.5)
      const x = p.x + 0.5;
      const y = p.y + 0.5;
      const heightWorld = Math.max(MIN_PLANT_HEIGHT, (p.height_mm || 10) * MM_TO_WORLD);
      // Use canopy_scale from visuals if available
      const canopyScale = vis ? (vis.canopy_scale || 1.0) : 1.0;
      const trunkRadius = Math.max(0.05, heightWorld * 0.08);
      const canopyRadius = Math.max(0.25, heightWorld * 0.45 * canopyScale);

      // Trunk: cylinder from ground to top of trunk (trunk is ~60% of total height)
      const trunkHeight = heightWorld * 0.6;
      const trunkCenterZ = trunkHeight * 0.5;
      trunkPolys.push(makeCylinderPolyData(x, y, trunkCenterZ, trunkHeight, trunkRadius));

      // Canopy: sphere sitting on top of trunk
      const canopyCenterZ = trunkHeight + canopyRadius * 0.5;
      canopyPolys.push(makeSpherePolyData(x, y, canopyCenterZ, canopyRadius));

      // Colors — backend sends stem_rgb, leaf_rgb as 0-1 float arrays
      const stemRgb = vis ? normalizeRgb(vis.stem_rgb) : [0.55, 0.35, 0.2];
      const leafRgb = vis ? normalizeRgb(vis.leaf_rgb) : [0.2, 0.5, 0.15];
      stemR += stemRgb[0]; stemG += stemRgb[1]; stemB += stemRgb[2]; stemCount++;
      leafR += leafRgb[0]; leafG += leafRgb[1]; leafB += leafRgb[2]; leafCount++;

      // Pick index — use the canopy center as the representative point
      this._pickIndex.push({
        kind: 'plant',
        index: i,
        x,
        y,
        z: canopyCenterZ,
        data: p,
      });
    }

    // Average stem/leaf colors across all plants for the merged actor
    const avgStem = stemCount > 0
      ? [stemR / stemCount, stemG / stemCount, stemB / stemCount]
      : FALLBACK_RGB;
    const avgLeaf = leafCount > 0
      ? [leafR / leafCount, leafG / leafCount, leafB / leafCount]
      : FALLBACK_RGB;

    // Trunk actor
    const mergedTrunks = mergePolyDatas(trunkPolys);
    if (mergedTrunks) {
      this._addActor(makeColoredActor(mergedTrunks, avgStem));
    }

    // Canopy actor
    const mergedCanopies = mergePolyDatas(canopyPolys);
    if (mergedCanopies) {
      this._addActor(makeColoredActor(mergedCanopies, avgLeaf));
    }
  }

  // =========================================================================
  // Internal — Flies
  // =========================================================================

  _buildFlies(msg) {
    const flies = msg.flies;
    const visuals = msg.fly_visuals;
    if (!flies || flies.length === 0) return;

    const FLY_RADIUS = 0.08;
    const polys = [];
    let r = 0, g = 0, b = 0, count = 0;

    for (let i = 0; i < flies.length; i++) {
      const f = flies[i];
      const vis = visuals && visuals[i] ? visuals[i] : null;

      const x = f.x;
      const y = f.y;
      const z = f.z != null ? f.z : 0.1;

      polys.push(makeSpherePolyData(x, y, z, FLY_RADIUS));

      const rgb = vis ? normalizeRgb(vis.body_rgb || vis.rgb) : [0.15, 0.12, 0.1];
      r += rgb[0]; g += rgb[1]; b += rgb[2]; count++;

      this._pickIndex.push({
        kind: 'fly',
        index: i,
        x,
        y,
        z,
        data: f,
      });
    }

    const avgColor = count > 0 ? [r / count, g / count, b / count] : FALLBACK_RGB;
    const merged = mergePolyDatas(polys);
    if (merged) {
      this._addActor(makeColoredActor(merged, avgColor));
    }
  }

  // =========================================================================
  // Internal — Fruits
  // =========================================================================

  _buildFruits(msg) {
    const fruits = msg.fruits;
    const fullFruits = msg.full_fruits;
    const visuals = msg.fruit_visuals;
    if (!fruits || fruits.length === 0) return;

    const DEFAULT_FRUIT_RADIUS = 0.1;
    const polys = [];
    let r = 0, g = 0, b = 0, count = 0;

    for (let i = 0; i < fruits.length; i++) {
      const f = fruits[i];
      const full = fullFruits && fullFruits[i] ? fullFruits[i] : null;
      const vis = visuals && visuals[i] ? visuals[i] : null;

      const x = f.x;
      const y = f.y;
      const radius = full && full.radius ? Math.max(0.05, full.radius * 0.1) : DEFAULT_FRUIT_RADIUS;
      // Fruits sit slightly above ground
      const z = radius + 0.01;

      polys.push(makeSpherePolyData(x, y, z, Math.max(0.02, radius)));

      const rgb = vis ? normalizeRgb(vis.rgb || vis.fruit_rgb) : FALLBACK_RGB;
      r += rgb[0]; g += rgb[1]; b += rgb[2]; count++;

      this._pickIndex.push({
        kind: 'fruit',
        index: i,
        x,
        y,
        z,
        data: { ...f, ...(full || {}) },
      });
    }

    const avgColor = count > 0 ? [r / count, g / count, b / count] : FALLBACK_RGB;
    const merged = mergePolyDatas(polys);
    if (merged) {
      this._addActor(makeColoredActor(merged, avgColor));
    }
  }

  // =========================================================================
  // Internal — Seeds
  // =========================================================================

  _buildSeeds(msg) {
    const seeds = msg.seeds;
    const visuals = msg.seed_visuals;
    if (!seeds || seeds.length === 0) return;

    const SEED_RADIUS = 0.12;
    const polys = [];
    let r = 0, g = 0, b = 0, count = 0;

    for (let i = 0; i < seeds.length; i++) {
      const s = seeds[i];
      const vis = visuals && visuals[i] ? visuals[i] : null;

      const x = s.x;
      const y = s.y;
      const z = SEED_RADIUS; // ground level

      polys.push(makeSpherePolyData(x, y, z, SEED_RADIUS));

      const rgb = vis ? normalizeRgb(vis.shell_rgb || vis.rgb) : [0.6, 0.4, 0.25];
      r += rgb[0]; g += rgb[1]; b += rgb[2]; count++;

      this._pickIndex.push({
        kind: 'seed',
        index: i,
        x,
        y,
        z,
        data: s,
      });
    }

    const avgColor = count > 0 ? [r / count, g / count, b / count] : FALLBACK_RGB;
    const merged = mergePolyDatas(polys);
    if (merged) {
      this._addActor(makeColoredActor(merged, avgColor));
    }
  }

  // =========================================================================
  // Internal — Earthworms
  // =========================================================================

  _buildEarthworms(msg) {
    const worms = msg.earthworms;
    const visuals = msg.earthworm_visuals;
    if (!worms || worms.length === 0) return;

    const WORM_RADIUS = 0.015;
    const WORM_LENGTH = 0.12;
    const polys = [];
    let r = 0, g = 0, b = 0, count = 0;

    for (let i = 0; i < worms.length; i++) {
      const w = worms[i];
      const vis = visuals && visuals[i] ? visuals[i] : null;

      const x = w.x;
      const y = w.y;
      // Worms are at/below ground level — show as a horizontal cylinder
      const z = -0.01;

      polys.push(makeCylinderPolyData(x, y, z, WORM_LENGTH, WORM_RADIUS));

      const rgb = vis ? normalizeRgb(vis.rgb || vis.body_rgb) : FALLBACK_RGB;
      r += rgb[0]; g += rgb[1]; b += rgb[2]; count++;

      this._pickIndex.push({
        kind: 'earthworm',
        index: i,
        x,
        y,
        z,
        data: w,
      });
    }

    const avgColor = count > 0 ? [r / count, g / count, b / count] : FALLBACK_RGB;
    const merged = mergePolyDatas(polys);
    if (merged) {
      const actor = makeColoredActor(merged, avgColor);
      this._addActor(actor);
    }
  }

  // =========================================================================
  // Internal — Nematodes
  // =========================================================================

  _buildNematodes(msg) {
    const nematodes = msg.nematodes;
    const visuals = msg.nematode_visuals;
    if (!nematodes || nematodes.length === 0) return;

    const NEMA_RADIUS = 0.008;
    const NEMA_LENGTH = 0.04;
    const polys = [];
    let r = 0, g = 0, b = 0, count = 0;

    for (let i = 0; i < nematodes.length; i++) {
      const n = nematodes[i];
      const vis = visuals && visuals[i] ? visuals[i] : null;

      const x = n.x;
      const y = n.y;
      const z = -0.02; // subsurface

      polys.push(makeCylinderPolyData(x, y, z, NEMA_LENGTH, NEMA_RADIUS));

      const rgb = vis ? normalizeRgb(vis.rgb || vis.body_rgb) : FALLBACK_RGB;
      r += rgb[0]; g += rgb[1]; b += rgb[2]; count++;

      this._pickIndex.push({
        kind: 'nematode',
        index: i,
        x,
        y,
        z,
        data: n,
      });
    }

    const avgColor = count > 0 ? [r / count, g / count, b / count] : FALLBACK_RGB;
    const merged = mergePolyDatas(polys);
    if (merged) {
      this._addActor(makeColoredActor(merged, avgColor));
    }
  }
}
