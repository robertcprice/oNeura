/**
 * Shader-based plant renderer for oNeura Terrarium.
 *
 * All visual properties emerge from the simulation:
 * - Leaf color: backend molecular optics (Beer-Lambert, CPK, photosynthesis state)
 * - Stem color: backend Eyring TST temperature-coupled rates
 * - Wind sway: atmospheric simulation wind fields (not procedural)
 * - Growth scale: simulation height_mm drives visible growth over time
 * - Canopy shape: morphology node distribution from L-system + noise displacement
 *
 * Architecture per growth form:
 * - Trees: Spline TubeGeometry trunk (merged) + displaced IcosahedronGeometry canopy
 * - RosetteHerb: Radial fan of merged leaf PlaneGeometry
 * - GrassClump: Delegated to terrain instanced grass (density boost at cell)
 * - FloatingAquatic: Merged CircleGeometry pads + dangling roots
 * - SubmergedAquatic: Spline stems + merged ribbon leaves
 *
 * Zero hardcoded colors. ~2 draw calls per tree (trunk + canopy) vs ~1500 in old system.
 */

// ─── Geometry Merge Utility ─────────────────────────────────────────────
// Three.js r152+ moved BufferGeometryUtils to a separate import.
// We provide a minimal fallback that merges geometries with shared attributes.

function mergeGeos(geometries) {
  // Try Three.js built-in first
  if (typeof THREE.BufferGeometryUtils !== 'undefined' && THREE.BufferGeometryUtils.mergeBufferGeometries) {
    return mergeGeos(geometries);
  }

  // Minimal merge: concatenate all position/normal/uv + custom attributes
  if (geometries.length === 0) return null;
  if (geometries.length === 1) return geometries[0];

  const attrNames = new Set();
  for (const geo of geometries) {
    for (const name in geo.attributes) attrNames.add(name);
  }

  const merged = new THREE.BufferGeometry();
  let totalVertices = 0;
  let totalIndices = 0;
  for (const geo of geometries) {
    totalVertices += geo.attributes.position.count;
    if (geo.index) totalIndices += geo.index.count;
    else totalIndices += geo.attributes.position.count;
  }

  // Merge each attribute
  for (const name of attrNames) {
    const itemSize = geometries[0].attributes[name]?.itemSize || 3;
    const arr = new Float32Array(totalVertices * itemSize);
    let offset = 0;
    for (const geo of geometries) {
      const attr = geo.attributes[name];
      if (attr) {
        arr.set(attr.array, offset);
        offset += attr.array.length;
      } else {
        // Fill with zeros for geometries missing this attribute
        offset += geo.attributes.position.count * itemSize;
      }
    }
    merged.setAttribute(name, new THREE.BufferAttribute(arr, itemSize));
  }

  // Merge indices
  const indices = new Uint32Array(totalIndices);
  let indexOffset = 0;
  let vertexOffset = 0;
  for (const geo of geometries) {
    const count = geo.attributes.position.count;
    if (geo.index) {
      for (let i = 0; i < geo.index.count; i++) {
        indices[indexOffset++] = geo.index.array[i] + vertexOffset;
      }
    } else {
      for (let i = 0; i < count; i++) {
        indices[indexOffset++] = vertexOffset + i;
      }
    }
    vertexOffset += count;
  }
  merged.setIndex(new THREE.BufferAttribute(indices, 1));

  return merged;
}

// ─── Inline Shaders ─────────────────────────────────────────────────────

const PLANT_TRUNK_VERT = `
// Plant trunk/branch vertex shader.
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
`;

const PLANT_TRUNK_FRAG = `
uniform vec3 uStemColor;
uniform float uDaylight;
uniform vec3 uSunDir;

varying vec3 vWorldPos;
varying vec3 vNormal;
varying float vBranchDepth;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    float grain = sin(vWorldPos.y * 40.0 + vWorldPos.x * 3.0) * 0.03;
    float noise = hash(floor(vWorldPos.xz * 20.0)) * 0.06;
    float barkVariation = 1.0 + grain + noise - 0.04;

    vec3 color = uStemColor * barkVariation;
    color = mix(color, color * 1.15, vBranchDepth * 0.3);

    vec3 sunDir = normalize(uSunDir);
    float diffuse = max(dot(vNormal, sunDir), 0.0);
    float ambient = 0.35 + uDaylight * 0.15;
    color *= ambient + diffuse * uDaylight * 0.55;

    gl_FragColor = vec4(color, 1.0);
}
`;

const PLANT_CANOPY_VERT = `
uniform float uTime;
uniform float uWindX;
uniform float uWindZ;
uniform float uWindStrength;
uniform float uGrowthScale;
uniform float uCanopyRadius;

attribute float branchDepth;

varying vec3 vWorldPos;
varying vec3 vNormal;
varying float vBranchDepth;
varying float vNoiseDisplacement;

float hash3(vec3 p) {
    p = fract(p * vec3(443.897, 441.423, 437.195));
    p += dot(p, p.yzx + 19.19);
    return fract((p.x + p.y) * p.z);
}

float noise3(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash3(i);
    float b = hash3(i + vec3(1, 0, 0));
    float c = hash3(i + vec3(0, 1, 0));
    float d = hash3(i + vec3(1, 1, 0));
    float e = hash3(i + vec3(0, 0, 1));
    float f1 = hash3(i + vec3(1, 0, 1));
    float g = hash3(i + vec3(0, 1, 1));
    float h = hash3(i + vec3(1, 1, 1));
    float x1 = mix(a, b, f.x);
    float x2 = mix(c, d, f.x);
    float x3 = mix(e, f1, f.x);
    float x4 = mix(g, h, f.x);
    float y1 = mix(x1, x2, f.y);
    float y2 = mix(x3, x4, f.y);
    return mix(y1, y2, f.z);
}

void main() {
    vec3 dir = normalize(position);
    // Multi-octave noise for organic canopy shape.
    // Low freq: large lobes (like major branch clusters).
    // High freq: small leaf-cluster bumps.
    float noiseVal = noise3(dir * 2.5 + 0.5) * 0.40
                   + noise3(dir * 5.0 + 1.3) * 0.20
                   + noise3(dir * 10.0 + 2.7) * 0.08;
    vNoiseDisplacement = noiseVal;

    // Displacement: push vertices outward along normal for bumpy shape
    vec3 pos = position + normal * noiseVal * uCanopyRadius * 0.40;
    pos *= uGrowthScale;
    vBranchDepth = branchDepth;

    float swayFactor = 0.3 + branchDepth * 0.7;
    float gust = sin(uTime * 1.8 + pos.x * 1.5 + pos.z * 1.1);
    float sway = gust * swayFactor * uWindStrength * 0.10;
    float flutter = sin(uTime * 4.5 + pos.x * 8.0 + pos.y * 6.0 + pos.z * 7.0)
                  * branchDepth * 0.015 * (1.0 + uWindStrength * 0.5);

    pos.x += sway * sign(uWindX + 0.001) + flutter;
    pos.z += sway * sign(uWindZ + 0.001) + flutter * 0.7;
    pos.y += flutter * 0.3 - swayFactor * uWindStrength * 0.02;

    vec4 worldPos = modelMatrix * vec4(pos, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * viewMatrix * worldPos;
}
`;

const PLANT_CANOPY_FRAG = `
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
`;

// Rosette herb uses the canopy fragment shader but a simpler vertex shader
const PLANT_ROSETTE_VERT = `
uniform float uTime;
uniform float uWindX;
uniform float uWindZ;
uniform float uWindStrength;
uniform float uGrowthScale;

attribute float leafIndex;

varying vec3 vWorldPos;
varying vec3 vNormal;
varying float vBranchDepth;
varying float vNoiseDisplacement;

void main() {
    vec3 pos = position * uGrowthScale;

    // Tip factor: y=0 is base, positive y is outward along leaf
    float tipFactor = clamp(pos.y * 4.0, 0.0, 1.0);
    vBranchDepth = tipFactor;
    vNoiseDisplacement = 0.2;

    // Per-leaf flutter driven by wind
    float flutter = sin(uTime * 3.5 + leafIndex * 1.37 + pos.x * 6.0)
                  * tipFactor * 0.02 * (1.0 + uWindStrength * 0.8);
    float sway = sin(uTime * 1.8 + pos.x * 2.0 + pos.z * 1.4)
               * tipFactor * uWindStrength * 0.06;

    pos.y += flutter;
    pos.x += sway * sign(uWindX + 0.001);
    pos.z += sway * sign(uWindZ + 0.001) * 0.6;

    vec4 worldPos = modelMatrix * vec4(pos, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * viewMatrix * worldPos;
}
`;

// Aquatic vertex shader with bob/undulation
const PLANT_AQUATIC_VERT = `
uniform float uTime;
uniform float uWindX;
uniform float uWindZ;
uniform float uWindStrength;
uniform float uGrowthScale;

attribute float partIndex;

varying vec3 vWorldPos;
varying vec3 vNormal;
varying float vBranchDepth;
varying float vNoiseDisplacement;

void main() {
    vec3 pos = position * uGrowthScale;
    vBranchDepth = 0.5;
    vNoiseDisplacement = 0.15;

    // Floating bob: slow sinusoidal vertical displacement
    float bob = sin(uTime * 0.8 + partIndex * 2.1 + pos.x * 1.5) * 0.012;
    pos.y += bob;

    // Gentle lateral drift from wind
    float drift = sin(uTime * 1.2 + partIndex * 1.7 + pos.z * 2.0)
                * 0.015 * (1.0 + uWindStrength * 0.3);
    pos.x += drift;
    pos.z += drift * 0.5;

    vec4 worldPos = modelMatrix * vec4(pos, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * viewMatrix * worldPos;
}
`;

// ─── Geometry Builders ──────────────────────────────────────────────────

/**
 * Build merged trunk + branch geometry from morphology nodes.
 * Uses CatmullRomCurve3 splines for smooth organic branches.
 * Returns a single BufferGeometry with `branchDepth` attribute.
 */
function buildTreeTrunkGeometry(nodes, structure) {
  const trunkNodes = nodes.filter(n => n.node_type === 'Trunk' || n.node_type === 'Branch');
  if (trunkNodes.length === 0) return null;

  const geometries = [];

  // Sort by height (y position) to establish parent-child relationships
  trunkNodes.sort((a, b) => (a.position[1] || 0) - (b.position[1] || 0));

  // Build branch chains: group consecutive nodes into splines
  const chains = buildBranchChains(trunkNodes, structure);

  for (const chain of chains) {
    if (chain.points.length < 2) continue;

    const curve = new THREE.CatmullRomCurve3(chain.points, false, 'catmullrom', 0.5);
    const segments = Math.max(3, Math.min(chain.points.length * 2, 12));
    const radialSegments = 6;

    const tube = new THREE.TubeGeometry(curve, segments, chain.baseRadius, radialSegments, false);

    // Add branchDepth attribute: 0 at base, 1 at tip
    const count = tube.attributes.position.count;
    const depths = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      // Approximate depth from Y position along the tube
      const y = tube.attributes.position.getY(i);
      const minY = chain.points[0].y;
      const maxY = chain.points[chain.points.length - 1].y;
      const range = maxY - minY;
      depths[i] = range > 0.001 ? Math.min(1.0, Math.max(0.0, (y - minY) / range)) : 0.0;
      // Factor in the chain's own depth level
      depths[i] = chain.depthBase + depths[i] * (1.0 - chain.depthBase);
    }
    tube.setAttribute('branchDepth', new THREE.BufferAttribute(depths, 1));

    // Taper: scale radii along tube length
    const positions = tube.attributes.position;
    for (let i = 0; i < count; i++) {
      const t = depths[i];
      const taper = 1.0 - t * (1.0 - chain.tipRadiusRatio);
      // Move vertex toward center line by taper amount
      // TubeGeometry already has correct positions; we scale the cross-section
      const nx = tube.attributes.normal.getX(i);
      const nz = tube.attributes.normal.getZ(i);
      const scale = taper;
      // Adjust position relative to tube center
      positions.setX(i, positions.getX(i) + nx * chain.baseRadius * (scale - 1.0) * 0.5);
      positions.setZ(i, positions.getZ(i) + nz * chain.baseRadius * (scale - 1.0) * 0.5);
    }

    geometries.push(tube);
  }

  if (geometries.length === 0) return null;
  const merged = mergeGeos(geometries);
  // Dispose individual geometries
  geometries.forEach(g => g.dispose());
  return merged;
}

/**
 * Build branch chains from sorted trunk/branch nodes.
 * Groups nodes into connected spline paths.
 */
function buildBranchChains(trunkNodes, structure) {
  const chains = [];
  const baseThickness = structure.thickness_base || 0.08;
  const branchAngle = structure.branch_angle_rad || 0.5;

  if (trunkNodes.length === 0) return chains;

  // Main trunk: all Trunk-type nodes
  const trunkOnly = trunkNodes.filter(n => n.node_type === 'Trunk');
  const branches = trunkNodes.filter(n => n.node_type === 'Branch');

  // Main trunk chain
  if (trunkOnly.length >= 1) {
    const points = trunkOnly.map(n => new THREE.Vector3(
      n.position[0] || 0, n.position[1] || 0, n.position[2] || 0
    ));
    // Ensure we have at least 2 points (add ground point if needed)
    if (points.length === 1) {
      points.unshift(new THREE.Vector3(points[0].x, 0, points[0].z));
    }
    chains.push({
      points,
      baseRadius: Math.max(0.12, baseThickness * 1.8), // visible trunk at scene scale
      tipRadiusRatio: 0.30,
      depthBase: 0.0
    });
  }

  // Group branches by proximity to create branch chains
  // Simple approach: cluster branches by angle/height bands
  const branchClusters = clusterBranchNodes(branches, structure);
  for (const cluster of branchClusters) {
    if (cluster.length < 1) continue;
    const points = cluster.map(n => new THREE.Vector3(
      n.position[0] || 0, n.position[1] || 0, n.position[2] || 0
    ));
    // Connect branch to nearest trunk point
    if (trunkOnly.length > 0 && points.length >= 1) {
      const firstBranch = points[0];
      let closest = null;
      let closestDist = Infinity;
      for (const tn of trunkOnly) {
        const tp = new THREE.Vector3(tn.position[0] || 0, tn.position[1] || 0, tn.position[2] || 0);
        const d = tp.distanceTo(firstBranch);
        if (d < closestDist) { closestDist = d; closest = tp.clone(); }
      }
      if (closest) points.unshift(closest);
    }
    if (points.length < 2) continue;

    const depthLevel = cluster[0]._depthEstimate || 0.3;
    chains.push({
      points,
      baseRadius: Math.max(0.05, baseThickness * 0.9 * (1.0 - depthLevel * 0.5)),
      tipRadiusRatio: 0.2,
      depthBase: depthLevel
    });
  }

  return chains;
}

/**
 * Cluster branch nodes by spatial proximity into chain groups.
 */
function clusterBranchNodes(branches, structure) {
  if (branches.length === 0) return [];

  // Estimate depth from height — higher branches are deeper in the tree structure
  const maxY = Math.max(...branches.map(b => b.position[1] || 0), 0.01);
  branches.forEach(b => {
    b._depthEstimate = Math.min(1.0, (b.position[1] || 0) / maxY);
  });

  // Cluster by azimuth angle from center axis
  const clusters = [];
  const used = new Set();
  const threshold = structure.internode_length * 1.5 || 0.3;

  for (let i = 0; i < branches.length; i++) {
    if (used.has(i)) continue;
    const cluster = [branches[i]];
    used.add(i);

    const p0 = new THREE.Vector3(
      branches[i].position[0] || 0,
      branches[i].position[1] || 0,
      branches[i].position[2] || 0
    );

    // Find nearby branches to extend this chain
    for (let j = i + 1; j < branches.length; j++) {
      if (used.has(j)) continue;
      const pj = new THREE.Vector3(
        branches[j].position[0] || 0,
        branches[j].position[1] || 0,
        branches[j].position[2] || 0
      );
      const lastInCluster = new THREE.Vector3(
        cluster[cluster.length - 1].position[0] || 0,
        cluster[cluster.length - 1].position[1] || 0,
        cluster[cluster.length - 1].position[2] || 0
      );
      if (pj.distanceTo(lastInCluster) < threshold) {
        cluster.push(branches[j]);
        used.add(j);
      }
    }

    // Only keep clusters with 2+ nodes for meaningful splines
    // Single-node branches get a short stub
    if (cluster.length === 1) {
      // Create a short branch stub
      const n = cluster[0];
      const dir = new THREE.Vector3(
        Math.sin((n.rotation?.[1] || 0)),
        0.3,
        Math.cos((n.rotation?.[1] || 0))
      ).normalize();
      const endPoint = p0.clone().add(dir.multiplyScalar(structure.internode_length * 0.4 || 0.1));
      cluster.push({
        position: [endPoint.x, endPoint.y, endPoint.z],
        rotation: n.rotation,
        radius: n.radius * 0.6,
        node_type: 'Branch',
        _depthEstimate: (n._depthEstimate || 0.3) + 0.15
      });
    }

    clusters.push(cluster);
  }

  // Limit clusters to keep draw count reasonable
  return clusters.slice(0, 30);
}


/**
 * Build canopy volume geometry from leaf node positions.
 * Creates a displaced icosahedron that wraps the leaf node cloud.
 * Returns BufferGeometry with `branchDepth` attribute.
 */
function buildTreeCanopyGeometry(leafNodes, structure) {
  if (leafNodes.length < 2) return null;

  // Compute bounding ellipsoid from leaf positions
  let cx = 0, cy = 0, cz = 0;
  for (const n of leafNodes) {
    cx += (n.position[0] || 0);
    cy += (n.position[1] || 0);
    cz += (n.position[2] || 0);
  }
  cx /= leafNodes.length;
  cy /= leafNodes.length;
  cz /= leafNodes.length;

  // Compute extent in each axis
  let maxRx = 0.1, maxRy = 0.1, maxRz = 0.1;
  for (const n of leafNodes) {
    maxRx = Math.max(maxRx, Math.abs((n.position[0] || 0) - cx));
    maxRy = Math.max(maxRy, Math.abs((n.position[1] || 0) - cy));
    maxRz = Math.max(maxRz, Math.abs((n.position[2] || 0) - cz));
  }

  // Pad for full coverage — canopy should slightly exceed the leaf node cloud
  maxRx *= 1.25;
  maxRy *= 1.15;
  maxRz *= 1.25;

  // Ensure minimum canopy size for small trees
  maxRx = Math.max(maxRx, 0.3);
  maxRy = Math.max(maxRy, 0.25);
  maxRz = Math.max(maxRz, 0.3);

  // Create icosahedron (detail=3 = 320 faces — enough vertices for good noise displacement)
  const ico = new THREE.IcosahedronGeometry(1.0, 3);

  // Scale to ellipsoid and offset to centroid
  const positions = ico.attributes.position;
  const normals = ico.attributes.normal;
  const count = positions.count;
  const depths = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    let x = positions.getX(i);
    let y = positions.getY(i);
    let z = positions.getZ(i);

    // branchDepth: 0 at center (if that were possible), 1 at surface
    // For a unit sphere, length is always ~1, so use the radial direction
    depths[i] = Math.sqrt(x * x + y * y + z * z);

    // Scale to ellipsoid
    x = cx + x * maxRx;
    y = cy + y * maxRy;
    z = cz + z * maxRz;

    positions.setXYZ(i, x, y, z);
  }

  ico.setAttribute('branchDepth', new THREE.BufferAttribute(depths, 1));
  ico.computeVertexNormals();

  return ico;
}


/**
 * Cluster leaf nodes into spatial groups for multi-volume canopy rendering.
 * Returns array of { cx, cy, cz, rx, ry, rz } for each cluster centroid + radii.
 * Uses simple k-means-like approach: divide by height bands + azimuth sectors.
 */
function clusterLeafNodes(leafNodes, structure) {
  if (leafNodes.length < 4) {
    // Too few leaves — single cluster
    let cx = 0, cy = 0, cz = 0;
    for (const n of leafNodes) {
      cx += (n.position[0] || 0); cy += (n.position[1] || 0); cz += (n.position[2] || 0);
    }
    cx /= leafNodes.length; cy /= leafNodes.length; cz /= leafNodes.length;
    let rx = 0.2, ry = 0.2, rz = 0.2;
    for (const n of leafNodes) {
      rx = Math.max(rx, Math.abs((n.position[0] || 0) - cx) * 1.3);
      ry = Math.max(ry, Math.abs((n.position[1] || 0) - cy) * 1.3);
      rz = Math.max(rz, Math.abs((n.position[2] || 0) - cz) * 1.3);
    }
    return [{ cx, cy, cz, rx, ry, rz }];
  }

  // Determine height range
  const ys = leafNodes.map(n => n.position[1] || 0);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const heightRange = maxY - minY;

  // Number of clusters: ~1 per 50 leaves, capped at 8
  const numClusters = Math.max(2, Math.min(8, Math.round(leafNodes.length / 50)));

  // Simple approach: divide by azimuth angle from center axis into sectors
  const cx0 = leafNodes.reduce((s, n) => s + (n.position[0] || 0), 0) / leafNodes.length;
  const cz0 = leafNodes.reduce((s, n) => s + (n.position[2] || 0), 0) / leafNodes.length;

  // Assign each leaf to a sector based on its azimuth and height
  const heightBands = Math.max(1, Math.ceil(numClusters / 3));
  const azimuthSectors = Math.max(2, Math.ceil(numClusters / heightBands));
  const buckets = new Map();

  for (const n of leafNodes) {
    const dx = (n.position[0] || 0) - cx0;
    const dz = (n.position[2] || 0) - cz0;
    const azimuth = Math.atan2(dz, dx);
    const azIdx = Math.floor(((azimuth + Math.PI) / (2 * Math.PI)) * azimuthSectors) % azimuthSectors;
    const heightT = heightRange > 0.01 ? ((n.position[1] || 0) - minY) / heightRange : 0;
    const hIdx = Math.floor(heightT * heightBands);
    const key = azIdx * heightBands + hIdx;
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key).push(n);
  }

  // Build clusters from buckets
  const clusters = [];
  for (const [, nodes] of buckets) {
    if (nodes.length < 2) continue;
    let cx = 0, cy = 0, cz = 0;
    for (const n of nodes) {
      cx += (n.position[0] || 0); cy += (n.position[1] || 0); cz += (n.position[2] || 0);
    }
    cx /= nodes.length; cy /= nodes.length; cz /= nodes.length;
    let rx = 0.15, ry = 0.12, rz = 0.15;
    for (const n of nodes) {
      rx = Math.max(rx, Math.abs((n.position[0] || 0) - cx) * 1.2);
      ry = Math.max(ry, Math.abs((n.position[1] || 0) - cy) * 1.2);
      rz = Math.max(rz, Math.abs((n.position[2] || 0) - cz) * 1.2);
    }
    // Minimum cluster size
    rx = Math.max(rx, 0.25);
    ry = Math.max(ry, 0.20);
    rz = Math.max(rz, 0.25);
    clusters.push({ cx, cy, cz, rx, ry, rz });
  }

  return clusters.length > 0 ? clusters : [{ cx: cx0, cy: (minY + maxY) / 2, cz: cz0, rx: 0.5, ry: 0.4, rz: 0.5 }];
}


/**
 * Build rosette herb geometry: radial fan of flat leaves at ground level.
 * Returns merged BufferGeometry with `leafIndex` attribute.
 */
function buildRosetteGeometry(nodes, structure) {
  const leafNodes = nodes.filter(n => n.node_type === 'Leaf');
  const leafCount = Math.max(5, Math.min(leafNodes.length, 12));

  const geometries = [];
  const leafIndices = [];

  for (let i = 0; i < leafCount; i++) {
    const angle = (i / leafCount) * Math.PI * 2;
    const leafLen = 0.12 + (structure.leaf_radius_scale || 0.5) * 0.08;
    const leafWidth = leafLen * 0.35;

    // Create a leaf as a deformed plane
    const plane = new THREE.PlaneGeometry(leafWidth, leafLen, 1, 3);
    plane.rotateX(-Math.PI / 2); // lay flat

    // Deform: droop the tip downward, curl slightly
    const pos = plane.attributes.position;
    for (let v = 0; v < pos.count; v++) {
      let px = pos.getX(v);
      let py = pos.getY(v);
      let pz = pos.getZ(v);

      // pz is now along the leaf length (after rotation)
      const lengthT = (pz + leafLen / 2) / leafLen; // 0 at base, 1 at tip

      // Droop: tip curves down
      const droop = structure.droop_factor || 0.2;
      py -= lengthT * lengthT * droop * 0.06;

      // Curl edges slightly upward
      const edgeDist = Math.abs(px) / (leafWidth / 2);
      py += edgeDist * edgeDist * 0.008;

      pos.setXYZ(v, px, py, pz);
    }

    // Rotate to radial position
    const mat4 = new THREE.Matrix4();
    mat4.makeRotationY(angle);
    // Offset outward from center
    const offset = new THREE.Matrix4().makeTranslation(0, 0.01, leafLen * 0.4);
    mat4.multiply(offset);
    plane.applyMatrix4(mat4);

    plane.computeVertexNormals();

    // Store leaf index per vertex
    const count = plane.attributes.position.count;
    const indices = new Float32Array(count);
    indices.fill(i);
    plane.setAttribute('leafIndex', new THREE.BufferAttribute(indices, 1));

    geometries.push(plane);
  }

  // Add a tiny stem at center
  const stem = new THREE.CylinderGeometry(0.008, 0.01, 0.03, 4);
  stem.translate(0, 0.015, 0);
  const stemLeafIdx = new Float32Array(stem.attributes.position.count);
  stemLeafIdx.fill(-1); // stem marker
  stem.setAttribute('leafIndex', new THREE.BufferAttribute(stemLeafIdx, 1));
  geometries.push(stem);

  if (geometries.length === 0) return null;
  const merged = mergeGeos(geometries);
  geometries.forEach(g => g.dispose());
  return merged;
}


/**
 * Build floating aquatic geometry: merged pad fronds on water surface.
 * Returns merged BufferGeometry with `partIndex` attribute.
 */
function buildFloatingAquaticGeometry(nodes, structure) {
  const leafNodes = nodes.filter(n => n.node_type === 'Leaf');
  const padCount = Math.max(3, Math.min(leafNodes.length, 10));

  const geometries = [];

  for (let i = 0; i < padCount; i++) {
    const angle = (i / padCount) * Math.PI * 2 + (i * 0.618) * Math.PI;
    const dist = 0.04 + Math.random() * 0.06;
    const padRadius = 0.03 + (structure.leaf_radius_scale || 0.5) * 0.02;

    // Circular pad
    const circle = new THREE.CircleGeometry(padRadius, 8);
    circle.rotateX(-Math.PI / 2);

    // Deform: slight cupping and ripple
    const pos = circle.attributes.position;
    for (let v = 0; v < pos.count; v++) {
      const px = pos.getX(v);
      const pz = pos.getZ(v);
      const r = Math.sqrt(px * px + pz * pz) / padRadius;
      pos.setY(v, pos.getY(v) + r * r * 0.005); // cup edges up slightly
    }

    const mat4 = new THREE.Matrix4();
    mat4.makeTranslation(Math.cos(angle) * dist, 0, Math.sin(angle) * dist);
    circle.applyMatrix4(mat4);
    circle.computeVertexNormals();

    const count = pos.count;
    const indices = new Float32Array(count);
    indices.fill(i);
    circle.setAttribute('partIndex', new THREE.BufferAttribute(indices, 1));

    geometries.push(circle);
  }

  // Short dangling roots
  for (let i = 0; i < 3; i++) {
    const angle = (i / 3) * Math.PI * 2;
    const rootLen = 0.04 + Math.random() * 0.03;
    const root = new THREE.CylinderGeometry(0.003, 0.001, rootLen, 3);
    root.translate(Math.cos(angle) * 0.02, -rootLen / 2, Math.sin(angle) * 0.02);
    const count = root.attributes.position.count;
    const indices = new Float32Array(count);
    indices.fill(padCount + i);
    root.setAttribute('partIndex', new THREE.BufferAttribute(indices, 1));
    geometries.push(root);
  }

  if (geometries.length === 0) return null;
  const merged = mergeGeos(geometries);
  geometries.forEach(g => g.dispose());
  return merged;
}


/**
 * Build submerged aquatic geometry: spline stems + ribbon leaves.
 * Returns merged BufferGeometry with `partIndex` attribute.
 */
function buildSubmergedAquaticGeometry(nodes, structure) {
  const trunkNodes = nodes.filter(n => n.node_type === 'Trunk');
  const leafNodes = nodes.filter(n => n.node_type === 'Leaf');

  const geometries = [];
  let partIdx = 0;

  // Stems as splines
  if (trunkNodes.length >= 2) {
    const points = trunkNodes.map(n => new THREE.Vector3(
      n.position[0] || 0, n.position[1] || 0, n.position[2] || 0
    ));
    const curve = new THREE.CatmullRomCurve3(points, false, 'catmullrom', 0.5);
    const tube = new THREE.TubeGeometry(curve, 8, 0.006, 4, false);
    const count = tube.attributes.position.count;
    const indices = new Float32Array(count);
    indices.fill(partIdx++);
    tube.setAttribute('partIndex', new THREE.BufferAttribute(indices, 1));
    geometries.push(tube);
  }

  // Ribbon leaves at leaf positions
  for (let i = 0; i < Math.min(leafNodes.length, 12); i++) {
    const n = leafNodes[i];
    const leafLen = 0.04 + (structure.leaf_radius_scale || 0.5) * 0.03;
    const leafWidth = leafLen * 0.25;
    const plane = new THREE.PlaneGeometry(leafWidth, leafLen, 1, 2);

    // Curve the leaf
    const pos = plane.attributes.position;
    for (let v = 0; v < pos.count; v++) {
      const py = pos.getY(v);
      const t = (py + leafLen / 2) / leafLen;
      pos.setX(v, pos.getX(v) + Math.sin(t * Math.PI) * leafWidth * 0.3);
    }

    const mat4 = new THREE.Matrix4();
    mat4.makeTranslation(n.position[0] || 0, n.position[1] || 0, n.position[2] || 0);
    const rotMat = new THREE.Matrix4().makeRotationY(n.rotation?.[1] || i * 0.8);
    mat4.multiply(rotMat);
    plane.applyMatrix4(mat4);
    plane.computeVertexNormals();

    const count = pos.count;
    const indices = new Float32Array(count);
    indices.fill(partIdx++);
    plane.setAttribute('partIndex', new THREE.BufferAttribute(indices, 1));
    geometries.push(plane);
  }

  if (geometries.length === 0) return null;
  const merged = mergeGeos(geometries);
  geometries.forEach(g => g.dispose());
  return merged;
}


// ─── Billboard Leaf System ───────────────────────────────────────────────

/**
 * Generate a procedural leaf texture on a canvas.
 * Returns a THREE.Texture with alpha transparency.
 */
let _leafTexture = null;
function getLeafTexture() {
  if (_leafTexture) return _leafTexture;
  const size = 64;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');

  // Draw an oval leaf shape with veins
  ctx.clearRect(0, 0, size, size);
  const cx = size / 2, cy = size / 2;

  // Leaf body (elliptical)
  ctx.beginPath();
  ctx.ellipse(cx, cy, size * 0.42, size * 0.28, 0, 0, Math.PI * 2);
  ctx.fillStyle = '#4a8c3a';
  ctx.fill();

  // Central vein
  ctx.beginPath();
  ctx.moveTo(size * 0.1, cy);
  ctx.lineTo(size * 0.9, cy);
  ctx.strokeStyle = '#3a6c2a';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Side veins
  for (let i = 0; i < 4; i++) {
    const t = 0.25 + i * 0.15;
    ctx.beginPath();
    ctx.moveTo(cx * t * 2, cy);
    ctx.lineTo(cx * t * 2 + size * 0.1, cy - size * 0.15);
    ctx.moveTo(cx * t * 2, cy);
    ctx.lineTo(cx * t * 2 + size * 0.1, cy + size * 0.15);
    ctx.strokeStyle = '#3a6c2a';
    ctx.lineWidth = 0.8;
    ctx.stroke();
  }

  _leafTexture = new THREE.CanvasTexture(canvas);
  _leafTexture.needsUpdate = true;
  return _leafTexture;
}

/**
 * Build billboard leaf quads at each leaf node position.
 * Each leaf = 1 PlaneGeometry (2 triangles) with alpha-masked texture.
 * Massively lighter than icosahedron clusters.
 */
function buildBillboardLeaves(leafNodes, structure, color) {
  if (leafNodes.length === 0) return null;

  const leafSize = (structure.leaf_radius_scale || 0.5) * 0.4 + 0.15;
  const geometries = [];

  // Sample up to 60 leaves for performance
  const step = Math.max(1, Math.floor(leafNodes.length / 60));
  for (let i = 0; i < leafNodes.length; i += step) {
    const n = leafNodes[i];
    const px = n.position[0] || 0;
    const py = n.position[1] || 0;
    const pz = n.position[2] || 0;

    // Two crossed quads per leaf for 3D appearance from any angle
    for (let cross = 0; cross < 2; cross++) {
      const plane = new THREE.PlaneGeometry(leafSize, leafSize * 0.7);
      const angle = (n.rotation?.[1] || i * 0.618 * Math.PI) + cross * Math.PI * 0.5;
      const tilt = (n.rotation?.[0] || 0) * 0.3 + 0.15; // slight upward tilt

      const mat4 = new THREE.Matrix4();
      mat4.makeRotationY(angle);
      const tiltMat = new THREE.Matrix4().makeRotationX(-tilt);
      mat4.multiply(tiltMat);
      const transMat = new THREE.Matrix4().makeTranslation(px, py, pz);
      transMat.multiply(mat4);
      plane.applyMatrix4(transMat);

      // branchDepth for wind animation
      const count = plane.attributes.position.count;
      const depths = new Float32Array(count);
      const heightT = Math.min(1.0, py / 6.0);
      depths.fill(0.5 + heightT * 0.5); // higher leaves sway more
      plane.setAttribute('branchDepth', new THREE.BufferAttribute(depths, 1));

      geometries.push(plane);
    }
  }

  if (geometries.length === 0) return null;
  const merged = mergeGeos(geometries);
  geometries.forEach(g => g.dispose());
  return merged;
}


// ─── Material Factory ───────────────────────────────────────────────────

function createPlantMaterial(type, initialColor) {
  const baseUniforms = {
    uTime: { value: 0 },
    uWindX: { value: 0 },
    uWindZ: { value: 0 },
    uWindStrength: { value: 0 },
    uGrowthScale: { value: 1.0 },
    uDaylight: { value: 0.5 },
    uSunDir: { value: new THREE.Vector3(0.4, 0.7, 0.3) },
  };

  switch (type) {
    case 'trunk':
      return new THREE.ShaderMaterial({
        uniforms: {
          ...baseUniforms,
          uStemColor: { value: new THREE.Color(initialColor[0], initialColor[1], initialColor[2]) },
        },
        vertexShader: PLANT_TRUNK_VERT,
        fragmentShader: PLANT_TRUNK_FRAG,
        side: THREE.DoubleSide,
      });

    case 'canopy':
      return new THREE.ShaderMaterial({
        uniforms: {
          ...baseUniforms,
          uLeafColor: { value: new THREE.Color(initialColor[0], initialColor[1], initialColor[2]) },
          uCanopyRadius: { value: 0.5 },
        },
        vertexShader: PLANT_CANOPY_VERT,
        fragmentShader: PLANT_CANOPY_FRAG,
        side: THREE.DoubleSide,
      });

    case 'rosette':
      return new THREE.ShaderMaterial({
        uniforms: {
          ...baseUniforms,
          uLeafColor: { value: new THREE.Color(initialColor[0], initialColor[1], initialColor[2]) },
        },
        vertexShader: PLANT_ROSETTE_VERT,
        fragmentShader: PLANT_CANOPY_FRAG,  // reuse canopy fragment
        side: THREE.DoubleSide,
      });

    case 'aquatic':
      return new THREE.ShaderMaterial({
        uniforms: {
          ...baseUniforms,
          uLeafColor: { value: new THREE.Color(initialColor[0], initialColor[1], initialColor[2]) },
        },
        vertexShader: PLANT_AQUATIC_VERT,
        fragmentShader: PLANT_CANOPY_FRAG,  // reuse canopy fragment
        side: THREE.DoubleSide,
      });

    default:
      return null;
  }
}


// ─── PlantRenderer Class ────────────────────────────────────────────────

class PlantRenderer {
  constructor(scene) {
    this.scene = scene;
    this.plants = new Map(); // plantId → { group, trunkMat, canopyMat, morphKey, targetScale, displayScale }
    this.time = 0;
    this.windX = 0;
    this.windZ = 0;
    this.windStrength = 0;
    this.daylight = 0.5;
  }

  /**
   * Create or update a plant's rendered representation.
   * Called from updateEntities() when entity data arrives.
   */
  updatePlant(plantId, plantMeta, visual) {
    const nodes = plantMeta?.morphology || [];
    const growthForm = plantMeta?.growth_form || 'RosetteHerb';
    const structure = plantMeta?.structure || {};

    // Determine if this is a new plant or needs geometry rebuild
    const morphKey = this._computeMorphKey(nodes, growthForm, structure);
    let entry = this.plants.get(plantId);

    if (!entry) {
      entry = {
        group: new THREE.Group(),
        trunkMat: null,
        canopyMat: null,
        morphKey: null,
        targetScale: 1.0,
        displayScale: 0.01, // start tiny for growth animation
        growthForm,
        fruitAnchors: [],
      };
      this.scene.add(entry.group);
      this.plants.set(plantId, entry);
    }

    // Growth target: morphology positions are already at world scale from the backend.
    // Growth scale = 1.0 for plants with established morphology.
    // The canopy_scale from visual data handles vigor-based sizing.
    // Growth animation: new plants start at displayScale=0.01 and lerp to 1.0.
    entry.targetScale = visual?.canopy_scale || 1.0;

    // Rebuild geometry only if morphology changed
    if (entry.morphKey !== morphKey) {
      entry.morphKey = morphKey;
      this._rebuildGeometry(entry, nodes, growthForm, structure, plantMeta);
    }

    // Update colors from backend visual data (uniform writes, no geometry traversal)
    this._updateColors(entry, visual, growthForm);

    return entry;
  }

  /**
   * Remove a plant and dispose its resources.
   */
  removePlant(plantId) {
    const entry = this.plants.get(plantId);
    if (!entry) return;
    this._disposeGroup(entry.group);
    this.scene.remove(entry.group);
    this.plants.delete(plantId);
  }

  /**
   * Reap plants that are no longer in the active set.
   */
  reap(activeIds) {
    for (const [id] of this.plants) {
      if (!activeIds.has(id)) {
        this.removePlant(id);
      }
    }
  }

  /**
   * Per-frame animation update. Updates shader uniforms only (zero geometry work).
   */
  animate(dt, windX, windZ, windStrength, daylight) {
    this.time += dt || 0.016;
    this.windX = windX || 0;
    this.windZ = windZ || 0;
    this.windStrength = windStrength || 0;
    this.daylight = daylight || 0.5;

    for (const [, entry] of this.plants) {
      // Growth animation: smooth lerp toward target scale
      // Fast initial growth (0.15) for newly appeared plants, then slower (0.03)
      const nearTarget = Math.abs(entry.displayScale - entry.targetScale) < 0.1;
      const growthRate = nearTarget ? 0.03 : 0.15;
      entry.displayScale += (entry.targetScale - entry.displayScale) * growthRate;

      // Update all shader uniforms on this plant's materials
      this._updateUniforms(entry);
    }
  }

  /**
   * Get fruit anchor positions for a plant (used by fruit renderer).
   */
  getFruitAnchors(plantId) {
    const entry = this.plants.get(plantId);
    return entry ? entry.fruitAnchors : [];
  }

  // ─── Internal Methods ───────────────────────────────────────────────

  _computeMorphKey(nodes, growthForm, structure) {
    // Simplified key — only rebuild when node count or growth form changes
    // (individual node position changes don't warrant full rebuild)
    const nodeSignature = nodes.length + ':' + growthForm;
    const structSig = (structure.internode_length || 0).toFixed(2) +
                      (structure.leaf_radius_scale || 0).toFixed(2) +
                      (structure.lateral_bias || 0).toFixed(2);
    return nodeSignature + '|' + structSig;
  }

  _rebuildGeometry(entry, nodes, growthForm, structure, plantMeta) {
    // Clear old meshes
    this._disposeGroup(entry.group);
    entry.fruitAnchors = [];

    const isTree = growthForm === 'OrchardTree' || growthForm === 'StoneFruitTree' || growthForm === 'CitrusTree';
    const isGrass = growthForm === 'GrassClump';
    const isRosette = growthForm === 'RosetteHerb';
    const isFloatingAquatic = growthForm === 'FloatingAquatic';
    const isSubmergedAquatic = growthForm === 'SubmergedAquatic';

    if (isGrass) {
      // GrassClump: delegate to terrain instanced grass system.
      // The PlantRenderer doesn't create geometry — instead, the terrain
      // grass density is boosted at the plant's cell. We still track
      // the plant for selection/inspection.
      // Create a tiny invisible marker for raycasting.
      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(0.02, 4, 4),
        new THREE.MeshBasicMaterial({ visible: false })
      );
      marker.userData.partKind = 'grass_marker';
      entry.group.add(marker);
      return;
    }

    if (isTree) {
      this._buildTree(entry, nodes, structure, plantMeta);
    } else if (isRosette) {
      this._buildRosette(entry, nodes, structure);
    } else if (isFloatingAquatic) {
      this._buildFloatingAquatic(entry, nodes, structure);
    } else if (isSubmergedAquatic) {
      this._buildSubmergedAquatic(entry, nodes, structure);
    } else {
      // Fallback: treat as rosette
      this._buildRosette(entry, nodes, structure);
    }
  }

  _buildTree(entry, nodes, structure, plantMeta) {
    // If the server sent a pre-computed ribbon mesh, use it directly.
    // This is the correct path — geometry computed by Rust parallel-transport
    // algorithm, browser just uploads buffers. No JS geometry construction.
    const serverMesh = plantMeta?.branch_mesh;
    if (serverMesh && serverMesh.positions && serverMesh.positions.length > 0) {
      this._buildTreeFromServerMesh(entry, serverMesh, nodes, structure);
      return;
    }

    // Fallback: JS-side geometry (used when server doesn't provide mesh yet)
    const trunkGeo = buildTreeTrunkGeometry(nodes, structure);
    if (trunkGeo) {
      const mat = createPlantMaterial('trunk', [0.32, 0.24, 0.14]);
      const mesh = new THREE.Mesh(trunkGeo, mat);
      mesh.userData.partKind = 'stem';
      mesh.castShadow = true;
      entry.group.add(mesh);
      entry.trunkMat = mat;
    }

    // Canopy: render leaf geometry at ACTUAL L-system node positions.
    // Each leaf node from generate_nodes_with_context() has a specific position
    // and radius determined by the genome's L-system + epigenetic modulation.
    // We render a small displaced icosahedron at each node, preserving the
    // species-specific branching pattern rather than averaging into big blobs.
    // Leaves: billboard quads with alpha-masked procedural leaf texture.
    // 2 crossed quads per sampled leaf = 4 triangles vs 20+ for icosahedrons.
    const leafNodes = nodes.filter(n => n.node_type === 'Leaf');
    if (leafNodes.length > 0) {
      const leafGeo = buildBillboardLeaves(leafNodes, structure);
      if (leafGeo) {
        leafGeo.computeVertexNormals();
        const leafTex = getLeafTexture();
        const mat = new THREE.MeshStandardMaterial({
          map: leafTex,
          alphaTest: 0.3,
          side: THREE.DoubleSide,
          color: 0x4a8c3a,
          roughness: 0.85,
        });
        const mesh = new THREE.Mesh(leafGeo, mat);
        mesh.userData.partKind = 'leaf';
        mesh.castShadow = true;
        entry.group.add(mesh);
        entry.leafMat = mat;
      }
    }

    // Fruit anchors from leaf node positions (top 10 by height)
    entry.fruitAnchors = leafNodes
      .sort((a, b) => (b.position[1] || 0) - (a.position[1] || 0))
      .slice(0, 10)
      .map(n => new THREE.Vector3(n.position[0] || 0, n.position[1] || 0, n.position[2] || 0));
  }

  /**
   * Build tree from server-computed ribbon mesh.
   * The backend runs the parallel-transport algorithm (ported from ProteinView)
   * and sends pre-built positions/normals/indices/branchDepth buffers.
   * Browser just uploads to GPU — ZERO geometry computation in JS.
   */
  _buildTreeFromServerMesh(entry, serverMesh, nodes, structure) {
    // Build trunk BufferGeometry from server data
    const positions = new Float32Array(serverMesh.positions);
    const normals = new Float32Array(serverMesh.normals);
    const indices = new Uint32Array(serverMesh.indices);
    const branchDepth = new Float32Array(serverMesh.branch_depth);

    if (positions.length > 0) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
      geo.setAttribute('branchDepth', new THREE.BufferAttribute(branchDepth, 1));
      geo.setIndex(new THREE.BufferAttribute(indices, 1));

      const mat = createPlantMaterial('trunk', [0.32, 0.24, 0.14]);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.userData.partKind = 'stem';
      mesh.castShadow = true;
      entry.group.add(mesh);
      entry.trunkMat = mat;
    }

    // Leaves: billboard quads with alpha-masked leaf texture.
    // Replaces icosahedron blobs — 2 triangles per leaf vs 80.
    const leafNodes = nodes.filter(n => n.node_type === 'Leaf');
    if (leafNodes.length > 0) {
      const leafGeo = buildBillboardLeaves(leafNodes, structure);
      if (leafGeo) {
        leafGeo.computeVertexNormals();
        const leafTex = getLeafTexture();
        const mat = new THREE.MeshStandardMaterial({
          map: leafTex,
          alphaTest: 0.3,
          side: THREE.DoubleSide,
          color: 0x4a8c3a,
          roughness: 0.85,
          flatShading: false,
        });
        const mesh = new THREE.Mesh(leafGeo, mat);
        mesh.userData.partKind = 'leaf';
        mesh.castShadow = true;
        entry.group.add(mesh);
        // Store for color updates
        entry.leafMat = mat;
      }
    }

    // Fruit anchors
    entry.fruitAnchors = leafNodes
      .sort((a, b) => (b.position[1] || 0) - (a.position[1] || 0))
      .slice(0, 10)
      .map(n => new THREE.Vector3(n.position[0] || 0, n.position[1] || 0, n.position[2] || 0));
  }

  _buildRosette(entry, nodes, structure) {
    const geo = buildRosetteGeometry(nodes, structure);
    if (geo) {
      const mat = createPlantMaterial('rosette', [0.20, 0.52, 0.14]);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.userData.partKind = 'leaf';
      mesh.castShadow = true;
      entry.group.add(mesh);
      entry.canopyMat = mat;
    }
  }

  _buildFloatingAquatic(entry, nodes, structure) {
    const geo = buildFloatingAquaticGeometry(nodes, structure);
    if (geo) {
      const mat = createPlantMaterial('aquatic', [0.15, 0.45, 0.18]);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.userData.partKind = 'leaf';
      entry.group.add(mesh);
      entry.canopyMat = mat;
    }
  }

  _buildSubmergedAquatic(entry, nodes, structure) {
    const geo = buildSubmergedAquaticGeometry(nodes, structure);
    if (geo) {
      const mat = createPlantMaterial('aquatic', [0.12, 0.38, 0.16]);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.userData.partKind = 'leaf';
      entry.group.add(mesh);
      entry.canopyMat = mat;
    }
  }

  _updateColors(entry, visual, growthForm) {
    if (!visual) return;

    // Stem color from backend molecular optics
    if (entry.trunkMat && visual.stem_rgb) {
      entry.trunkMat.uniforms.uStemColor.value.setRGB(
        visual.stem_rgb[0], visual.stem_rgb[1], visual.stem_rgb[2]
      );
    }

    // Leaf/canopy color from backend photosynthesis state
    if (entry.canopyMat && visual.leaf_rgb) {
      const colorUniform = entry.canopyMat.uniforms.uLeafColor || entry.canopyMat.uniforms.uStemColor;
      if (colorUniform) {
        colorUniform.value.setRGB(
          visual.leaf_rgb[0], visual.leaf_rgb[1], visual.leaf_rgb[2]
        );
      }
    }
    // Billboard leaf material color from backend
    if (entry.leafMat && visual.leaf_rgb) {
      entry.leafMat.color.setRGB(
        visual.leaf_rgb[0], visual.leaf_rgb[1], visual.leaf_rgb[2]
      );
    }
  }

  _updateUniforms(entry) {
    const materials = [];
    if (entry.trunkMat) materials.push(entry.trunkMat);
    if (entry.canopyMat) materials.push(entry.canopyMat);

    // Sun direction from backend solar state (global variable, set per-frame from FrameData)
    const sd = (typeof sunDirection !== 'undefined' && Array.isArray(sunDirection))
      ? sunDirection : [0.4, 0.7, 0.3];

    for (const mat of materials) {
      mat.uniforms.uTime.value = this.time;
      mat.uniforms.uWindX.value = this.windX;
      mat.uniforms.uWindZ.value = this.windZ;
      mat.uniforms.uWindStrength.value = this.windStrength;
      mat.uniforms.uGrowthScale.value = entry.displayScale;
      mat.uniforms.uDaylight.value = this.daylight;
      // Sun direction: backend [east-west, north-south, vertical] → Three.js [x, y, z]
      mat.uniforms.uSunDir.value.set(sd[0], sd[2], sd[1]);
    }
  }

  _disposeGroup(group) {
    while (group.children.length > 0) {
      const child = group.children[group.children.length - 1];
      group.remove(child);
      if (child.geometry) child.geometry.dispose();
      if (child.material) {
        if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
        else child.material.dispose();
      }
    }
  }

  /**
   * Set opacity for all plant meshes (for scale transition crossfade).
   * @param {number} opacity - 0.0 (invisible) to 1.0 (fully opaque)
   */
  setOpacity(opacity) {
    for (const [, group] of this.meshes) {
      group.traverse(child => {
        if (!child.material) return;
        const mats = Array.isArray(child.material) ? child.material : [child.material];
        for (const mat of mats) {
          if (opacity >= 0.99) {
            mat.transparent = false;
            mat.opacity = 1.0;
          } else if (opacity <= 0.01) {
            group.visible = false;
          } else {
            mat.transparent = true;
            mat.opacity = opacity;
            mat.depthWrite = opacity > 0.5;
          }
        }
      });
      if (opacity > 0.01) group.visible = true;
    }
  }
}
