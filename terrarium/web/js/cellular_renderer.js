// ---------------------------------------------------------------------------
// CellularRenderer — dense cytoplasmic crowding visualization
// ---------------------------------------------------------------------------
// Inspired by Luthey-Schulten Lab's Lattice Microbes / JCVI-syn3A renders.
// Instead of 3-4 orbiting metabolite dots, this shows the ACTUAL molecular
// crowding inside a cell — hundreds of particles packed at realistic density
// (~30% volume fraction for cytoplasm).
//
// Uses the MolecularRenderer's impostor ray-cast shader for performance.
//
// Color scheme (functional classes, 6-8 hues):
//   Energy carriers (ATP/ADP/NADH):  gold/amber
//   Sugars (glucose/fructose/sucrose): cyan
//   Gases (O₂/CO₂):                   red/gray
//   Water:                             blue (translucent)
//   Amino acids/proteins:              light gray
//   Nucleic acids (RNA):               orange
//   Ions/minerals:                     purple
//   Signaling (hormones/VOC):          pink/magenta

// Functional class color palette (Lattice Microbes inspired)
const CELL_COLORS = {
  energy:    [1.0, 0.78, 0.15],   // gold — ATP, ADP, NADH
  sugar:     [0.20, 0.70, 0.95],  // cyan — glucose, fructose, sucrose, trehalose
  gas:       [0.85, 0.25, 0.20],  // red — O₂
  co2:       [0.55, 0.55, 0.55],  // gray — CO₂
  water:     [0.30, 0.50, 0.85],  // blue — water
  protein:   [0.70, 0.70, 0.72],  // light gray — generic proteins
  rna:       [0.90, 0.55, 0.20],  // orange — RNA, nucleotides
  ion:       [0.65, 0.45, 0.80],  // purple — ions, minerals
  signal:    [0.85, 0.40, 0.65],  // pink — hormones, VOC
  membrane:  [0.35, 0.70, 0.35],  // green — membrane lipids
  organelle: [0.25, 0.60, 0.30],  // darker green — chloroplast
  nucleus:   [0.30, 0.35, 0.75],  // blue — nucleus
};

// Map metabolite names to functional classes
function classifyMetabolite(label) {
  const l = (label || '').toLowerCase();
  if (/atp|adp|nadh|nad\+|energy|charge/.test(l)) return 'energy';
  if (/glucose|fructose|sucrose|trehalose|sugar|starch|glycogen|carbon/.test(l)) return 'sugar';
  if (/\bo2\b|oxygen/.test(l)) return 'gas';
  if (/co2|carbon dioxide/.test(l)) return 'co2';
  if (/water|h2o|moisture/.test(l)) return 'water';
  if (/amino|protein|alanine|glycine/.test(l)) return 'protein';
  if (/rna|mrna|trna|nucleotide/.test(l)) return 'rna';
  if (/ion|calcium|potassium|sodium|iron|phosph|nitrate|ammonium/.test(l)) return 'ion';
  if (/hormone|auxin|ethylene|jasmonate|signal|voc/.test(l)) return 'signal';
  return 'protein'; // default
}

class CellularRenderer {
  constructor(scene) {
    this.scene = scene;
    this.group = new THREE.Group();
    this.group.visible = false;
    this.group.name = 'cellularRenderer';
    this.scene.add(this.group);

    // We'll use MolecularRenderer's impostor system for particles
    this._particleMesh = null;
    this._membraneMesh = null;
    this._organelleMeshes = [];
    this._labelSprites = [];

    // Animation
    this._time = 0;
    this._particleData = []; // for Brownian motion animation
  }

  /**
   * Render cellular detail with dense molecular crowding.
   */
  setCellularDetail(detail, worldPosition) {
    if (!detail) return;
    this._clear();

    const cellRadius = 0.06;
    const particles = [];

    // 1. Build membrane shell as lattice cubes (Lattice Microbes style)
    this._buildMembrane(cellRadius);

    // 2. Build organelles as larger colored spheres
    if (detail.organelles) {
      this._buildOrganelles(detail.organelles, cellRadius);
    }

    // 3. Dense metabolite particle field from metabolic_state
    if (detail.metabolic_state) {
      this._buildDenseParticleField(detail.metabolic_state, cellRadius, particles);
    }

    // 4. Gene expression indicators
    if (detail.active_genes) {
      this._buildGeneIndicators(detail.active_genes, cellRadius);
    }

    // 5. Create impostor mesh for all particles
    if (particles.length > 0 && typeof MolecularRenderer !== 'undefined') {
      // Use a temporary MolecularRenderer just for the impostor creation
      const tmpRenderer = new MolecularRenderer(this.scene);
      const mesh = tmpRenderer.createImpostorParticles(particles);
      if (mesh) {
        mesh.name = 'cellularParticles';
        this._particleMesh = mesh;
        this.group.add(mesh);
      }
      // Don't dispose tmpRenderer — we just used it as a factory
      this.scene.remove(tmpRenderer.group);
    }

    // Store particle data for animation
    this._particleData = particles;

    if (worldPosition) {
      this.group.position.copy(worldPosition);
    }
    this.group.visible = true;
  }

  _buildMembrane(radius) {
    // Lattice-cube membrane: semi-transparent green cubes forming a shell
    // (Lattice Microbes style — 10nm voxels as green cubes)
    const cubeSize = radius * 0.12;
    const cubeGeo = new THREE.BoxGeometry(cubeSize, cubeSize, cubeSize);
    const cubeMat = new THREE.MeshPhongMaterial({
      color: new THREE.Color(CELL_COLORS.membrane[0], CELL_COLORS.membrane[1], CELL_COLORS.membrane[2]),
      transparent: true,
      opacity: 0.15,
      depthWrite: false,
      side: THREE.DoubleSide,
    });

    // Place cubes on a spherical shell
    const shellGroup = new THREE.Group();
    shellGroup.name = 'cellMembrane';
    const nCubes = 80;
    for (let i = 0; i < nCubes; i++) {
      // Fibonacci sphere distribution
      const phi = Math.acos(1 - 2 * (i + 0.5) / nCubes);
      const theta = Math.PI * (1 + Math.sqrt(5)) * i;

      const mesh = new THREE.Mesh(cubeGeo, cubeMat);
      mesh.position.set(
        radius * Math.sin(phi) * Math.cos(theta),
        radius * Math.cos(phi),
        radius * Math.sin(phi) * Math.sin(theta)
      );
      // Random slight rotation for organic feel
      mesh.rotation.set(
        Math.random() * 0.3,
        Math.random() * 0.3,
        Math.random() * 0.3
      );
      shellGroup.add(mesh);
    }

    this._membraneMesh = shellGroup;
    this.group.add(shellGroup);
  }

  _buildOrganelles(organelles, cellRadius) {
    const orgColors = {
      'chloroplast':   CELL_COLORS.organelle,
      'mitochondria':  [0.80, 0.25, 0.20],
      'mitochondrion': [0.80, 0.25, 0.20],
      'nucleus':       CELL_COLORS.nucleus,
      'vacuole':       [0.45, 0.55, 0.80],
      'cell wall':     [0.55, 0.48, 0.35],
      'ribosome':      CELL_COLORS.rna,
    };

    const sphereGeo = new THREE.SphereGeometry(1, 12, 8);
    const count = organelles.length;

    for (let i = 0; i < count; i++) {
      const org = organelles[i];
      const name = (org.name || '').toLowerCase();
      const rgb = orgColors[name] || [0.5, 0.5, 0.5];
      const orgCount = org.count || 1;

      // Size proportional to count, within the cell
      const size = Math.max(0.006, Math.min(0.02, orgCount * 0.0005));

      const mat = new THREE.MeshPhongMaterial({
        color: new THREE.Color(rgb[0], rgb[1], rgb[2]),
        shininess: 30,
        transparent: true,
        opacity: 0.75,
      });

      const mesh = new THREE.Mesh(sphereGeo, mat);
      mesh.scale.set(size, size, size);

      // Nucleus centered; others distributed inside
      if (name === 'nucleus') {
        mesh.position.set(0, 0, 0);
        const nSize = Math.max(0.015, size * 1.5);
        mesh.scale.set(nSize, nSize, nSize);
      } else {
        const phi = Math.acos(1 - 2 * (i + 0.5) / count);
        const theta = Math.PI * (1 + Math.sqrt(5)) * i;
        const r = cellRadius * 0.55;
        mesh.position.set(
          r * Math.sin(phi) * Math.cos(theta),
          r * Math.cos(phi),
          r * Math.sin(phi) * Math.sin(theta)
        );
      }

      this._organelleMeshes.push(mesh);
      this.group.add(mesh);
    }
  }

  _buildDenseParticleField(metabolicState, cellRadius, particles) {
    // For each metabolite in the metabolic state, generate proportional
    // number of particles to show molecular crowding
    const innerRadius = cellRadius * 0.85; // inside membrane

    for (const metric of metabolicState) {
      const label = metric.label || '';
      const fill = metric.fraction ?? 0.5;
      if (fill < 0.01) continue;

      const funcClass = classifyMetabolite(label);
      const color = CELL_COLORS[funcClass] || CELL_COLORS.protein;

      // Number of particles proportional to concentration
      // Aim for ~200-500 total particles across all metabolites
      const numParticles = Math.max(2, Math.min(60, Math.round(fill * 40)));

      // Particle radius: smaller for abundant species (realistic crowding)
      const baseRadius = funcClass === 'water' ? 0.0015 :
                         funcClass === 'ion' ? 0.002 :
                         funcClass === 'energy' ? 0.003 :
                         0.0025;

      for (let i = 0; i < numParticles; i++) {
        // Random position inside the cell sphere
        const r = innerRadius * Math.cbrt(Math.random()); // uniform volume distribution
        const phi = Math.acos(2 * Math.random() - 1);
        const theta = Math.random() * Math.PI * 2;

        particles.push({
          position: [
            r * Math.sin(phi) * Math.cos(theta),
            r * Math.cos(phi),
            r * Math.sin(phi) * Math.sin(theta),
          ],
          radius: baseRadius * (0.8 + Math.random() * 0.4), // slight variation
          color: [
            color[0] * (0.85 + Math.random() * 0.15),
            color[1] * (0.85 + Math.random() * 0.15),
            color[2] * (0.85 + Math.random() * 0.15),
          ],
        });
      }
    }

    // Add background "protein soup" — gray particles filling empty space
    // This creates the visual density that makes it look like a real cell
    const soupCount = Math.max(50, 300 - particles.length);
    for (let i = 0; i < soupCount; i++) {
      const r = innerRadius * Math.cbrt(Math.random());
      const phi = Math.acos(2 * Math.random() - 1);
      const theta = Math.random() * Math.PI * 2;

      particles.push({
        position: [
          r * Math.sin(phi) * Math.cos(theta),
          r * Math.cos(phi),
          r * Math.sin(phi) * Math.sin(theta),
        ],
        radius: 0.002 * (0.7 + Math.random() * 0.6),
        color: [
          0.55 + Math.random() * 0.15,
          0.55 + Math.random() * 0.15,
          0.58 + Math.random() * 0.12,
        ],
      });
    }
  }

  _buildGeneIndicators(genes, cellRadius) {
    // Show active genes as small glowing bars along the nucleus surface
    const barGeo = new THREE.BoxGeometry(1, 1, 1);
    const maxGenes = Math.min(genes.length, 12);
    const arcR = cellRadius * 0.15; // near nucleus

    for (let i = 0; i < maxGenes; i++) {
      const gene = genes[i];
      const level = gene.expression_level ?? 0.5;
      if (level < 0.05) continue;

      const barH = 0.012 * level;
      const barW = 0.002;
      const angle = (i / maxGenes) * Math.PI * 2;

      const glow = level > 0.7 ? [0.2, 0.9, 0.3] :
                   level > 0.3 ? [0.9, 0.8, 0.2] :
                                 [0.6, 0.4, 0.4];

      const mat = new THREE.MeshPhongMaterial({
        color: new THREE.Color(glow[0], glow[1], glow[2]),
        emissive: new THREE.Color(glow[0] * 0.4, glow[1] * 0.4, glow[2] * 0.4),
        shininess: 40,
        transparent: true,
        opacity: 0.8,
      });

      const mesh = new THREE.Mesh(barGeo, mat);
      mesh.scale.set(barW, barH, barW);
      mesh.position.set(
        Math.cos(angle) * arcR,
        barH * 0.5,
        Math.sin(angle) * arcR
      );

      this.group.add(mesh);
    }
  }

  /**
   * Render soil as densely packed particles — sand grains, clay, organic matter,
   * water films, mineral crystals, iron oxides, dissolved ions.
   * This is what you see when zooming into a patch of dirt.
   * @param {Object} inspectData - soil inspect data with composition[] and molecular[]
   * @param {THREE.Vector3} worldPosition - where in the scene
   * @param {string} zoomLevel - 'organism' (soil grains), 'cellular' (aggregate interior), 'molecular' (dissolved species)
   */
  setSoilParticleField(inspectData, worldPosition, zoomLevel, soilRGB) {
    this._clear();
    const particles = [];

    // Parse soil composition into particle counts — all values are EMERGENT
    // from the simulation's chemistry engine (not hardcoded)
    const comp = {};
    for (const entry of (inspectData.composition || [])) {
      comp[(entry.label || '').toLowerCase()] = entry.amount || 0;
    }

    // soilRGB: the backend-computed Beer-Lambert color for this cell
    // (already accounts for iron, organic, carbonate, moisture)
    // If not provided, we compute from composition (fallback)
    this._soilBaseRGB = soilRGB || null;

    const fieldRadius = zoomLevel === 'organism' ? 0.15 : 0.08;

    if (zoomLevel === 'organism') {
      // Organism zoom: soil grains visible — sand, silt, clay particles
      this._buildSoilGrains(particles, comp, fieldRadius);
    } else {
      // Cellular zoom: inside a soil aggregate — everything densely packed
      this._buildSoilAggregate(particles, comp, fieldRadius);
    }

    // Create impostor mesh
    if (particles.length > 0 && typeof MolecularRenderer !== 'undefined') {
      const factory = new MolecularRenderer(this.scene);
      const mesh = factory.createImpostorParticles(particles);
      if (mesh) {
        mesh.name = 'soilParticles';
        this._particleMesh = mesh;
        this.group.add(mesh);
      }
      this.scene.remove(factory.group);
    }

    this._particleData = particles;
    if (worldPosition) this.group.position.copy(worldPosition);
    this.group.visible = true;
  }

  _buildSoilGrains(particles, comp, radius) {
    // Emergent soil grains: particle counts, sizes, and colors all derived
    // from the simulation's actual chemistry concentrations.
    // Iron → reddish tint (Burns 1993 d-electron absorption)
    // Organic matter → darker (Beer-Lambert extinction)
    // Calcium carbonate → whitening (Kubelka-Munk scattering)
    // Moisture → darkening (Lobell & Asner 2002)

    const moisture = comp['water'] || 0;
    const organic = (comp['glucose'] || 0) + (comp['amino acids'] || 0) + (comp['nucleotides'] || 0);
    const calcium = comp['exchangeable calcium'] || 0;
    const iron = comp['sorbed ferric hydroxide'] || 0;
    const silicate = comp['dissolved silicate'] || 0;
    const carbonate = comp['calcium bicarbonate complex'] || 0;
    const clay = Math.min(1.0, (calcium + iron) * 5);
    const sand = Math.max(0.2, 1.0 - clay);

    // Emergent base color from mineral content (Beer-Lambert + CPK)
    // Iron → reddish, carbonate → whitish, organic → dark
    const ironTint = Math.min(0.3, iron * 15);    // Fe₂O₃ red
    const whitening = Math.min(0.2, carbonate * 8); // CaCO₃ scatter
    const darkening = Math.min(0.3, organic * 4);   // organic absorption
    const moistDarken = Math.min(0.15, moisture * 0.05); // wet = darker

    // Sand grains — larger, tan/brown, angular
    const sandCount = Math.round(sand * 120);
    for (let i = 0; i < sandCount; i++) {
      const r = radius * Math.cbrt(Math.random());
      const phi = Math.acos(2 * Math.random() - 1);
      const theta = Math.random() * Math.PI * 2;
      const grainSize = (0.004 + Math.random() * 0.006) * (0.8 + sand * 0.4);

      // Emergent sand color: use backend Beer-Lambert soil RGB if available,
      // otherwise derive from composition chemistry
      let baseR, baseG, baseB;
      if (this._soilBaseRGB) {
        // Backend-computed color (fully emergent from molecular optics)
        baseR = this._soilBaseRGB[0];
        baseG = this._soilBaseRGB[1];
        baseB = this._soilBaseRGB[2];
      } else {
        // Fallback: derive from composition (still emergent, just computed client-side)
        baseR = 0.58 + ironTint * 0.8 + whitening * 0.3 - darkening * 0.4 - moistDarken;
        baseG = 0.45 - ironTint * 0.3 + whitening * 0.3 - darkening * 0.3 - moistDarken;
        baseB = 0.28 - ironTint * 0.2 + whitening * 0.4 - darkening * 0.2 - moistDarken;
      }

      particles.push({
        position: [
          r * Math.sin(phi) * Math.cos(theta),
          r * Math.cos(phi),
          r * Math.sin(phi) * Math.sin(theta),
        ],
        radius: grainSize,
        color: [
          Math.max(0.1, baseR + (Math.random() - 0.5) * 0.08),
          Math.max(0.08, baseG + (Math.random() - 0.5) * 0.06),
          Math.max(0.05, baseB + (Math.random() - 0.5) * 0.05),
        ],
      });
    }

    // Clay particles — smaller, gray-brown, more uniform
    const clayCount = Math.round(clay * 80);
    for (let i = 0; i < clayCount; i++) {
      const r = radius * Math.cbrt(Math.random());
      const phi = Math.acos(2 * Math.random() - 1);
      const theta = Math.random() * Math.PI * 2;

      particles.push({
        position: [
          r * Math.sin(phi) * Math.cos(theta),
          r * Math.cos(phi),
          r * Math.sin(phi) * Math.sin(theta),
        ],
        radius: 0.002 + Math.random() * 0.002,
        color: [
          0.45 + Math.random() * 0.08,  // gray-brown
          0.40 + Math.random() * 0.06,
          0.35 + Math.random() * 0.05,
        ],
      });
    }

    // Organic matter — dark brown/black, amorphous
    const organicCount = Math.round(Math.min(organic * 200, 40));
    for (let i = 0; i < organicCount; i++) {
      const r = radius * Math.cbrt(Math.random());
      const phi = Math.acos(2 * Math.random() - 1);
      const theta = Math.random() * Math.PI * 2;

      particles.push({
        position: [
          r * Math.sin(phi) * Math.cos(theta),
          r * Math.cos(phi),
          r * Math.sin(phi) * Math.sin(theta),
        ],
        radius: 0.003 + Math.random() * 0.004,
        color: [
          0.15 + Math.random() * 0.08,  // dark brown
          0.10 + Math.random() * 0.06,
          0.05 + Math.random() * 0.03,
        ],
      });
    }

    // Water films — blue translucent, filling gaps between particles
    const waterCount = Math.round(Math.min(moisture * 30, 50));
    for (let i = 0; i < waterCount; i++) {
      const r = radius * Math.cbrt(Math.random());
      const phi = Math.acos(2 * Math.random() - 1);
      const theta = Math.random() * Math.PI * 2;

      particles.push({
        position: [
          r * Math.sin(phi) * Math.cos(theta),
          r * Math.cos(phi),
          r * Math.sin(phi) * Math.sin(theta),
        ],
        radius: 0.003 + Math.random() * 0.003,
        color: [
          0.25 + Math.random() * 0.05,  // water blue
          0.45 + Math.random() * 0.08,
          0.75 + Math.random() * 0.10,
        ],
      });
    }

    // Mineral crystals — white/sparkle (carbonate, silicate)
    const mineralCount = Math.round((silicate + calcium) * 60);
    for (let i = 0; i < Math.min(mineralCount, 25); i++) {
      const r = radius * Math.cbrt(Math.random());
      const phi = Math.acos(2 * Math.random() - 1);
      const theta = Math.random() * Math.PI * 2;

      particles.push({
        position: [
          r * Math.sin(phi) * Math.cos(theta),
          r * Math.cos(phi),
          r * Math.sin(phi) * Math.sin(theta),
        ],
        radius: 0.002 + Math.random() * 0.003,
        color: [
          0.85 + Math.random() * 0.10,  // white-ish crystal
          0.82 + Math.random() * 0.10,
          0.78 + Math.random() * 0.10,
        ],
      });
    }

    // Iron oxide — rust red particles
    const ironCount = Math.round(iron * 300);
    for (let i = 0; i < Math.min(ironCount, 20); i++) {
      const r = radius * Math.cbrt(Math.random());
      const phi = Math.acos(2 * Math.random() - 1);
      const theta = Math.random() * Math.PI * 2;

      particles.push({
        position: [
          r * Math.sin(phi) * Math.cos(theta),
          r * Math.cos(phi),
          r * Math.sin(phi) * Math.sin(theta),
        ],
        radius: 0.002 + Math.random() * 0.002,
        color: [
          0.65 + Math.random() * 0.15,  // rust red
          0.20 + Math.random() * 0.10,
          0.08 + Math.random() * 0.05,
        ],
      });
    }
  }

  _buildSoilAggregate(particles, comp, radius) {
    // Inside a soil aggregate: densely packed like a cell's cytoplasm
    // Show the molecular soup between mineral surfaces

    const moisture = comp['water'] || 0;

    // Dense mineral matrix (the "walls" of the aggregate)
    for (let i = 0; i < 100; i++) {
      const r = radius * (0.7 + Math.random() * 0.3); // mostly on the outer shell
      const phi = Math.acos(2 * Math.random() - 1);
      const theta = Math.random() * Math.PI * 2;

      particles.push({
        position: [
          r * Math.sin(phi) * Math.cos(theta),
          r * Math.cos(phi),
          r * Math.sin(phi) * Math.sin(theta),
        ],
        radius: 0.003 + Math.random() * 0.003,
        color: [
          0.50 + Math.random() * 0.10,
          0.42 + Math.random() * 0.08,
          0.32 + Math.random() * 0.06,
        ],
      });
    }

    // Dissolved species in the soil solution (inside the aggregate pores)
    const species = [
      { key: 'water',                  color: CELL_COLORS.water,   scale: 15, size: 0.0012 },
      { key: 'glucose',                color: CELL_COLORS.sugar,   scale: 80, size: 0.002 },
      { key: 'oxygen',                 color: CELL_COLORS.gas,     scale: 40, size: 0.0015 },
      { key: 'co2',                    color: CELL_COLORS.co2,     scale: 40, size: 0.0015 },
      { key: 'ammonium',               color: CELL_COLORS.ion,     scale: 60, size: 0.0015 },
      { key: 'nitrate',                color: [0.4, 0.6, 0.9],    scale: 60, size: 0.0015 },
      { key: 'dissolved silicate',     color: [0.8, 0.8, 0.85],   scale: 50, size: 0.002 },
      { key: 'bicarbonate',            color: [0.7, 0.7, 0.75],   scale: 50, size: 0.0015 },
      { key: 'exchangeable calcium',   color: [0.9, 0.9, 0.8],    scale: 30, size: 0.002 },
      { key: 'exchangeable potassium', color: [0.7, 0.5, 0.8],    scale: 30, size: 0.0018 },
      { key: 'amino acids',            color: CELL_COLORS.protein, scale: 60, size: 0.002 },
      { key: 'sorbed ferric hydroxide',color: [0.7, 0.25, 0.1],   scale: 80, size: 0.002 },
    ];

    for (const sp of species) {
      const amount = comp[sp.key] || 0;
      const count = Math.min(50, Math.round(amount * sp.scale));
      for (let i = 0; i < count; i++) {
        const r = radius * 0.65 * Math.cbrt(Math.random()); // inside the mineral shell
        const phi = Math.acos(2 * Math.random() - 1);
        const theta = Math.random() * Math.PI * 2;

        particles.push({
          position: [
            r * Math.sin(phi) * Math.cos(theta),
            r * Math.cos(phi),
            r * Math.sin(phi) * Math.sin(theta),
          ],
          radius: sp.size * (0.8 + Math.random() * 0.4),
          color: [
            sp.color[0] * (0.85 + Math.random() * 0.15),
            sp.color[1] * (0.85 + Math.random() * 0.15),
            sp.color[2] * (0.85 + Math.random() * 0.15),
          ],
        });
      }
    }
  }

  animate(dt) {
    if (!this.group.visible) return;
    this._time += dt;

    // Gentle Brownian rotation of the whole cell view
    this.group.rotation.y += 0.05 * dt;

    // Membrane cubes subtle wobble
    if (this._membraneMesh) {
      this._membraneMesh.children.forEach((cube, i) => {
        cube.rotation.x += Math.sin(this._time * 0.8 + i * 0.5) * 0.002;
        cube.rotation.z += Math.cos(this._time * 0.6 + i * 0.7) * 0.002;
      });
    }
  }

  show() { this.group.visible = true; }
  hide() { this.group.visible = false; }

  _clear() {
    while (this.group.children.length) {
      const child = this.group.children[0];
      this.group.remove(child);
      if (child.geometry) child.geometry.dispose();
      if (child.material) {
        if (child.material.map) child.material.map.dispose();
        child.material.dispose();
      }
      if (child.children) {
        child.traverse(c => {
          if (c.geometry) c.geometry.dispose();
          if (c.material) c.material.dispose();
        });
      }
    }
    this._particleMesh = null;
    this._membraneMesh = null;
    this._organelleMeshes = [];
    this._labelSprites = [];
    this._particleData = [];
    this._time = 0;
  }

  dispose() {
    this._clear();
    this.scene.remove(this.group);
  }
}
