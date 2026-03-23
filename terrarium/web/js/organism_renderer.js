// ---------------------------------------------------------------------------
// OrganismDetailRenderer — tissue indicator tags at organism zoom scale
// ---------------------------------------------------------------------------
// Shows small floating indicator pills near the selected organism at organism
// zoom scale. NOT giant replacement geometry — the existing plant/fly meshes
// remain visible. These are subtle metabolic readouts.

class OrganismDetailRenderer {
  constructor(scene) {
    this.scene = scene;
    this.group = new THREE.Group();
    this.group.visible = false;
    this.group.name = 'organismRenderer';
    this.scene.add(this.group);

    this._sphereGeo = new THREE.SphereGeometry(1, 10, 8);
  }

  // Tissue indicator layout — small tags positioned around the organism
  static PLANT_TISSUES = {
    'leaf':     { color: [0.20, 0.75, 0.20], offset: [0.15, 0.20, 0.0] },
    'stem':     { color: [0.47, 0.31, 0.16], offset: [-0.15, 0.05, 0.0] },
    'root':     { color: [0.55, 0.40, 0.25], offset: [0.0, -0.10, 0.15] },
    'meristem': { color: [0.85, 0.92, 0.35], offset: [0.0, 0.30, -0.10] },
  };

  static FLY_SEGMENTS = {
    'head':     { color: [0.85, 0.75, 0.20], offset: [0.0, 0.03, 0.05] },
    'thorax':   { color: [0.80, 0.70, 0.15], offset: [0.0, 0.03, 0.0] },
    'abdomen':  { color: [0.75, 0.65, 0.10], offset: [0.0, 0.0, -0.05] },
    'wings':    { color: [0.90, 0.92, 0.85], offset: [0.0, 0.05, 0.0] },
    'neural circuit': { color: [0.60, 0.70, 0.90], offset: [0.0, 0.04, 0.04] },
  };

  setOrganismComponents(components, entityKind, worldPosition) {
    if (!components || !components.length) return;
    this._clear();

    const layouts = entityKind === 'fly'
      ? OrganismDetailRenderer.FLY_SEGMENTS
      : OrganismDetailRenderer.PLANT_TISSUES;

    for (const comp of components) {
      const name = (comp.component_name || '').toLowerCase();
      const layout = layouts[name];
      if (!layout) continue;

      const cellCount = comp.cell_count || 100;

      // Small indicator sphere — radius 0.015-0.035, proportional to cell count
      const r = Math.max(0.015, Math.min(0.035, cellCount * 0.00015));

      // Vitality-based color
      const vitalityMetric = (comp.metrics || []).find(m =>
        m.label && m.label.toLowerCase().includes('vitality'));
      const vitality = vitalityMetric?.fraction ?? 1.0;
      const c = layout.color.map((v, i) => {
        const brown = [0.3, 0.15, 0.05][i];
        return v + (brown - v) * (1.0 - vitality);
      });

      const mat = new THREE.MeshPhongMaterial({
        color: new THREE.Color(c[0], c[1], c[2]),
        emissive: new THREE.Color(c[0] * 0.2, c[1] * 0.2, c[2] * 0.2),
        shininess: 40,
        transparent: true,
        opacity: 0.8,
      });

      const mesh = new THREE.Mesh(this._sphereGeo, mat);
      mesh.scale.set(r, r, r);
      mesh.position.set(layout.offset[0], layout.offset[1], layout.offset[2]);
      mesh.userData.inspectRef = {
        kind: entityKind === 'fly' ? 'fly_segment' : 'plant_tissue',
        tissue: comp.component_name,
      };
      this.group.add(mesh);

      // Label sprite
      this._addLabel(
        `${comp.component_name} (${cellCount})`,
        layout.offset,
        r
      );
    }

    if (worldPosition) {
      this.group.position.copy(worldPosition);
    }
    this.group.visible = true;
  }

  _addLabel(text, offset, size) {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 40;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, 256, 40);

    // Background pill
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    const textWidth = ctx.measureText(text).width || 80;
    const pillW = Math.min(240, textWidth + 20);
    const pillX = (256 - pillW) / 2;
    ctx.beginPath();
    ctx.roundRect(pillX, 4, pillW, 32, 8);
    ctx.fill();

    ctx.font = '16px monospace';
    ctx.fillStyle = '#eadfbc';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 128, 20);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const mat = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      opacity: 0.85,
      depthWrite: false,
    });
    const sprite = new THREE.Sprite(mat);
    sprite.position.set(offset[0], offset[1] + size + 0.03, offset[2]);
    sprite.scale.set(0.12, 0.02, 1);
    this.group.add(sprite);
  }

  show() { this.group.visible = true; }
  hide() { this.group.visible = false; }

  _clear() {
    while (this.group.children.length) {
      const child = this.group.children[0];
      this.group.remove(child);
      if (child.material) {
        if (child.material.map) child.material.map.dispose();
        child.material.dispose();
      }
    }
  }

  dispose() {
    this._clear();
    this.scene.remove(this.group);
    this._sphereGeo.dispose();
  }
}
