/**
 * ProvenanceTracker -- Provenance tracking for the oNeura scientific viewer.
 *
 * Connects visible pixels/entities to their computation derivation chains
 * from the Rust backend. Every color, every rate, every reaction traces
 * back to quantum descriptors, Eyring TST, and literature citations.
 *
 * Cache: LRU Map, max 50 entries.
 * Panel: #derivation-panel DOM element, created on demand.
 */

const CACHE_MAX = 50;

export class ProvenanceTracker {
  constructor() {
    /** @type {Map<string, {kind: string, index: number, worldPos: number[]}>} */
    this._entities = new Map();

    /** @type {Map<string, {result: object, accessOrder: number}>} */
    this._cache = new Map();

    /** Monotonic counter for LRU ordering. */
    this._accessCounter = 0;

    /** @type {HTMLElement|null} */
    this._panel = null;
  }

  // ---------------------------------------------------------------------------
  // Entity registration
  // ---------------------------------------------------------------------------

  /**
   * Register an entity for position-based picking.
   * @param {string} entityId - Unique identifier (e.g., "plant_0", "fly_3")
   * @param {string} kind - Entity kind ("plant", "fly", "soil", "fruit", etc.)
   * @param {number} index - Entity index within its kind
   * @param {number[]} worldPos - [x, y, z] position in world coordinates
   */
  registerEntity(entityId, kind, index, worldPos) {
    this._entities.set(entityId, {
      kind,
      index,
      worldPos: Array.isArray(worldPos) ? worldPos.slice() : [0, 0, 0],
    });
  }

  /**
   * Remove a registered entity.
   * @param {string} entityId
   */
  unregisterEntity(entityId) {
    this._entities.delete(entityId);
  }

  /**
   * Get a registered entity by ID.
   * @param {string} entityId
   * @returns {{kind: string, index: number, worldPos: number[]}|undefined}
   */
  getEntity(entityId) {
    return this._entities.get(entityId);
  }

  /**
   * Clear all registered entities.
   */
  clearEntities() {
    this._entities.clear();
  }

  // ---------------------------------------------------------------------------
  // API fetching with LRU cache
  // ---------------------------------------------------------------------------

  /**
   * Build a cache key from inspect params.
   * @param {object} params
   * @returns {string}
   */
  _cacheKey(params) {
    const parts = [];
    // Deterministic key order
    const keys = Object.keys(params).sort();
    for (const k of keys) {
      if (params[k] !== undefined && params[k] !== null) {
        parts.push(`${k}=${params[k]}`);
      }
    }
    return parts.join('&');
  }

  /**
   * Evict the least-recently-used entry when cache exceeds CACHE_MAX.
   */
  _evictLRU() {
    if (this._cache.size <= CACHE_MAX) return;

    let oldestKey = null;
    let oldestOrder = Infinity;
    for (const [key, entry] of this._cache) {
      if (entry.accessOrder < oldestOrder) {
        oldestOrder = entry.accessOrder;
        oldestKey = key;
      }
    }
    if (oldestKey !== null) {
      this._cache.delete(oldestKey);
    }
  }

  /**
   * Fetch derivation data from /api/inspect.
   *
   * Results are cached with LRU eviction at 50 entries.
   *
   * @param {object} params - Query params for /api/inspect
   *   (kind, index, x, y, scale, molecule, tissue, atom_index)
   * @returns {Promise<object>} The parsed inspect response
   */
  async fetchDerivation(params) {
    const key = this._cacheKey(params);

    // Cache hit -- update access order and return
    const cached = this._cache.get(key);
    if (cached) {
      cached.accessOrder = ++this._accessCounter;
      return cached.result;
    }

    // Build query string
    const qs = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null) {
        qs.set(k, String(v));
      }
    }

    const resp = await fetch(`/api/inspect?${qs.toString()}`);
    if (!resp.ok) {
      throw new Error(`Inspect API error: ${resp.status} ${resp.statusText}`);
    }

    const result = await resp.json();

    // Store in cache with current access order
    this._cache.set(key, {
      result,
      accessOrder: ++this._accessCounter,
    });
    this._evictLRU();

    return result;
  }

  /**
   * Invalidate all cached results.
   */
  clearCache() {
    this._cache.clear();
    this._accessCounter = 0;
  }

  // ---------------------------------------------------------------------------
  // Derivation formatting
  // ---------------------------------------------------------------------------

  /**
   * Convert an inspect API response into HTML for the derivation panel.
   *
   * Shows: entity title/subtitle, summary metrics as labeled bars,
   * molecular detail (atoms, bonds, optical derivation, Eyring TST rates),
   * and literature citations.
   *
   * @param {object} inspectResult - Response from /api/inspect
   * @returns {string} HTML string
   */
  formatDerivation(inspectResult) {
    const sections = [];

    // --- Header ---
    const title = inspectResult.title || 'Unknown Entity';
    const subtitle = inspectResult.subtitle || '';
    const kind = inspectResult.kind || '';
    const scale = inspectResult.scale || 'ecosystem';

    sections.push(
      `<div class="prov-header">`,
      `  <div class="prov-title">${_esc(title)}</div>`,
      subtitle ? `  <div class="prov-subtitle">${_esc(subtitle)}</div>` : '',
      `  <div class="prov-scale-badge">${_esc(scale)}</div>`,
      `</div>`
    );

    // --- Summary metrics as bars ---
    const summary = inspectResult.summary;
    if (summary && summary.length > 0) {
      sections.push(`<div class="prov-section">`);
      sections.push(`  <div class="prov-section-title">Summary</div>`);
      for (const metric of summary) {
        sections.push(this._formatMetricBar(metric));
      }
      sections.push(`</div>`);
    }

    // --- Composition ---
    const composition = inspectResult.composition;
    if (composition && composition.length > 0) {
      sections.push(`<div class="prov-section">`);
      sections.push(`  <div class="prov-section-title">Composition</div>`);
      for (const comp of composition) {
        sections.push(
          `  <div class="prov-comp-row">`,
          `    <span class="prov-comp-label">${_esc(comp.label)}</span>`,
          `    <span class="prov-comp-amount">${_fmtNum(comp.amount)}</span>`,
          `  </div>`
        );
      }
      sections.push(`</div>`);
    }

    // --- Molecular detail ---
    const mol = inspectResult.molecular_detail;
    if (mol) {
      sections.push(this._formatMolecularDetail(mol));
    }

    // --- Notes ---
    const notes = inspectResult.notes;
    if (notes && notes.length > 0) {
      sections.push(`<div class="prov-section">`);
      sections.push(`  <div class="prov-section-title">Notes</div>`);
      for (const note of notes) {
        sections.push(`  <div class="prov-note">${_esc(note)}</div>`);
      }
      sections.push(`</div>`);
    }

    return sections.join('\n');
  }

  /**
   * Format a single InspectMetric as a labeled bar.
   * @param {{label: string, value: string, fraction?: number}} metric
   * @returns {string} HTML
   */
  _formatMetricBar(metric) {
    const fraction = metric.fraction;
    const hasBar = fraction !== undefined && fraction !== null;
    const pct = hasBar ? Math.max(0, Math.min(100, fraction * 100)) : 0;

    const lines = [
      `  <div class="prov-metric">`,
      `    <div class="prov-metric-header">`,
      `      <span class="prov-metric-label">${_esc(metric.label)}</span>`,
      `      <span class="prov-metric-value">${_esc(metric.value)}</span>`,
      `    </div>`,
    ];

    if (hasBar) {
      lines.push(
        `    <div class="prov-metric-bar-track">`,
        `      <div class="prov-metric-bar-fill" style="width:${pct.toFixed(1)}%"></div>`,
        `    </div>`
      );
    }

    lines.push(`  </div>`);
    return lines.join('\n');
  }

  /**
   * Format the full molecular detail section: atoms, bonds, quantum
   * descriptors, optical derivation, and Eyring TST rate derivation.
   *
   * @param {object} mol - MolecularDetail from the backend
   * @returns {string} HTML
   */
  _formatMolecularDetail(mol) {
    const lines = [];

    lines.push(`<div class="prov-section prov-molecular">`);
    lines.push(`  <div class="prov-section-title">Molecular Detail</div>`);

    // Identity
    const molName = mol.name || 'Unknown';
    const formula = mol.formula || '';
    const mw = mol.molecular_weight;

    lines.push(
      `  <div class="prov-mol-identity">`,
      `    <span class="prov-mol-name">${_esc(molName)}</span>`,
      formula ? ` <span class="prov-mol-formula">${_esc(formula)}</span>` : '',
      mw !== undefined ? ` <span class="prov-mol-mw">${_fmtNum(mw)} Da</span>` : '',
      `  </div>`
    );

    // Atom summary
    const atoms = mol.atoms;
    if (atoms && atoms.length > 0) {
      lines.push(`  <div class="prov-subsection">`);
      lines.push(`    <div class="prov-subsection-title">Atoms (${atoms.length})</div>`);
      lines.push(`    <div class="prov-atom-grid">`);

      // Group atoms by element for a compact display
      const elementCounts = new Map();
      for (const atom of atoms) {
        const sym = atom.symbol || atom.element || '?';
        const entry = elementCounts.get(sym);
        if (entry) {
          entry.count++;
        } else {
          elementCounts.set(sym, {
            count: 1,
            cpk: atom.cpk_color || [128, 128, 128],
            vdw: atom.vdw_radius,
            charge: atom.formal_charge,
            electronConfig: atom.electron_config || '',
          });
        }
      }

      for (const [sym, info] of elementCounts) {
        const cpkHex = _rgbHex(info.cpk);
        lines.push(
          `      <div class="prov-atom-chip" style="border-color:${cpkHex}">`,
          `        <span class="prov-atom-symbol" style="color:${cpkHex}">${_esc(sym)}</span>`,
          `        <span class="prov-atom-count">&times;${info.count}</span>`,
          info.vdw !== undefined
            ? `        <span class="prov-atom-detail">r=${_fmtNum(info.vdw)}\u00C5</span>`
            : '',
          info.charge !== undefined && info.charge !== 0
            ? `        <span class="prov-atom-detail">q=${info.charge > 0 ? '+' : ''}${info.charge}</span>`
            : '',
          info.electronConfig
            ? `        <span class="prov-atom-econfig">${_esc(info.electronConfig)}</span>`
            : '',
          `      </div>`
        );
      }

      lines.push(`    </div>`);
      lines.push(`  </div>`);
    }

    // Bond summary
    const bonds = mol.bonds;
    if (bonds && bonds.length > 0) {
      lines.push(`  <div class="prov-subsection">`);
      lines.push(`    <div class="prov-subsection-title">Bonds (${bonds.length})</div>`);

      // Group bonds by order
      const orderCounts = new Map();
      for (const bond of bonds) {
        const order = bond.order || 'single';
        orderCounts.set(order, (orderCounts.get(order) || 0) + 1);
      }

      lines.push(`    <div class="prov-bond-summary">`);
      for (const [order, count] of orderCounts) {
        lines.push(
          `      <span class="prov-bond-chip">${_esc(order)}: ${count}</span>`
        );
      }
      lines.push(`    </div>`);
      lines.push(`  </div>`);
    }

    // Quantum descriptors
    const qd = mol.quantum_descriptor;
    if (qd) {
      lines.push(`  <div class="prov-subsection">`);
      lines.push(`    <div class="prov-subsection-title">Quantum Descriptors</div>`);
      lines.push(`    <div class="prov-qd-table">`);

      const descriptors = [
        ['Ground state energy', qd.ground_state_energy_ev, 'eV'],
        ['Energy per atom', qd.ground_state_energy_per_atom_ev, 'eV/atom'],
        ['Dipole moment', qd.dipole_magnitude_e_angstrom, 'e\u00B7\u00C5'],
        ['Mean |eff. charge|', qd.mean_abs_effective_charge, 'e'],
        ['Charge span', qd.charge_span, 'e'],
        ['LDA exchange pot.', qd.mean_lda_exchange_potential_ev, 'eV'],
        ['Frontier occ.', qd.frontier_occupancy_fraction, ''],
      ];

      for (const [label, val, unit] of descriptors) {
        if (val === undefined || val === null) continue;
        lines.push(
          `      <div class="prov-qd-row">`,
          `        <span class="prov-qd-label">${_esc(label)}</span>`,
          `        <span class="prov-qd-value">${_fmtNum(val)} ${_esc(unit)}</span>`,
          `      </div>`
        );
      }

      lines.push(`    </div>`);
      lines.push(`  </div>`);
    }

    // Derivation chain
    const dc = mol.derivation_chain;
    if (dc) {
      lines.push(this._formatDerivationChain(dc, atoms));
    }

    lines.push(`</div>`);
    return lines.join('\n');
  }

  /**
   * Format the derivation chain: optical path and Eyring TST rate path.
   *
   * Shows the full computation chain:
   *   atoms -> bonds -> quantum descriptors -> Eyring TST -> rate@T
   *
   * @param {object} dc - DerivationChain
   * @param {object[]|undefined} atoms - Atom list (for cross-referencing)
   * @returns {string} HTML
   */
  _formatDerivationChain(dc, atoms) {
    const lines = [];

    lines.push(`  <div class="prov-subsection prov-derivation-chain">`);
    lines.push(`    <div class="prov-subsection-title">Derivation Chain</div>`);

    // Computation flow diagram
    lines.push(
      `    <div class="prov-chain-flow">`,
      `      <span class="prov-chain-node">Atoms</span>`,
      `      <span class="prov-chain-arrow">\u2192</span>`,
      `      <span class="prov-chain-node">Bonds</span>`,
      `      <span class="prov-chain-arrow">\u2192</span>`,
      `      <span class="prov-chain-node">Quantum Descriptors</span>`,
      `      <span class="prov-chain-arrow">\u2192</span>`,
      `      <span class="prov-chain-node">Eyring TST</span>`,
      `      <span class="prov-chain-arrow">\u2192</span>`,
      `      <span class="prov-chain-node">rate@T</span>`,
      `    </div>`
    );

    // Optical derivation
    const opt = dc.optical;
    if (opt) {
      const cpkR = opt.cpk_rgb ? opt.cpk_rgb[0] : 0;
      const cpkG = opt.cpk_rgb ? opt.cpk_rgb[1] : 0;
      const cpkB = opt.cpk_rgb ? opt.cpk_rgb[2] : 0;
      // cpk_rgb from OpticalDerivation is f32 0.0-1.0
      const r8 = Math.round(cpkR * 255);
      const g8 = Math.round(cpkG * 255);
      const b8 = Math.round(cpkB * 255);
      const swatchColor = `rgb(${r8},${g8},${b8})`;

      lines.push(
        `    <div class="prov-optical">`,
        `      <div class="prov-optical-title">Optical Derivation (Beer-Lambert)</div>`,
        `      <div class="prov-optical-row">`,
        `        <span class="prov-optical-swatch" style="background:${swatchColor}"></span>`,
        `        <span class="prov-optical-label">CPK molecular color</span>`,
        `        <span class="prov-optical-value">rgb(${r8}, ${g8}, ${b8})</span>`,
        `      </div>`,
        `      <div class="prov-optical-row">`,
        `        <span class="prov-optical-label">Molar extinction</span>`,
        `        <span class="prov-rate-value">${_fmtSci(opt.molar_extinction)}</span>`,
        `      </div>`,
        `      <div class="prov-optical-row">`,
        `        <span class="prov-optical-label">Scattering cross-section</span>`,
        `        <span class="prov-rate-value">${_fmtSci(opt.scattering_cross_section)}</span>`,
        `      </div>`,
        `    </div>`
      );
    }

    // Rate derivation (Eyring TST)
    const rates = dc.rates;
    if (rates && rates.length > 0) {
      lines.push(
        `    <div class="prov-rates">`,
        `      <div class="prov-rates-title">Rate Derivation (Eyring TST)</div>`
      );

      for (const rate of rates) {
        const rateAtT = rate.rate_at_current_temp;
        const hasLiveRate = rateAtT !== undefined && rateAtT !== null;

        lines.push(
          `      <div class="prov-rate-card">`,
          `        <div class="prov-rate-pathway">${_esc(rate.pathway)}</div>`,
          `        <div class="prov-rate-details">`,
          `          <div class="prov-rate-row">`,
          `            <span class="prov-rate-label">Bond</span>`,
          `            <span class="prov-rate-value">${_esc(rate.bond_type)}</span>`,
          `          </div>`,
          `          <div class="prov-rate-row">`,
          `            <span class="prov-rate-label">BDE</span>`,
          `            <span class="prov-rate-value">${_fmtNum(rate.bond_energy_ev)} eV</span>`,
          `          </div>`,
          `          <div class="prov-rate-row">`,
          `            <span class="prov-rate-label">Enzyme eff.</span>`,
          `            <span class="prov-rate-value">${_fmtPct(rate.enzyme_efficiency)}</span>`,
          `          </div>`,
          `          <div class="prov-rate-row">`,
          `            <span class="prov-rate-label">V<sub>max</sub> @25\u00B0C</span>`,
          `            <span class="prov-rate-value">${_fmtSci(rate.vmax_25)}</span>`,
          `          </div>`,
        );

        if (hasLiveRate) {
          lines.push(
            `          <div class="prov-rate-row prov-rate-live">`,
            `            <span class="prov-rate-label">rate@T</span>`,
            `            <span class="prov-rate-value prov-rate-live-value">${_fmtSci(rateAtT)}</span>`,
            `          </div>`
          );
        }

        lines.push(
          `          <div class="prov-rate-citation">`,
          `            <span class="prov-citation">${_esc(rate.citation)}</span>`,
          `          </div>`,
          `        </div>`,
          `      </div>`
        );
      }

      lines.push(`    </div>`);
    }

    lines.push(`  </div>`);
    return lines.join('\n');
  }

  // ---------------------------------------------------------------------------
  // Panel management
  // ---------------------------------------------------------------------------

  /**
   * Get or create the #derivation-panel DOM element.
   * @returns {HTMLElement}
   */
  _ensurePanel() {
    if (this._panel) return this._panel;

    let panel = document.getElementById('derivation-panel');
    if (!panel) {
      panel = document.createElement('div');
      panel.id = 'derivation-panel';
      document.body.appendChild(panel);
    }

    // Apply base styles if not already styled by external CSS
    if (!panel.dataset.styled) {
      _applyPanelStyles(panel);
      panel.dataset.styled = '1';
    }

    this._panel = panel;
    return panel;
  }

  /**
   * Display the derivation panel with the given HTML content.
   * @param {string} html
   */
  showPanel(html) {
    const panel = this._ensurePanel();
    panel.innerHTML = html;
    panel.style.display = 'block';
    panel.style.opacity = '1';
    panel.style.pointerEvents = 'auto';

    // Accessibility: make panel focusable and announce it
    panel.setAttribute('role', 'complementary');
    panel.setAttribute('aria-label', 'Derivation chain detail');
    panel.setAttribute('tabindex', '-1');
    panel.focus();
  }

  /**
   * Hide the derivation panel.
   */
  hidePanel() {
    if (this._panel) {
      this._panel.style.opacity = '0';
      this._panel.style.pointerEvents = 'none';
      // Allow transition to finish before hiding
      setTimeout(() => {
        if (this._panel && this._panel.style.opacity === '0') {
          this._panel.style.display = 'none';
        }
      }, 200);
    }
  }

  /**
   * Whether the panel is currently visible.
   * @returns {boolean}
   */
  isPanelVisible() {
    return (
      this._panel !== null &&
      this._panel.style.display !== 'none' &&
      this._panel.style.opacity !== '0'
    );
  }

  /**
   * Convenience: fetch derivation for params, format, and show the panel.
   * Returns the raw inspect result for further use.
   *
   * @param {object} params
   * @returns {Promise<object>} The inspect result
   */
  async inspectAndShow(params) {
    const result = await this.fetchDerivation(params);
    const html = this.formatDerivation(result);
    this.showPanel(html);
    return result;
  }
}

// =============================================================================
// Private helpers
// =============================================================================

/**
 * Escape HTML special characters to prevent XSS.
 * @param {string} str
 * @returns {string}
 */
function _esc(str) {
  if (typeof str !== 'string') return String(str ?? '');
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * Format a number for display: up to 4 significant figures.
 * @param {number} n
 * @returns {string}
 */
function _fmtNum(n) {
  if (n === undefined || n === null) return '--';
  if (typeof n !== 'number') return String(n);
  if (Number.isInteger(n)) return n.toLocaleString();
  if (Math.abs(n) >= 100) return n.toFixed(1);
  if (Math.abs(n) >= 1) return n.toFixed(2);
  if (Math.abs(n) >= 0.01) return n.toFixed(4);
  return n.toExponential(3);
}

/**
 * Format a number in scientific notation.
 * @param {number} n
 * @returns {string}
 */
function _fmtSci(n) {
  if (n === undefined || n === null) return '--';
  if (typeof n !== 'number') return String(n);
  if (n === 0) return '0';
  if (Math.abs(n) >= 0.01 && Math.abs(n) < 10000) return _fmtNum(n);
  return n.toExponential(3);
}

/**
 * Format a fraction as a percentage string.
 * @param {number} f - A value in [0, 1]
 * @returns {string}
 */
function _fmtPct(f) {
  if (f === undefined || f === null) return '--';
  return (f * 100).toFixed(1) + '%';
}

/**
 * Convert [R, G, B] (0-255 integers) to a hex color string.
 * @param {number[]} rgb
 * @returns {string}
 */
function _rgbHex(rgb) {
  if (!rgb || rgb.length < 3) return '#808080';
  const r = Math.round(Math.max(0, Math.min(255, rgb[0])));
  const g = Math.round(Math.max(0, Math.min(255, rgb[1])));
  const b = Math.round(Math.max(0, Math.min(255, rgb[2])));
  return '#' + ((1 << 24) | (r << 16) | (g << 8) | b).toString(16).slice(1);
}

/**
 * Apply inline styles to the derivation panel. These are defaults that can
 * be overridden by external CSS targeting #derivation-panel.
 *
 * Design: dark translucent sidebar, monospace accents for scientific data,
 * accessible contrast ratios (WCAG AA on dark backgrounds).
 *
 * @param {HTMLElement} panel
 */
function _applyPanelStyles(panel) {
  Object.assign(panel.style, {
    position: 'fixed',
    top: '0',
    right: '0',
    width: '360px',
    maxWidth: '90vw',
    height: '100vh',
    overflowY: 'auto',
    overflowX: 'hidden',
    background: 'rgba(10, 12, 16, 0.92)',
    backdropFilter: 'blur(12px)',
    WebkitBackdropFilter: 'blur(12px)',
    color: '#e0e0e0',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    fontSize: '13px',
    lineHeight: '1.5',
    padding: '16px',
    boxSizing: 'border-box',
    borderLeft: '1px solid rgba(255,255,255,0.08)',
    zIndex: '10000',
    display: 'none',
    opacity: '0',
    pointerEvents: 'none',
    transition: 'opacity 0.2s ease',
  });

  // Inject scoped CSS for child elements
  const styleId = 'prov-tracker-styles';
  if (!document.getElementById(styleId)) {
    const style = document.createElement('style');
    style.id = styleId;
    style.textContent = `
      #derivation-panel .prov-header {
        margin-bottom: 12px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
      }
      #derivation-panel .prov-title {
        font-size: 16px;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 2px;
      }
      #derivation-panel .prov-subtitle {
        font-size: 12px;
        color: #a0a0a0;
      }
      #derivation-panel .prov-scale-badge {
        display: inline-block;
        margin-top: 6px;
        padding: 2px 8px;
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #80cbc4;
        border: 1px solid rgba(128,203,196,0.3);
        border-radius: 3px;
      }
      #derivation-panel .prov-section {
        margin-bottom: 14px;
      }
      #derivation-panel .prov-section-title {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #90a4ae;
        margin-bottom: 6px;
      }
      #derivation-panel .prov-metric {
        margin-bottom: 6px;
      }
      #derivation-panel .prov-metric-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 2px;
      }
      #derivation-panel .prov-metric-label {
        color: #b0bec5;
        font-size: 12px;
      }
      #derivation-panel .prov-metric-value {
        font-family: "SF Mono", "Fira Code", "Cascadia Code", monospace;
        font-size: 12px;
        color: #ffffff;
      }
      #derivation-panel .prov-metric-bar-track {
        height: 3px;
        background: rgba(255,255,255,0.08);
        border-radius: 1.5px;
        overflow: hidden;
      }
      #derivation-panel .prov-metric-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #26a69a, #80cbc4);
        border-radius: 1.5px;
        transition: width 0.3s ease;
      }
      #derivation-panel .prov-comp-row {
        display: flex;
        justify-content: space-between;
        padding: 2px 0;
        font-size: 12px;
      }
      #derivation-panel .prov-comp-label {
        color: #b0bec5;
      }
      #derivation-panel .prov-comp-amount {
        font-family: "SF Mono", "Fira Code", monospace;
        color: #e0e0e0;
      }
      #derivation-panel .prov-molecular {
        border-top: 1px solid rgba(255,255,255,0.06);
        padding-top: 10px;
      }
      #derivation-panel .prov-mol-identity {
        margin-bottom: 8px;
      }
      #derivation-panel .prov-mol-name {
        font-weight: 600;
        color: #ffffff;
        font-size: 14px;
      }
      #derivation-panel .prov-mol-formula {
        font-family: "SF Mono", "Fira Code", monospace;
        font-size: 12px;
        color: #80cbc4;
        margin-left: 8px;
      }
      #derivation-panel .prov-mol-mw {
        font-size: 11px;
        color: #78909c;
        margin-left: 8px;
      }
      #derivation-panel .prov-subsection {
        margin: 10px 0;
      }
      #derivation-panel .prov-subsection-title {
        font-size: 11px;
        font-weight: 600;
        color: #78909c;
        margin-bottom: 4px;
      }
      #derivation-panel .prov-atom-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
      }
      #derivation-panel .prov-atom-chip {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 4px 8px;
        border: 1px solid;
        border-radius: 4px;
        background: rgba(255,255,255,0.03);
        min-width: 48px;
      }
      #derivation-panel .prov-atom-symbol {
        font-weight: 700;
        font-size: 16px;
      }
      #derivation-panel .prov-atom-count {
        font-size: 11px;
        color: #b0bec5;
      }
      #derivation-panel .prov-atom-detail {
        font-size: 10px;
        font-family: "SF Mono", "Fira Code", monospace;
        color: #90a4ae;
      }
      #derivation-panel .prov-atom-econfig {
        font-size: 9px;
        font-family: "SF Mono", "Fira Code", monospace;
        color: #607d8b;
        margin-top: 2px;
      }
      #derivation-panel .prov-bond-summary {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
      }
      #derivation-panel .prov-bond-chip {
        font-size: 12px;
        color: #b0bec5;
        background: rgba(255,255,255,0.05);
        padding: 2px 8px;
        border-radius: 3px;
      }
      #derivation-panel .prov-qd-table {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 2px 12px;
      }
      #derivation-panel .prov-qd-row {
        display: contents;
      }
      #derivation-panel .prov-qd-label {
        font-size: 11px;
        color: #90a4ae;
        padding: 2px 0;
      }
      #derivation-panel .prov-qd-value {
        font-family: "SF Mono", "Fira Code", monospace;
        font-size: 11px;
        color: #e0e0e0;
        text-align: right;
        padding: 2px 0;
      }
      #derivation-panel .prov-derivation-chain {
        border-top: 1px solid rgba(128,203,196,0.15);
        padding-top: 10px;
        margin-top: 10px;
      }
      #derivation-panel .prov-chain-flow {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 4px;
        margin-bottom: 10px;
        padding: 8px;
        background: rgba(128,203,196,0.06);
        border-radius: 4px;
      }
      #derivation-panel .prov-chain-node {
        font-size: 11px;
        font-weight: 600;
        color: #80cbc4;
        padding: 2px 6px;
        background: rgba(128,203,196,0.12);
        border-radius: 3px;
      }
      #derivation-panel .prov-chain-arrow {
        font-size: 14px;
        color: #546e7a;
      }
      #derivation-panel .prov-optical {
        margin: 8px 0;
        padding: 8px;
        background: rgba(255,255,255,0.02);
        border-radius: 4px;
        border-left: 2px solid #4dd0e1;
      }
      #derivation-panel .prov-optical-title {
        font-size: 11px;
        font-weight: 600;
        color: #4dd0e1;
        margin-bottom: 6px;
      }
      #derivation-panel .prov-optical-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 3px;
        font-size: 12px;
      }
      #derivation-panel .prov-optical-swatch {
        display: inline-block;
        width: 14px;
        height: 14px;
        border-radius: 2px;
        border: 1px solid rgba(255,255,255,0.15);
        flex-shrink: 0;
      }
      #derivation-panel .prov-optical-label {
        color: #90a4ae;
        flex: 1;
      }
      #derivation-panel .prov-optical-value {
        font-family: "SF Mono", "Fira Code", monospace;
        font-size: 11px;
        color: #e0e0e0;
      }
      #derivation-panel .prov-rates {
        margin-top: 8px;
      }
      #derivation-panel .prov-rates-title {
        font-size: 11px;
        font-weight: 600;
        color: #ffab40;
        margin-bottom: 6px;
      }
      #derivation-panel .prov-rate-card {
        margin-bottom: 8px;
        padding: 8px;
        background: rgba(255,171,64,0.04);
        border-radius: 4px;
        border-left: 2px solid #ffab40;
      }
      #derivation-panel .prov-rate-pathway {
        font-size: 13px;
        font-weight: 600;
        color: #ffcc80;
        margin-bottom: 4px;
      }
      #derivation-panel .prov-rate-details {
        display: grid;
        grid-template-columns: 1fr;
        gap: 2px;
      }
      #derivation-panel .prov-rate-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        font-size: 12px;
      }
      #derivation-panel .prov-rate-label {
        color: #90a4ae;
      }
      #derivation-panel .prov-rate-value {
        font-family: "SF Mono", "Fira Code", monospace;
        color: #ffab40;
        font-size: 11px;
      }
      #derivation-panel .prov-rate-live {
        padding-top: 3px;
        margin-top: 3px;
        border-top: 1px solid rgba(255,171,64,0.15);
      }
      #derivation-panel .prov-rate-live-value {
        font-weight: 700;
        font-size: 13px;
        color: #ff9100;
      }
      #derivation-panel .prov-rate-citation {
        margin-top: 4px;
      }
      #derivation-panel .prov-citation {
        font-style: italic;
        font-size: 10px;
        color: #66bb6a;
      }
      #derivation-panel .prov-note {
        font-size: 12px;
        color: #90a4ae;
        padding: 2px 0;
        font-style: italic;
      }

      /* Scrollbar styling */
      #derivation-panel::-webkit-scrollbar {
        width: 4px;
      }
      #derivation-panel::-webkit-scrollbar-track {
        background: transparent;
      }
      #derivation-panel::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.12);
        border-radius: 2px;
      }

      /* Reduced motion: disable transitions */
      @media (prefers-reduced-motion: reduce) {
        #derivation-panel,
        #derivation-panel .prov-metric-bar-fill {
          transition: none;
        }
      }

      /* High contrast: strengthen borders and text */
      @media (forced-colors: active) {
        #derivation-panel {
          border-left: 2px solid CanvasText;
        }
        #derivation-panel .prov-rate-card,
        #derivation-panel .prov-optical {
          border-left: 2px solid CanvasText;
        }
      }
    `;
    document.head.appendChild(style);
  }
}
