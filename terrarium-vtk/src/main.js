/**
 * oNeura Terrarium — Scientific Instrument
 *
 * Architecture:
 *   Rust Backend → WebSocket → DataBridge → ShaderTerrain (GLSL raymarching)
 *                                         → ControlPanel (lil-gui)
 *                                         → ProvenanceTracker → Derivation Panel
 *
 * The ShaderTerrain does ALL visual rendering (terrain, trees, grass, sky, lighting).
 * Click-to-inspect uses raycasting math to find entities.
 */
import { DataBridge } from './protocol/DataBridge.js';
import { ControlPanel } from './ui/ControlPanel.js';
import { ShaderTerrain } from './visualization/ShaderTerrain.js';
import { ProvenanceTracker } from './visualization/ProvenanceTracker.js';

// --- Initialize ---
const container = document.getElementById('viewer');
const statusEl = document.getElementById('status');
const scaleEl = document.getElementById('scale-indicator');

const shaderTerrain = new ShaderTerrain(container);
const bridge = new DataBridge();
const provenance = new ProvenanceTracker();

// Stub viewer object for ControlPanel compatibility
const viewerStub = {
  camera: {
    zoomTo(scale) {
      const targets = { ecosystem: 12, organism: 5, cellular: 2, molecular: 0.5, atomic: 0.15 };
      shaderTerrain.targetCamDist = targets[scale] || 12;
      if (scaleEl) scaleEl.textContent = scale.toUpperCase();
    },
    on() {},
  },
  provenance,
  renderer: { getActiveCamera: () => null },
};

const controls = new ControlPanel(bridge, viewerStub);

// Entity tracking for click-to-inspect
let currentPlants = [];
let currentSeeds = [];
let gridWidth = 8, gridHeight = 8;

// --- Wire DataBridge ---

bridge.on('status', ({ connected }) => {
  statusEl.textContent = connected ? 'connected' : 'reconnecting...';
  if (connected) {
    bridge.setView('terrain');
    bridge.setVisualBlend(0.75);
  }
});

bridge.on('frame', (frameData) => {
  shaderTerrain.updateFromFrame(frameData.meta);
  gridWidth = frameData.width;
  gridHeight = frameData.height;
});

bridge.on('entities', (entitiesMsg) => {
  shaderTerrain.updatePlants(entitiesMsg);
  currentPlants = entitiesMsg.full_plants || [];
  currentSeeds = entitiesMsg.seeds || [];
});

bridge.on('snapshot', (snapshotMsg) => {
  controls.updateSnapshotDisplay(snapshotMsg);

  if (snapshotMsg.terrain_surface && snapshotMsg.terrain_visuals) {
    shaderTerrain.updateFromFrame(snapshotMsg);
  }
  if (snapshotMsg.entities) {
    shaderTerrain.updatePlants(snapshotMsg.entities);
    currentPlants = snapshotMsg.entities.full_plants || [];
    currentSeeds = snapshotMsg.entities.seeds || [];
  }
});

bridge.on('snapshot_history', () => {});

// --- Click to inspect ---

container.addEventListener('click', async (e) => {
  // Ignore clicks on the UI panel or if user was dragging (orbit)
  if (e.target.closest('.lil-gui') || e.target.closest('#derivation-panel')) return;
  if (shaderTerrain.wasDrag()) return;

  const rect = container.getBoundingClientRect();
  const nx = (e.clientX - rect.left) / rect.width;
  const ny = 1.0 - (e.clientY - rect.top) / rect.height;

  // Map screen position to approximate grid position
  // This is a rough approximation — proper raycasting would need the camera matrix
  const cam = shaderTerrain;
  const gridX = Math.floor(nx * gridWidth);
  const gridY = Math.floor(ny * gridHeight);

  // Check if click is near a plant
  let entity = null;
  for (let i = 0; i < currentPlants.length; i++) {
    const p = currentPlants[i];
    const dx = (p.x + 0.5) / gridWidth - nx;
    const dy = (p.y + 0.5) / gridHeight - ny;
    if (Math.sqrt(dx*dx + dy*dy) < 0.15) {
      entity = { kind: 'plant', index: i };
      break;
    }
  }

  if (!entity) {
    for (let i = 0; i < currentSeeds.length; i++) {
      const s = currentSeeds[i];
      const dx = s.x / gridWidth - nx;
      const dy = s.y / gridHeight - ny;
      if (Math.sqrt(dx*dx + dy*dy) < 0.1) {
        entity = { kind: 'seed', index: i };
        break;
      }
    }
  }

  if (!entity && gridX >= 0 && gridX < gridWidth && gridY >= 0 && gridY < gridHeight) {
    entity = { kind: 'soil', x: gridX, y: gridY };
  }

  if (entity) {
    try {
      const params = { ...entity, scale: 'ecosystem' };
      const result = await bridge.fetchInspect(params);
      provenance.showPanel(provenance.formatDerivation(result));
    } catch (e) {
      console.warn('Inspect failed:', e);
    }
  }
});

// --- Connect ---

bridge.connect();

// --- Keyboard shortcuts ---

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  switch (e.key) {
    case ' ':
      e.preventDefault();
      if (controls.state.playing) { bridge.pause(); controls.state.playing = false; }
      else { bridge.play(); controls.state.playing = true; }
      break;
    case 'n': bridge.step(); break;
    case 'r': bridge.reset(null, 'terrarium'); break;
    case '1': viewerStub.camera.zoomTo('ecosystem'); break;
    case '2': viewerStub.camera.zoomTo('organism'); break;
    case '3': viewerStub.camera.zoomTo('cellular'); break;
    case '4': viewerStub.camera.zoomTo('molecular'); break;
    case '5': viewerStub.camera.zoomTo('atomic'); break;
    case 'Escape': provenance.hidePanel(); break;
  }
});

// --- Responsive ---
window.addEventListener('resize', () => {});

console.log('oNeura Scientific Instrument — GLSL Raymarched Terrain');
console.log('Keys: Space=play/pause, 1-5=zoom, N=step, R=reset, Esc=close panel');
console.log('Mouse: drag=orbit, scroll=zoom, click=inspect');
