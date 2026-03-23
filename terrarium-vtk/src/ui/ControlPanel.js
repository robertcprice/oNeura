/**
 * ControlPanel — Scientific instrument calibration UI.
 * Uses lil-gui for parameter controls.
 */
import GUI from 'lil-gui';

export class ControlPanel {
  constructor(bridge, viewer) {
    this.bridge = bridge;
    this.viewer = viewer;

    this.state = {
      playing: true,
      fps: 16,
      timeScale: 1.0,
      viewMode: 'terrain',
      visualBlend: 0.75,
      climate: 'none',
    };

    this.gui = new GUI({ title: 'oNeura Instrument' });
    this._buildControls();
  }

  _buildControls() {
    // Simulation controls
    const sim = this.gui.addFolder('Simulation');
    sim.add({ play: () => { this.state.playing = true; this.bridge.play(); } }, 'play').name('▶ Play');
    sim.add({ pause: () => { this.state.playing = false; this.bridge.pause(); } }, 'pause').name('⏸ Pause');
    sim.add({ step: () => this.bridge.step() }, 'step').name('⏭ Step');
    sim.add({ reset: () => this.bridge.reset(null, 'terrarium') }, 'reset').name('↺ Reset');
    sim.add(this.state, 'fps', 1, 60, 1).name('Target FPS').onChange(v => this.bridge.setSpeed(v));
    sim.add(this.state, 'timeScale', 0.1, 10, 0.1).name('Time Scale').onChange(v => this.bridge.setTimeScale(v));

    // View controls
    const view = this.gui.addFolder('View');
    view.add(this.state, 'viewMode', [
      'terrain', 'moisture', 'canopy', 'chemistry', 'odor', 'gas_exchange',
    ]).name('Field Mode').onChange(v => this.bridge.setView(v));
    view.add(this.state, 'visualBlend', 0, 1, 0.01).name('Emergent Blend').onChange(v => this.bridge.setVisualBlend(v));

    // Climate
    const climate = this.gui.addFolder('Climate');
    climate.add(this.state, 'climate', [
      'none', 'pre_industrial', 'rcp26', 'rcp45', 'rcp85',
    ]).name('Scenario').onChange(v => {
      if (v === 'none') return;
      this.bridge.setClimate(v, null);
    });

    // Extreme events
    const events = this.gui.addFolder('Extreme Events');
    const makeEvent = (name, type) => {
      events.add({ [name]: () => this.bridge.triggerExtremeEvent(type, 0.8) }, name);
    };
    makeEvent('Heatwave', 'heatwave');
    makeEvent('Cold Snap', 'cold_snap');
    makeEvent('Drought', 'drought');
    makeEvent('Flood', 'flood');
    makeEvent('Wildfire', 'wildfire');
    makeEvent('Hurricane', 'hurricane');

    // Scale navigation
    const scale = this.gui.addFolder('Scale');
    const zoomTo = (s) => () => this.viewer.camera.zoomTo(s);
    scale.add({ ecosystem: zoomTo('ecosystem') }, 'ecosystem').name('Ecosystem (1km)');
    scale.add({ organism: zoomTo('organism') }, 'organism').name('Organism (1m)');
    scale.add({ cellular: zoomTo('cellular') }, 'cellular').name('Cellular (1mm)');
    scale.add({ molecular: zoomTo('molecular') }, 'molecular').name('Molecular (1nm)');
    scale.add({ atomic: zoomTo('atomic') }, 'atomic').name('Atomic (0.1nm)');

    // Add entities
    const add = this.gui.addFolder('Add');
    const addState = { x: 4, y: 4 };
    add.add(addState, 'x', 0, 7, 1).name('Grid X');
    add.add(addState, 'y', 0, 7, 1).name('Grid Y');
    add.add({ plant: () => this.bridge.addPlant(addState.x, addState.y) }, 'plant').name('+ Plant');
    add.add({ fly: () => this.bridge.addFly(addState.x + 0.5, addState.y + 0.5) }, 'fly').name('+ Fly');
    add.add({ water: () => this.bridge.addWater(addState.x, addState.y) }, 'water').name('+ Water');

    // Close some folders by default
    climate.close();
    events.close();
    add.close();
    scale.close();
  }

  updateSnapshotDisplay(snapshot) {
    // Could add dynamic readouts here in the future
  }

  destroy() {
    this.gui.destroy();
  }
}
