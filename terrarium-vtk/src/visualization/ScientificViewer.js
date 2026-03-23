/**
 * ScientificViewer — Core vtk.js rendering engine for oNeura terrarium.
 *
 * Architecture: Source → Filter → Mapper → Actor (vtk.js scientific pipeline)
 * Philosophy: Every pixel is a projection of a computation. No hardcoded colors.
 */
import '@kitware/vtk.js/Rendering/Profiles/Geometry';
import '@kitware/vtk.js/Rendering/Profiles/Volume';

import vtkFullScreenRenderWindow from '@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkCellPicker from '@kitware/vtk.js/Rendering/Core/CellPicker';
import vtkLight from '@kitware/vtk.js/Rendering/Core/Light';

import { TerrainLayer } from './TerrainLayer.js';
import { EntityLayer } from './EntityLayer.js';
import { AtmosphereLayer } from './AtmosphereLayer.js';
import { MultiScaleCamera } from './MultiScaleCamera.js';
import { ProvenanceTracker } from './ProvenanceTracker.js';

export class ScientificViewer {
  constructor(container) {
    this.container = container;

    // vtk.js render window — transparent background so GLSL shader shows through
    this.fullScreen = vtkFullScreenRenderWindow.newInstance({
      rootContainer: container,
      background: [0, 0, 0, 0],
      containerStyle: { width: '100%', height: '100%', position: 'absolute', top: '0', left: '0', 'z-index': '1' },
    });

    this.renderWindow = this.fullScreen.getRenderWindow();
    this.renderer = this.fullScreen.getRenderer();
    this.interactor = this.renderWindow.getInteractor();

    // Picker for click-to-inspect
    this.picker = vtkCellPicker.newInstance();
    this.picker.setTolerance(0.005);

    // Layers
    this.terrain = new TerrainLayer(this.renderer);
    this.entities = new EntityLayer(this.renderer);
    this.atmosphere = new AtmosphereLayer(this.renderer);

    // Multi-scale camera
    this.camera = new MultiScaleCamera(this.renderer, this.renderWindow);

    // Provenance
    this.provenance = new ProvenanceTracker();

    // Current state
    this.gridWidth = 8;
    this.gridHeight = 8;
    this.lastFrame = null;
    this.lastEntities = null;
    this.lastSnapshot = null;
    this.selectedEntity = null;

    this._setupLighting();
    this._setupInteraction();
    this._setupInitialCamera();
  }

  _setupLighting() {
    // Remove default light
    this.renderer.removeAllLights();

    // Key light — bright warm sunlight from above
    const sunLight = vtkLight.newInstance({
      color: [1.0, 0.97, 0.9],
      intensity: 1.2,
      positional: false,
    });
    sunLight.setDirection(0.2, 0.3, -0.9);
    this.renderer.addLight(sunLight);

    // Fill light — cool from the side to show depth
    const fillLight = vtkLight.newInstance({
      color: [0.7, 0.8, 1.0],
      intensity: 0.5,
      positional: false,
    });
    fillLight.setDirection(-0.6, -0.4, -0.3);
    this.renderer.addLight(fillLight);

    // Rim light — from behind to outline objects
    const rimLight = vtkLight.newInstance({
      color: [1.0, 0.9, 0.8],
      intensity: 0.3,
      positional: false,
    });
    rimLight.setDirection(0.0, -0.8, -0.2);
    this.renderer.addLight(rimLight);
  }

  _setupInitialCamera() {
    const cam = this.renderer.getActiveCamera();
    cam.setPosition(4, -4, 10);
    cam.setFocalPoint(4, 4, 0);
    cam.setViewUp(0, 0, 1);
    cam.setClippingRange(0.01, 200);
    this.renderer.resetCameraClippingRange();
  }

  _setupInteraction() {
    // Left-click → inspect / derivation
    this.interactor.onLeftButtonPress((callData) => {
      if (callData.shiftKey || callData.controlKey) return; // don't intercept orbit modifiers
      const pos = callData.position;
      this.picker.pick([pos.x, pos.y, 0], this.renderer);
      const worldPos = this.picker.getPickPosition();
      if (worldPos[0] === 0 && worldPos[1] === 0 && worldPos[2] === 0) return;

      this._onPick(worldPos, this.picker.getActors());
    });
  }

  async _onPick(worldPos, actors) {
    const scale = this.camera.currentScale;
    const entity = this.entities.findEntityAt(worldPos);

    if (entity) {
      this.selectedEntity = entity;
      const params = {
        kind: entity.kind,
        index: entity.index,
        scale,
      };
      this._emit('inspect', { worldPos, entity, scale, params });
    } else {
      // Soil / terrain click
      const gridX = Math.floor(worldPos[0]);
      const gridY = Math.floor(worldPos[1]);
      if (gridX >= 0 && gridX < this.gridWidth && gridY >= 0 && gridY < this.gridHeight) {
        this._emit('inspect', {
          worldPos,
          entity: { kind: 'soil', x: gridX, y: gridY },
          scale,
          params: { kind: 'soil', x: gridX, y: gridY, scale },
        });
      }
    }
  }

  // === Data ingestion ===

  updateFrame(frameData) {
    this.lastFrame = frameData;
    const { width, height, field, meta } = frameData;
    this.gridWidth = width;
    this.gridHeight = height;

    this.terrain.update(width, height, field, meta);
    this.atmosphere.updateFromMeta(meta);
    this.renderWindow.render();
  }

  updateEntities(entitiesMsg) {
    this.lastEntities = entitiesMsg;
    this.entities.update(entitiesMsg);
    this.renderWindow.render();
  }

  updateSnapshot(snapshotMsg) {
    this.lastSnapshot = snapshotMsg;
    this._emit('snapshot', snapshotMsg);
  }

  // === Event system ===

  _listeners = new Map();

  on(event, fn) {
    if (!this._listeners.has(event)) this._listeners.set(event, []);
    this._listeners.get(event).push(fn);
  }

  _emit(event, data) {
    const fns = this._listeners.get(event);
    if (fns) fns.forEach(fn => fn(data));
  }

  resize() {
    this.fullScreen.resize();
  }

  render() {
    this.renderWindow.render();
  }

  destroy() {
    this.terrain.destroy();
    this.entities.destroy();
    this.atmosphere.destroy();
    this.fullScreen.delete();
  }
}
