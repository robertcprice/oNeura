/**
 * AtmosphereLayer — Sky and atmosphere rendering for the terrarium.
 *
 * NO sky dome sphere. Uses renderer gradient background (setBackground2) for
 * a natural horizon-to-zenith gradient. Sun is a small emissive sphere.
 *
 * All visual parameters derived from backend simulation data.
 */
import '@kitware/vtk.js/Rendering/Profiles/Geometry';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkSphereSource from '@kitware/vtk.js/Filters/Sources/SphereSource';

const SCENE_CENTER = [4.0, 4.0, 0.0];
const SUN_DISTANCE = 30.0;
const SUN_RADIUS = 0.6;

export class AtmosphereLayer {
  constructor(renderer) {
    this.renderer = renderer;

    // vtk.js supports gradient via setBackground + setBackground2
    // setBackground2 is the top color, setBackground is the bottom color
    // Gradient is automatic when both are set

    // Sun indicator — small bright sphere
    this.sunSource = vtkSphereSource.newInstance({
      center: [0, 0, 30],
      radius: SUN_RADIUS,
      thetaResolution: 16,
      phiResolution: 16,
    });
    this.sunMapper = vtkMapper.newInstance();
    this.sunMapper.setInputConnection(this.sunSource.getOutputPort());
    this.sunActor = vtkActor.newInstance();
    this.sunActor.setMapper(this.sunMapper);
    const sunProp = this.sunActor.getProperty();
    sunProp.setAmbient(1.0);
    sunProp.setDiffuse(0.0);
    sunProp.setSpecular(0.0);
    sunProp.setColor(1.0, 0.95, 0.75);
    this.sunActor.setVisibility(false);
    this.renderer.addActor(this.sunActor);
  }

  updateFromMeta(meta) {
    if (!meta) return;

    const daylight = meta.daylight ?? 0.0;
    const sunElev = meta.sun_elevation_rad ?? 0.0;
    const sunDir = meta.sun_direction || [0.5, 0.0, 0.7];

    // Compute sky gradient colors
    const { zenith, horizon } = this._computeSkyGradient(daylight);

    // Fog: if high humidity, blend toward gray
    const meanHum = this._meanHumidity(meta.atmosphere);
    if (meanHum > 0.8) {
      const fog = (meanHum - 0.8) * 5.0; // 0-1
      const gray = [0.7, 0.72, 0.75];
      for (let i = 0; i < 3; i++) {
        zenith[i] += (gray[i] - zenith[i]) * fog;
        horizon[i] += (gray[i] - horizon[i]) * fog;
      }
    }

    // Apply gradient: background = bottom color, background2 = top color
    this.renderer.setBackground(horizon[0], horizon[1], horizon[2]);
    this.renderer.setBackground2(zenith[0], zenith[1], zenith[2]);

    // Sun position
    if (sunElev > 0.02) {
      const sx = SCENE_CENTER[0] + sunDir[0] * SUN_DISTANCE;
      const sy = SCENE_CENTER[1] + sunDir[1] * SUN_DISTANCE;
      const sz = SCENE_CENTER[2] + sunDir[2] * SUN_DISTANCE;
      this.sunSource.setCenter(sx, sy, Math.max(2, sz));
      this.sunActor.setVisibility(true);

      // Sun color: white at noon, orange at low elevation
      const t = Math.min(1, sunElev / 0.5); // 0 at horizon, 1 at 30°
      this.sunActor.getProperty().setColor(
        1.0,
        0.8 + t * 0.15,
        0.5 + t * 0.3,
      );
    } else {
      this.sunActor.setVisibility(false);
    }
  }

  _computeSkyGradient(daylight) {
    if (daylight < 0.05) {
      // Night
      return {
        zenith: [0.01, 0.01, 0.05],
        horizon: [0.03, 0.03, 0.08],
      };
    }
    if (daylight < 0.2) {
      // Dawn/dusk — orange horizon, dark blue zenith
      const t = (daylight - 0.05) / 0.15;
      return {
        zenith: [0.05 + t * 0.1, 0.05 + t * 0.12, 0.15 + t * 0.2],
        horizon: [0.3 + t * 0.5, 0.15 + t * 0.2, 0.05 + t * 0.05],
      };
    }
    // Daytime — Rayleigh scattering: blue zenith, lighter horizon
    const intensity = Math.min(1, (daylight - 0.2) / 0.6);
    return {
      zenith: [
        0.15 + intensity * 0.15,   // slight blue-purple
        0.25 + intensity * 0.3,    // medium
        0.55 + intensity * 0.35,   // strong blue
      ],
      horizon: [
        0.55 + intensity * 0.2,    // lighter
        0.65 + intensity * 0.15,   // lighter
        0.75 + intensity * 0.15,   // lighter blue
      ],
    };
  }

  _meanHumidity(atm) {
    if (!atm || !atm.humidity || !atm.humidity.length) return 0;
    let s = 0;
    for (let i = 0; i < atm.humidity.length; i++) s += atm.humidity[i];
    return s / atm.humidity.length;
  }

  destroy() {
    this.renderer.removeActor(this.sunActor);
    this.sunActor.delete();
    this.sunMapper.delete();
    this.sunSource.delete();
  }
}
