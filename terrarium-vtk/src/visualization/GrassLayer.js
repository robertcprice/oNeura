/**
 * GrassLayer — Instanced grass blades across the terrain surface.
 *
 * Creates thousands of small triangular grass blades positioned on the terrain,
 * with density varying by organic content and height by soil moisture.
 * Uses merged PolyData for single-draw-call performance.
 */
import '@kitware/vtk.js/Rendering/Profiles/Geometry';

import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';
import vtkPoints from '@kitware/vtk.js/Common/Core/Points';
import vtkCellArray from '@kitware/vtk.js/Common/Core/CellArray';
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';

// Seeded pseudo-random for deterministic grass placement
function mulberry32(a) {
  return function() {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

export class GrassLayer {
  constructor(renderer) {
    this.renderer = renderer;
    this.grassPolyData = vtkPolyData.newInstance();
    this.grassMapper = vtkMapper.newInstance({ scalarVisibility: true });
    this.grassActor = vtkActor.newInstance();
    this.grassMapper.setInputData(this.grassPolyData);
    this.grassActor.setMapper(this.grassMapper);

    const prop = this.grassActor.getProperty();
    prop.setAmbient(0.5);
    prop.setDiffuse(0.6);
    prop.setSpecular(0.05);
    // Make grass two-sided so blades visible from both sides
    prop.setBackfaceCulling(false);

    this.renderer.addActor(this.grassActor);
    this._built = false;
  }

  /**
   * Build grass blades across the terrain.
   * @param {number} w - grid width
   * @param {number} h - grid height
   * @param {Float32Array|number[]} surface - terrain_surface height values
   * @param {object[]} visuals - terrain_visuals array
   * @param {number[]} moisture - moisture values
   * @param {number[]} soilStructure - soil_structure values (organic content)
   */
  update(w, h, surface, visuals, moisture, soilStructure) {
    if (!surface || surface.length < w * h) return;

    const rand = mulberry32(42); // deterministic seed
    const bladesPerCell = 12;
    const totalBlades = w * h * bladesPerCell;

    // Each blade = 1 triangle = 3 vertices
    const pts = new Float32Array(totalBlades * 3 * 3);
    const cols = new Uint8Array(totalBlades * 3 * 3);
    const tris = new Uint32Array(totalBlades * 4);

    let pi = 0, ci = 0, ti = 0, bladeIdx = 0;

    for (let gy = 0; gy < h; gy++) {
      for (let gx = 0; gx < w; gx++) {
        const cellIdx = gy * w + gx;
        const baseH = ((surface[cellIdx] || 0.3) - 0.3) * 3.0;

        // Density: more grass where there's more organic matter
        const organic = soilStructure ? (soilStructure[cellIdx] || 0.3) : 0.3;
        const moist = moisture ? (moisture[cellIdx] || 0.3) : 0.3;

        // Skip cells that are mostly water
        const density = Math.max(0.2, Math.min(1.0, organic * 1.5));

        // Get terrain color for grass base
        const vis = visuals && visuals[cellIdx];
        let baseR = 0.15, baseG = 0.45, baseB = 0.1;
        if (vis && vis.rgb) {
          // Grass is greener than terrain but influenced by soil color
          baseR = vis.rgb[0] * 0.3;
          baseG = Math.max(0.35, vis.rgb[1] * 0.8 + 0.2);
          baseB = vis.rgb[2] * 0.25;
        }

        for (let b = 0; b < bladesPerCell; b++) {
          if (rand() > density) continue;

          // Random position within cell
          const bx = gx + rand();
          const by = gy + rand();

          // Interpolate height at this position
          const localH = baseH + (rand() - 0.5) * 0.05;

          // Blade height varies with moisture (wetter = taller grass)
          const bladeH = 0.08 + moist * 0.15 + rand() * 0.08;
          const bladeW = 0.015 + rand() * 0.01;

          // Random tilt angle
          const angle = rand() * Math.PI * 2;
          const tiltX = Math.cos(angle) * bladeW;
          const tiltY = Math.sin(angle) * bladeW;

          // Wind sway
          const swayX = Math.sin(bx * 3.7 + by * 2.1) * 0.02;
          const swayY = Math.cos(bx * 2.3 + by * 4.1) * 0.02;

          const v0 = bladeIdx * 3;

          // Base left vertex
          pts[pi++] = bx - tiltX;
          pts[pi++] = by - tiltY;
          pts[pi++] = localH;

          // Base right vertex
          pts[pi++] = bx + tiltX;
          pts[pi++] = by + tiltY;
          pts[pi++] = localH;

          // Tip vertex (elevated + sway)
          pts[pi++] = bx + swayX;
          pts[pi++] = by + swayY;
          pts[pi++] = localH + bladeH;

          // Colors — base is darker, tip is brighter green
          const tipBright = 1.2 + rand() * 0.3;
          // Base vertices (darker)
          cols[ci++] = Math.min(255, Math.round(baseR * 200));
          cols[ci++] = Math.min(255, Math.round(baseG * 180));
          cols[ci++] = Math.min(255, Math.round(baseB * 200));

          cols[ci++] = Math.min(255, Math.round(baseR * 200));
          cols[ci++] = Math.min(255, Math.round(baseG * 180));
          cols[ci++] = Math.min(255, Math.round(baseB * 200));

          // Tip vertex (brighter, yellower)
          cols[ci++] = Math.min(255, Math.round(baseR * tipBright * 255));
          cols[ci++] = Math.min(255, Math.round(baseG * tipBright * 255));
          cols[ci++] = Math.min(255, Math.round(baseB * tipBright * 180));

          // Triangle
          tris[ti++] = 3;
          tris[ti++] = v0;
          tris[ti++] = v0 + 1;
          tris[ti++] = v0 + 2;

          bladeIdx++;
        }
      }
    }

    // Trim arrays to actual size
    const vtkPts = vtkPoints.newInstance();
    vtkPts.setData(pts.slice(0, pi), 3);
    this.grassPolyData.setPoints(vtkPts);

    const vtkPolys = vtkCellArray.newInstance();
    vtkPolys.setData(tris.slice(0, ti));
    this.grassPolyData.setPolys(vtkPolys);

    const scalars = vtkDataArray.newInstance({
      numberOfComponents: 3,
      values: cols.slice(0, ci),
      dataType: 'Uint8Array',
      name: 'GrassColor',
    });
    this.grassPolyData.getPointData().setScalars(scalars);
    this.grassMapper.setColorModeToDirectScalars();

    this.grassPolyData.modified();
    this.grassMapper.modified();
    this._built = true;
  }

  destroy() {
    this.renderer.removeActor(this.grassActor);
    this.grassActor.delete();
    this.grassMapper.delete();
  }
}
