/**
 * TerrainLayer — Continuous terrain rendering via vtk.js.
 *
 * Converts the discrete grid data from the backend into a smooth, continuous
 * terrain surface using vtk.js geometry. Heights from terrain_surface,
 * colors from emergent terrain_visuals (Beer-Lambert, Kubelka-Munk).
 *
 * No grid lines. No tiles. Mathematically interpolated surface.
 */
import '@kitware/vtk.js/Rendering/Profiles/Geometry';

import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';
import vtkPoints from '@kitware/vtk.js/Common/Core/Points';
import vtkCellArray from '@kitware/vtk.js/Common/Core/CellArray';
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';

export class TerrainLayer {
  constructor(renderer) {
    this.renderer = renderer;

    // Surface mesh
    this.surfacePolyData = vtkPolyData.newInstance();
    this.surfaceMapper = vtkMapper.newInstance({ scalarVisibility: true });
    this.surfaceActor = vtkActor.newInstance();
    this.surfaceMapper.setInputData(this.surfacePolyData);
    this.surfaceActor.setMapper(this.surfaceMapper);
    this.renderer.addActor(this.surfaceActor);

    // Water overlay mesh
    this.waterPolyData = vtkPolyData.newInstance();
    this.waterMapper = vtkMapper.newInstance({ scalarVisibility: true });
    this.waterActor = vtkActor.newInstance();
    this.waterMapper.setInputData(this.waterPolyData);
    this.waterActor.setMapper(this.waterMapper);
    const waterProp = this.waterActor.getProperty();
    waterProp.setOpacity(0.55);
    waterProp.setAmbient(0.3);
    waterProp.setDiffuse(0.5);
    waterProp.setSpecular(0.8);
    waterProp.setSpecularPower(60);
    this.renderer.addActor(this.waterActor);

    // Terrain side walls
    this.sidesPolyData = vtkPolyData.newInstance();
    this.sidesMapper = vtkMapper.newInstance({ scalarVisibility: true });
    this.sidesActor = vtkActor.newInstance();
    this.sidesMapper.setInputData(this.sidesPolyData);
    this.sidesActor.setMapper(this.sidesMapper);
    const sidesProp = this.sidesActor.getProperty();
    sidesProp.setAmbient(0.4);
    sidesProp.setDiffuse(0.6);
    this.renderer.addActor(this.sidesActor);

    // Subdivision factor: more = smoother continuous terrain
    this.subdivisions = 6;

    this._lastWidth = 0;
    this._lastHeight = 0;
  }

  /**
   * Update terrain from frame data.
   * @param {number} w - grid width
   * @param {number} h - grid height
   * @param {Float32Array} field - normalized field values (w*h)
   * @param {object} meta - frame metadata with terrain_surface, terrain_visuals, moisture, water_mask
   */
  update(w, h, field, meta) {
    const surface = meta.terrain_surface;
    const visuals = meta.terrain_visuals;
    const moisture = meta.moisture;
    const waterMask = meta.water_mask;

    if (!surface || surface.length < w * h) {
      // No terrain data yet — render flat plane with field-based coloring
      this._buildFlatTerrain(w, h, field);
      return;
    }

    this._buildSmoothTerrain(w, h, surface, visuals, moisture);

    if (waterMask) {
      this._buildWaterSurface(w, h, surface, waterMask);
    }
  }

  /**
   * Build smooth interpolated terrain mesh.
   * Uses bicubic-like interpolation (Catmull-Rom) for continuous surface.
   */
  _buildSmoothTerrain(w, h, surface, visuals, moisture) {
    const sub = this.subdivisions;
    const sw = (w - 1) * sub + 1; // subdivided width
    const sh = (h - 1) * sub + 1; // subdivided height
    const numPts = sw * sh;

    const points = new Float32Array(numPts * 3);
    const colors = new Uint8Array(numPts * 3);

    // Generate interpolated points
    for (let sy = 0; sy < sh; sy++) {
      for (let sx = 0; sx < sw; sx++) {
        const idx = sy * sw + sx;
        const fx = sx / sub; // fractional grid coordinate
        const fy = sy / sub;

        // Height: Catmull-Rom interpolation, amplified for visual relief
        // Surface values range ~0.3-0.6, so center and amplify
        const rawH = this._catmullRomSample(surface, w, h, fx, fy);
        const z = (rawH - 0.3) * 3.0; // amplify: 0.3→0, 0.6→0.9

        points[idx * 3] = fx;
        points[idx * 3 + 1] = fy;
        points[idx * 3 + 2] = z;

        // Color: bilinear interpolation of terrain_visuals
        if (visuals && visuals.length >= w * h) {
          const color = this._bilinearColorSample(visuals, w, h, fx, fy);
          // Boost colors so they're vivid even after lighting
          colors[idx * 3] = Math.min(255, Math.round(color[0] * 280));
          colors[idx * 3 + 1] = Math.min(255, Math.round(color[1] * 280));
          colors[idx * 3 + 2] = Math.min(255, Math.round(color[2] * 280));
        } else {
          // Warm earth-tone fallback
          const val = this._bilinearSample(surface, w, h, fx, fy);
          colors[idx * 3] = Math.round(160 + val * 80);
          colors[idx * 3 + 1] = Math.round(120 + val * 50);
          colors[idx * 3 + 2] = Math.round(70 + val * 40);
        }

        // Subtle moisture variation (NOT darkening — slight color shift)
        if (moisture && moisture.length >= w * h) {
          const m = this._bilinearSample(moisture, w, h, fx, fy);
          // Wet areas get slightly darker and greener
          colors[idx * 3] = Math.round(colors[idx * 3] * (1.0 - m * 0.12));
          colors[idx * 3 + 1] = Math.round(colors[idx * 3 + 1] * (1.0 + m * 0.04));
          colors[idx * 3 + 2] = Math.round(colors[idx * 3 + 2] * (1.0 - m * 0.08));
        }
      }
    }

    // Build triangle strip connectivity
    const numTris = (sw - 1) * (sh - 1) * 2;
    const polys = new Uint32Array(numTris * 4); // 4 values per triangle: count + 3 indices
    let offset = 0;
    for (let y = 0; y < sh - 1; y++) {
      for (let x = 0; x < sw - 1; x++) {
        const i00 = y * sw + x;
        const i10 = y * sw + x + 1;
        const i01 = (y + 1) * sw + x;
        const i11 = (y + 1) * sw + x + 1;

        polys[offset++] = 3;
        polys[offset++] = i00;
        polys[offset++] = i10;
        polys[offset++] = i11;

        polys[offset++] = 3;
        polys[offset++] = i00;
        polys[offset++] = i11;
        polys[offset++] = i01;
      }
    }

    const vtkPts = vtkPoints.newInstance();
    vtkPts.setData(points, 3);
    this.surfacePolyData.setPoints(vtkPts);

    const vtkPolys = vtkCellArray.newInstance();
    vtkPolys.setData(polys);
    this.surfacePolyData.setPolys(vtkPolys);

    const scalars = vtkDataArray.newInstance({
      numberOfComponents: 3,
      values: colors,
      dataType: 'Uint8Array',
      name: 'TerrainColor',
    });
    this.surfacePolyData.getPointData().setScalars(scalars);
    this.surfaceMapper.setColorModeToDirectScalars();

    // Compute normals for proper lighting
    this._computeNormals(this.surfacePolyData, sw, sh, points);

    // Material properties — high ambient so terrain stays bright
    const prop = this.surfaceActor.getProperty();
    prop.setAmbient(0.6);
    prop.setDiffuse(0.5);
    prop.setSpecular(0.02);
    prop.setSpecularPower(5);

    this.surfacePolyData.modified();
    this.surfaceMapper.modified();

    // Build side walls
    this._buildSideWalls(w, h, surface, visuals);
  }

  /**
   * Build side walls around the terrain perimeter so it doesn't float.
   */
  _buildSideWalls(w, h, surface, visuals) {
    const bottomZ = -0.5;
    // Collect perimeter points
    const perimPts = [];

    // Bottom edge (y=0)
    for (let x = 0; x < w; x++) perimPts.push({ x, y: 0, idx: x });
    // Right edge (x=w-1)
    for (let y = 0; y < h; y++) perimPts.push({ x: w-1, y, idx: y * w + w - 1 });
    // Top edge (y=h-1, reversed)
    for (let x = w-1; x >= 0; x--) perimPts.push({ x, y: h-1, idx: (h-1) * w + x });
    // Left edge (x=0, reversed)
    for (let y = h-1; y >= 0; y--) perimPts.push({ x: 0, y, idx: y * w });

    const n = perimPts.length;
    const pts = new Float32Array(n * 2 * 3); // top + bottom per point
    const cols = new Uint8Array(n * 2 * 3);
    const tris = new Uint32Array((n - 1) * 2 * 4);

    for (let i = 0; i < n; i++) {
      const p = perimPts[i];
      const rawH = surface[p.idx] || 0.3;
      const topZ = (rawH - 0.3) * 3.0;

      // Top vertex
      pts[i*6]     = p.x;
      pts[i*6 + 1] = p.y;
      pts[i*6 + 2] = topZ;
      // Bottom vertex
      pts[i*6 + 3] = p.x;
      pts[i*6 + 4] = p.y;
      pts[i*6 + 5] = bottomZ;

      // Side color: geological layers — darker warm brown
      const v = visuals && visuals[p.idx];
      let r = 0.45, g = 0.32, b = 0.2;
      if (v && v.rgb) { r = v.rgb[0] * 0.65; g = v.rgb[1] * 0.55; b = v.rgb[2] * 0.45; }
      // Top vertex: soil color
      cols[i*6]     = Math.round(r * 255);
      cols[i*6 + 1] = Math.round(g * 255);
      cols[i*6 + 2] = Math.round(b * 255);
      // Bottom vertex: darker bedrock
      cols[i*6 + 3] = Math.round(r * 0.4 * 255);
      cols[i*6 + 4] = Math.round(g * 0.35 * 255);
      cols[i*6 + 5] = Math.round(b * 0.3 * 255);
    }

    let ti = 0;
    for (let i = 0; i < n - 1; i++) {
      const topA = i * 2, botA = i * 2 + 1;
      const topB = (i+1) * 2, botB = (i+1) * 2 + 1;
      tris[ti++] = 3; tris[ti++] = topA; tris[ti++] = botA; tris[ti++] = botB;
      tris[ti++] = 3; tris[ti++] = topA; tris[ti++] = botB; tris[ti++] = topB;
    }

    const vtkPts = vtkPoints.newInstance();
    vtkPts.setData(pts, 3);
    this.sidesPolyData.setPoints(vtkPts);

    const vtkPolys = vtkCellArray.newInstance();
    vtkPolys.setData(tris.slice(0, ti));
    this.sidesPolyData.setPolys(vtkPolys);

    const scalars = vtkDataArray.newInstance({
      numberOfComponents: 3, values: cols, dataType: 'Uint8Array', name: 'SideColor',
    });
    this.sidesPolyData.getPointData().setScalars(scalars);
    this.sidesMapper.setColorModeToDirectScalars();
    this.sidesPolyData.modified();
    this.sidesMapper.modified();
  }

  _buildFlatTerrain(gridW, gridH, field) {
    // Use subdivisions for smooth interpolation even on flat terrain
    const sub = this.subdivisions;
    const sw = (gridW - 1) * sub + 1;
    const sh = (gridH - 1) * sub + 1;
    const numPts = sw * sh;
    const points = new Float32Array(numPts * 3);
    const colors = new Uint8Array(numPts * 3);

    for (let sy = 0; sy < sh; sy++) {
      for (let sx = 0; sx < sw; sx++) {
        const idx = sy * sw + sx;
        const fx = sx / sub;
        const fy = sy / sub;

        // Gentle height from field values
        const val = this._bilinearSample(field, gridW, gridH, fx, fy);
        points[idx * 3] = fx;
        points[idx * 3 + 1] = fy;
        points[idx * 3 + 2] = val * 0.5;

        // Earth-tone colors
        colors[idx * 3] = Math.round(120 + val * 100);
        colors[idx * 3 + 1] = Math.round(90 + val * 70);
        colors[idx * 3 + 2] = Math.round(50 + val * 40);
      }
    }

    const w = sw;
    const h = sh;

    const numTris = (w - 1) * (h - 1) * 2;
    const polys = new Uint32Array(numTris * 4);
    let offset = 0;
    for (let y = 0; y < h - 1; y++) {
      for (let x = 0; x < w - 1; x++) {
        const i00 = y * w + x;
        const i10 = y * w + x + 1;
        const i01 = (y + 1) * w + x;
        const i11 = (y + 1) * w + x + 1;
        polys[offset++] = 3; polys[offset++] = i00; polys[offset++] = i10; polys[offset++] = i11;
        polys[offset++] = 3; polys[offset++] = i00; polys[offset++] = i11; polys[offset++] = i01;
      }
    }

    const vtkPts = vtkPoints.newInstance();
    vtkPts.setData(points, 3);
    this.surfacePolyData.setPoints(vtkPts);

    const vtkPolys = vtkCellArray.newInstance();
    vtkPolys.setData(polys);
    this.surfacePolyData.setPolys(vtkPolys);

    const scalars = vtkDataArray.newInstance({
      numberOfComponents: 3, values: colors, dataType: 'Uint8Array', name: 'TerrainColor',
    });
    this.surfacePolyData.getPointData().setScalars(scalars);
    this.surfaceMapper.setColorModeToDirectScalars();

    this.surfacePolyData.modified();
    this.surfaceMapper.modified();
  }

  _buildWaterSurface(w, h, surface, waterMask) {
    // water_mask serves BOTH ecology (shoreline detection, full signal) AND rendering.
    // Most cells have moderate values from subsurface moisture — NOT standing water.
    // Only render where deposit_2d from actual Water entities creates high amplitude.
    // Water entities deposit amplitude = clamp(volume/140, 0.06, 1.0) at radius 2-3.
    // Substrate adds lower open_water values everywhere.
    // Threshold 0.4 captures water body cores + immediate shore.
    const threshold = 0.4;
    const waterCells = [];
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (waterMask[y * w + x] > threshold) {
          waterCells.push({ x, y, depth: waterMask[y * w + x] });
        }
      }
    }

    if (waterCells.length === 0) {
      this.waterActor.setVisibility(false);
      return;
    }
    this.waterActor.setVisibility(true);

    const numPts = waterCells.length * 4;
    const points = new Float32Array(numPts * 3);
    const colors = new Uint8Array(numPts * 3);
    const polys = new Uint32Array(waterCells.length * 2 * 4);

    let pi = 0, ci = 0, ti = 0;
    for (const cell of waterCells) {
      const rawH = surface[cell.y * w + cell.x] || 0.3;
      const baseZ = (rawH - 0.3) * 3.0 + 0.02; // match terrain height formula
      const baseIdx = pi / 3;

      // 4 corners of water quad
      for (const [dx, dy] of [[0,0],[1,0],[1,1],[0,1]]) {
        points[pi++] = cell.x + dx;
        points[pi++] = cell.y + dy;
        points[pi++] = baseZ;

        // Pope & Fry 1997 water — bright clear blue-green
        const depth01 = Math.min(1.0, (cell.depth - threshold) / (1.0 - threshold));
        colors[ci++] = Math.round(60 + 40 * (1 - depth01 * 0.3));   // red: low
        colors[ci++] = Math.round(150 + 60 * (1 - depth01 * 0.1));  // green: medium-high
        colors[ci++] = Math.round(210 + 40 * (1 - depth01 * 0.05)); // blue: high
      }

      polys[ti++] = 3; polys[ti++] = baseIdx; polys[ti++] = baseIdx+1; polys[ti++] = baseIdx+2;
      polys[ti++] = 3; polys[ti++] = baseIdx; polys[ti++] = baseIdx+2; polys[ti++] = baseIdx+3;
    }

    const vtkPts = vtkPoints.newInstance();
    vtkPts.setData(points.slice(0, pi), 3);
    this.waterPolyData.setPoints(vtkPts);

    const vtkPolys = vtkCellArray.newInstance();
    vtkPolys.setData(polys.slice(0, ti));
    this.waterPolyData.setPolys(vtkPolys);

    const scalars = vtkDataArray.newInstance({
      numberOfComponents: 3, values: colors.slice(0, ci), dataType: 'Uint8Array', name: 'WaterColor',
    });
    this.waterPolyData.getPointData().setScalars(scalars);
    this.waterMapper.setColorModeToDirectScalars();

    this.waterPolyData.modified();
    this.waterMapper.modified();
  }

  /**
   * Compute per-vertex normals from the grid topology.
   */
  _computeNormals(polyData, sw, sh, points) {
    const normals = new Float32Array(sw * sh * 3);

    for (let y = 0; y < sh; y++) {
      for (let x = 0; x < sw; x++) {
        const idx = y * sw + x;
        const p = [points[idx*3], points[idx*3+1], points[idx*3+2]];

        // Get neighbors
        const left = x > 0 ? (y * sw + (x-1)) : idx;
        const right = x < sw-1 ? (y * sw + (x+1)) : idx;
        const down = y > 0 ? ((y-1) * sw + x) : idx;
        const up = y < sh-1 ? ((y+1) * sw + x) : idx;

        // Finite differences for tangent vectors
        const tx = [
          points[right*3] - points[left*3],
          points[right*3+1] - points[left*3+1],
          points[right*3+2] - points[left*3+2],
        ];
        const ty = [
          points[up*3] - points[down*3],
          points[up*3+1] - points[down*3+1],
          points[up*3+2] - points[down*3+2],
        ];

        // Cross product = normal
        const nx = ty[1]*tx[2] - ty[2]*tx[1];
        const ny = ty[2]*tx[0] - ty[0]*tx[2];
        const nz = ty[0]*tx[1] - ty[1]*tx[0];
        const len = Math.sqrt(nx*nx + ny*ny + nz*nz) || 1;

        normals[idx*3] = nx / len;
        normals[idx*3+1] = ny / len;
        normals[idx*3+2] = nz / len;
      }
    }

    const normalArray = vtkDataArray.newInstance({
      numberOfComponents: 3,
      values: normals,
      name: 'Normals',
    });
    polyData.getPointData().setNormals(normalArray);
  }

  // === Interpolation ===

  _bilinearSample(data, w, h, fx, fy) {
    const x0 = Math.floor(fx), y0 = Math.floor(fy);
    const x1 = Math.min(x0 + 1, w - 1), y1 = Math.min(y0 + 1, h - 1);
    const tx = fx - x0, ty = fy - y0;

    const v00 = data[y0 * w + x0] || 0;
    const v10 = data[y0 * w + x1] || 0;
    const v01 = data[y1 * w + x0] || 0;
    const v11 = data[y1 * w + x1] || 0;

    return (v00 * (1 - tx) * (1 - ty) +
            v10 * tx * (1 - ty) +
            v01 * (1 - tx) * ty +
            v11 * tx * ty);
  }

  _bilinearColorSample(visuals, w, h, fx, fy) {
    const x0 = Math.floor(fx), y0 = Math.floor(fy);
    const x1 = Math.min(x0 + 1, w - 1), y1 = Math.min(y0 + 1, h - 1);
    const tx = fx - x0, ty = fy - y0;

    const get = (x, y) => {
      const v = visuals[y * w + x];
      if (!v) return [0.3, 0.25, 0.15];
      // Backend sends {rgb: [r, g, b]} with values 0-1
      if (v.rgb && Array.isArray(v.rgb)) return v.rgb;
      return [v.r ?? 0.3, v.g ?? 0.25, v.b ?? 0.15];
    };

    const c00 = get(x0, y0), c10 = get(x1, y0);
    const c01 = get(x0, y1), c11 = get(x1, y1);

    return [
      c00[0]*(1-tx)*(1-ty) + c10[0]*tx*(1-ty) + c01[0]*(1-tx)*ty + c11[0]*tx*ty,
      c00[1]*(1-tx)*(1-ty) + c10[1]*tx*(1-ty) + c01[1]*(1-tx)*ty + c11[1]*tx*ty,
      c00[2]*(1-tx)*(1-ty) + c10[2]*tx*(1-ty) + c01[2]*(1-tx)*ty + c11[2]*tx*ty,
    ];
  }

  /**
   * Catmull-Rom spline interpolation for smooth terrain.
   * Uses 4 neighboring points for each axis.
   */
  _catmullRomSample(data, w, h, fx, fy) {
    const xi = Math.floor(fx), yi = Math.floor(fy);
    const tx = fx - xi, ty = fy - yi;

    // Sample 4x4 neighborhood
    const samples = [];
    for (let dy = -1; dy <= 2; dy++) {
      const row = [];
      for (let dx = -1; dx <= 2; dx++) {
        const sx = Math.max(0, Math.min(w - 1, xi + dx));
        const sy = Math.max(0, Math.min(h - 1, yi + dy));
        row.push(data[sy * w + sx] || 0);
      }
      samples.push(row);
    }

    // Catmull-Rom along x for each row, then along y
    const col = [];
    for (let r = 0; r < 4; r++) {
      col.push(this._catmullRom1D(samples[r][0], samples[r][1], samples[r][2], samples[r][3], tx));
    }
    return this._catmullRom1D(col[0], col[1], col[2], col[3], ty);
  }

  _catmullRom1D(p0, p1, p2, p3, t) {
    const t2 = t * t, t3 = t2 * t;
    return 0.5 * (
      (2 * p1) +
      (-p0 + p2) * t +
      (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
      (-p0 + 3*p1 - 3*p2 + p3) * t3
    );
  }

  destroy() {
    this.renderer.removeActor(this.surfaceActor);
    this.renderer.removeActor(this.waterActor);
    this.renderer.removeActor(this.sidesActor);
    this.surfaceActor.delete();
    this.waterActor.delete();
    this.sidesActor.delete();
    this.surfaceMapper.delete();
    this.waterMapper.delete();
    this.sidesMapper.delete();
  }
}
