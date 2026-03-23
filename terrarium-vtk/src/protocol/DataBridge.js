/**
 * DataBridge — WebSocket client for oNeura terrarium backend.
 *
 * Binary frame format:
 *   [4B header: 'F','R', width, height]
 *   [width*height*2 bytes: u16 big-endian field values]
 *   [1B null separator: 0x00]
 *   [UTF-8 JSON metadata]
 *
 * JSON messages: { type: "snapshot"|"entities"|"error", ... }
 */
export class DataBridge {
  constructor(url = null) {
    this.url = url || `ws://${location.hostname}:8420/ws`;
    this.ws = null;
    this.listeners = new Map();
    this.reconnectDelay = 1000;
    this.maxReconnectDelay = 16000;
    this.connected = false;
    this._reconnectTimer = null;
  }

  connect() {
    if (this.ws && this.ws.readyState <= 1) return;
    try {
      this.ws = new WebSocket(this.url);
      this.ws.binaryType = 'arraybuffer';
      this.ws.onopen = () => {
        this.connected = true;
        this.reconnectDelay = 1000;
        this._emit('status', { connected: true });
      };
      this.ws.onclose = () => {
        this.connected = false;
        this._emit('status', { connected: false });
        this._scheduleReconnect();
      };
      this.ws.onerror = () => {
        this.connected = false;
      };
      this.ws.onmessage = (ev) => this._handleMessage(ev);
    } catch (e) {
      this._scheduleReconnect();
    }
  }

  _scheduleReconnect() {
    if (this._reconnectTimer) return;
    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay);
      this.connect();
    }, this.reconnectDelay);
  }

  _handleMessage(ev) {
    if (ev.data instanceof ArrayBuffer) {
      this._decodeBinaryFrame(ev.data);
    } else {
      try {
        const msg = JSON.parse(ev.data);
        this._emit(msg.type || 'json', msg);
      } catch (e) {
        console.warn('DataBridge: bad JSON', e);
      }
    }
  }

  _decodeBinaryFrame(buffer) {
    const bytes = new Uint8Array(buffer);
    if (bytes.length < 5 || bytes[0] !== 0x46 || bytes[1] !== 0x52) return;

    const width = bytes[2];
    const height = bytes[3];
    const fieldSize = width * height * 2;
    const fieldEnd = 4 + fieldSize;

    // Decode u16 big-endian field values
    const field = new Float32Array(width * height);
    for (let i = 0; i < width * height; i++) {
      const offset = 4 + i * 2;
      const raw = (bytes[offset] << 8) | bytes[offset + 1];
      field[i] = raw / 65535.0;
    }

    // Find null separator
    let nullIdx = fieldEnd;
    while (nullIdx < bytes.length && bytes[nullIdx] !== 0) nullIdx++;

    // Decode JSON metadata
    let meta = {};
    if (nullIdx + 1 < bytes.length) {
      const metaBytes = bytes.slice(nullIdx + 1);
      try {
        meta = JSON.parse(new TextDecoder().decode(metaBytes));
      } catch (e) {
        console.warn('DataBridge: bad frame metadata', e);
      }
    }

    // Denormalize field using min/max
    const mn = meta.mn ?? 0;
    const mx = meta.mx ?? 1;
    for (let i = 0; i < field.length; i++) {
      field[i] = field[i] * (mx - mn) + mn;
    }

    this._emit('frame', { width, height, field, meta });
  }

  send(cmd) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(cmd));
    }
  }

  // Convenience commands
  play() { this.send({ cmd: 'play' }); }
  pause() { this.send({ cmd: 'pause' }); }
  step() { this.send({ cmd: 'step' }); }
  reset(seed, preset) { this.send({ cmd: 'reset', seed, preset }); }
  setSpeed(fps) { this.send({ cmd: 'speed', fps }); }
  setView(mode) { this.send({ cmd: 'view', mode }); }
  setClimate(scenario, seed) { this.send({ cmd: 'set_climate', scenario, seed }); }
  setVisualBlend(blend) { this.send({ cmd: 'set_visual_blend', blend }); }
  triggerExtremeEvent(eventType, severity) { this.send({ cmd: 'trigger_extreme_event', event_type: eventType, severity }); }
  setTimeScale(scale) { this.send({ cmd: 'set_time_scale', scale }); }
  addPlant(x, y) { this.send({ cmd: 'add_plant', x, y }); }
  addFly(x, y) { this.send({ cmd: 'add_fly', x: parseFloat(x), y: parseFloat(y) }); }
  addWater(x, y) { this.send({ cmd: 'add_water', x, y }); }

  on(event, fn) {
    if (!this.listeners.has(event)) this.listeners.set(event, []);
    this.listeners.get(event).push(fn);
    return () => {
      const arr = this.listeners.get(event);
      if (arr) {
        const idx = arr.indexOf(fn);
        if (idx >= 0) arr.splice(idx, 1);
      }
    };
  }

  _emit(event, data) {
    const fns = this.listeners.get(event);
    if (fns) fns.forEach(fn => fn(data));
  }

  async fetchInspect(params) {
    const qs = new URLSearchParams(params).toString();
    const resp = await fetch(`/api/inspect?${qs}`);
    if (!resp.ok) throw new Error(`Inspect failed: ${resp.status}`);
    return resp.json();
  }

  async fetchSnapshot() {
    const resp = await fetch('/api/snapshot');
    return resp.json();
  }

  destroy() {
    if (this._reconnectTimer) clearTimeout(this._reconnectTimer);
    if (this.ws) this.ws.close();
    this.listeners.clear();
  }
}
