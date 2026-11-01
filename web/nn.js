export function parse(buf) {
  const n = Number(new DataView(buf).getBigUint64(0, true));
  const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n)));
  const w = {};
  for (const [k, v] of Object.entries(head)) {
    if (k === "__metadata__") continue;
    const [a, b] = v.data_offsets;
    w[k] = new Float32Array(buf.slice(8 + n + a, 8 + n + b));
  }
  return w;
}

function linear(x, wt, b) {
  const out = b.slice(), n = x.length;
  for (let o = 0; o < out.length; o++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += wt[o * n + i] * x[i];
    out[o] += s;
  }
  return out;
}

const relu = x => x.map(v => (v > 0 ? v : 0));

export function softmax(x) {
  const m = Math.max(...x), e = x.map(v => Math.exp(v - m)), s = e.reduce((a, b) => a + b);
  return e.map(v => v / s);
}

export function Predictor(w) {
  const L = i => [w[`net.layers.${i}.weight`], w[`net.layers.${i}.bias`]];
  return x => linear(relu(linear(relu(linear(x, ...L(0))), ...L(3))), ...L(5));
}

const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];

// box -> 112px, imagenet norm, nchw
export function crop(px, w, h, [cx, cy, side], size = 112) {
  const out = new Float32Array(3 * size * size), k = side / size, plane = size * size;
  for (let v = 0; v < size; v++) {
    const sy = Math.min(Math.max(cy - side / 2 + v * k, 0), h - 1), y0 = Math.floor(sy), y1 = Math.min(y0 + 1, h - 1), fy = sy - y0;
    for (let u = 0; u < size; u++) {
      const sx = Math.min(Math.max(cx - side / 2 + u * k, 0), w - 1), x0 = Math.floor(sx), x1 = Math.min(x0 + 1, w - 1), fx = sx - x0;
      const a = (y0 * w + x0) * 4, b = (y0 * w + x1) * 4, c = (y1 * w + x0) * 4, d = (y1 * w + x1) * 4;
      for (let ch = 0; ch < 3; ch++) {
        const top = px[a + ch] * (1 - fx) + px[b + ch] * fx, bot = px[c + ch] * (1 - fx) + px[d + ch] * fx;
        out[ch * plane + v * size + u] = ((top * (1 - fy) + bot * fy) / 255 - MEAN[ch]) / STD[ch];
      }
    }
  }
  return out;
}
