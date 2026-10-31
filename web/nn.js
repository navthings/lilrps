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

// conv+relu+pool
function block(x, s, { cin, cout, wt, b }) {
  const h = s >> 1, n = 9 * cin, out = new Float32Array(h * h * cout), patch = new Float32Array(n);
  for (let y = 0; y < s; y++) {
    for (let x0 = 0; x0 < s; x0++) {
      // im2col
      patch.fill(0);
      for (let ky = 0; ky < 3; ky++) {
        const iy = y + ky - 1;
        if (iy < 0 || iy >= s) continue;
        for (let kx = 0; kx < 3; kx++) {
          const ix = x0 + kx - 1;
          if (ix < 0 || ix >= s) continue;
          patch.set(x.subarray((iy * s + ix) * cin, (iy * s + ix + 1) * cin), (ky * 3 + kx) * cin);
        }
      }
      const op = ((y >> 1) * h + (x0 >> 1)) * cout;
      for (let o = 0; o < cout; o++) {
        const wo = o * n;
        let s0 = b[o], s1 = 0, s2 = 0, s3 = 0, i = 0;
        for (; i + 3 < n; i += 4) {
          s0 += wt[wo + i] * patch[i];
          s1 += wt[wo + i + 1] * patch[i + 1];
          s2 += wt[wo + i + 2] * patch[i + 2];
          s3 += wt[wo + i + 3] * patch[i + 3];
        }
        for (; i < n; i++) s0 += wt[wo + i] * patch[i];
        const v = s0 + s1 + s2 + s3;
        if (v > out[op + o]) out[op + o] = v;
      }
    }
  }
  return out;
}

export function Classifier(w) {
  const chans = [3, 24, 48, 96, 160];
  // fold batchnorm
  const layers = chans.slice(1).map((cout, l) => {
    const cin = chans[l], wt = w[`convs.${l}.weight`].slice(), b = w[`convs.${l}.bias`].slice();
    const g = w[`norms.${l}.weight`], beta = w[`norms.${l}.bias`];
    const m = w[`norms.${l}.running_mean`], v = w[`norms.${l}.running_var`];
    for (let o = 0; o < cout; o++) {
      const k = g[o] / Math.sqrt(v[o] + 1e-5);
      for (let i = 0; i < 9 * cin; i++) wt[o * 9 * cin + i] *= k;
      b[o] = (b[o] - m[o]) * k + beta[o];
    }
    return { cin, cout, wt, b };
  });
  return (img, size) => {
    let x = img, s = size;
    for (const L of layers) x = block(x, s, L), s >>= 1;
    const c = 160, f = new Float32Array(c);
    for (let p = 0; p < s * s; p++) for (let i = 0; i < c; i++) f[i] += x[p * c + i] / (s * s);
    return softmax(linear(f, w["head.weight"], w["head.bias"]));
  };
}
