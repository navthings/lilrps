import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.webgpu.min.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";
const make = (model, ep) => ort.InferenceSession.create(model, { executionProviders: [ep] });

try {
  const r = await fetch("./classifier_r18.onnx");
  if (!r.ok) throw new Error(`couldnt load classifier_r18.onnx (${r.status})`);
  const model = new Uint8Array(await r.arrayBuffer());
  let clf, ep = "webgpu";
  try {
    clf = await make(model, ep);
  } catch {
    // no webgpu
    clf = await make(model, (ep = "wasm"));
  }
  const run = async x => Array.from((await clf.run({ x: new ort.Tensor("float32", x, [1, 3, 112, 112]) })).probs.data);
  onmessage = async e => {
    try {
      postMessage(await run(e.data));
    } catch (err) {
      console.error(`classifier (${ep})`, err);
      // gpu broke mid game
      if (ep === "webgpu") clf = await make(model, (ep = "wasm")).catch(() => clf);
      postMessage({ error: String(err) });
    }
  };
  postMessage(null);
} catch (e) {
  postMessage({ error: `hand model: ${e.message}` });
}
