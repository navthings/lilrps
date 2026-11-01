import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.webgpu.min.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";
const model = await (await fetch("./classifier_r18.onnx")).arrayBuffer();
let clf;
try {
  clf = await ort.InferenceSession.create(model, { executionProviders: ["webgpu"] });
} catch {
  // no webgpu
  clf = await ort.InferenceSession.create(model, { executionProviders: ["wasm"] });
}
postMessage(null);
onmessage = async e => {
  const out = await clf.run({ x: new ort.Tensor("float32", e.data, [1, 3, 112, 112]) });
  postMessage(Array.from(out.probs.data));
};
