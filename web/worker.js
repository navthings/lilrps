import { parse, Classifier } from "./nn.js";

const clf = Classifier(parse(await (await fetch("../results/classifier.safetensors")).arrayBuffer()));
postMessage(null);
onmessage = e => postMessage(clf(e.data, 96));
