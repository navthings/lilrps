# /// script
# dependencies = ["onnx", "numpy", "safetensors"]
# ///
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from safetensors.numpy import load_file

HERE = Path(__file__).parent
SRC = HERE / "results" / "classifier_r18.safetensors"
OUT = HERE / "web" / "classifier_r18.onnx"
EPS = 1e-5


def main():
    w = load_file(str(SRC))
    nodes, inits = [], []
    n = [0]

    def name(s):
        n[0] += 1
        return f"{s}_{n[0]}"

    def weight(arr, half=True):
        # fp16 on disk, fp32 to run
        k = name("w")
        if not half:
            inits.append(numpy_helper.from_array(arr.astype(np.float32), k))
            return k
        inits.append(numpy_helper.from_array(arr.astype(np.float16), k + "_h"))
        nodes.append(helper.make_node("Cast", [k + "_h"], [k], to=TensorProto.FLOAT))
        return k

    def conv_bn(x, conv, bn, stride, pad, relu=True):
        k = w[f"{conv}.weight"].transpose(0, 3, 1, 2)
        s = w[f"{bn}.weight"] / np.sqrt(w[f"{bn}.running_var"] + EPS)
        b = w[f"{bn}.bias"] - w[f"{bn}.running_mean"] * s
        y = name("conv")
        nodes.append(helper.make_node("Conv", [x, weight(k * s[:, None, None, None]), weight(b, False)], [y],
                                      kernel_shape=list(k.shape[2:]), strides=[stride] * 2, pads=[pad] * 4))
        if relu:
            nodes.append(helper.make_node("Relu", [y], [y + "_r"]))
            return y + "_r"
        return y

    x = conv_bn("x", "conv1", "bn1", 2, 3)
    nodes.append(helper.make_node("MaxPool", [x], ["pool"], kernel_shape=[3, 3], strides=[2, 2], pads=[1] * 4))
    x = "pool"
    for i in range(1, 5):
        for j in range(2):
            p = f"layer{i}.layers.{j}"
            stride = 2 if i > 1 and j == 0 else 1
            y = conv_bn(x, f"{p}.conv1", f"{p}.bn1", stride, 1)
            y = conv_bn(y, f"{p}.conv2", f"{p}.bn2", 1, 1, relu=False)
            skip = conv_bn(x, f"{p}.downsample.layers.0", f"{p}.downsample.layers.1", stride, 0, relu=False) \
                if f"{p}.downsample.layers.0.weight" in w else x
            a = name("add")
            nodes.append(helper.make_node("Add", [y, skip], [a]))
            nodes.append(helper.make_node("Relu", [a], [a + "_r"]))
            x = a + "_r"
    nodes.append(helper.make_node("GlobalAveragePool", [x], ["gap"]))
    nodes.append(helper.make_node("Flatten", ["gap"], ["feat"]))
    nodes.append(helper.make_node("Gemm", ["feat", weight(w["fc.weight"]), weight(w["fc.bias"], False)], ["logits"], transB=1))
    nodes.append(helper.make_node("Softmax", ["logits"], ["probs"], axis=1))

    graph = helper.make_graph(nodes, "lilrps", [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", 3, 112, 112])],
                              [helper.make_tensor_value_info("probs", TensorProto.FLOAT, ["n", 4])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(OUT))
    print(f"saved {OUT.relative_to(HERE)} ({OUT.stat().st_size / 1e6:.1f}mb)")


if __name__ == "__main__":
    main()
