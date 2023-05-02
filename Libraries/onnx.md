


```python
import onnx

# LOAD MODEL
onnx_model = onnx.load("model.onnx")

# SAVE MODEL
onnx.save(onnx_model, "model.onnx")

# CREATE Tensors
t1 = onnx.helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, ["BS", 3, "H", "W"])
t2 = onnx.helper.make_tensor_value_info('scalar_tensor', onnx.TensorProto.INT64, [])



def get_input(model, name):
    for elem in model.graph.input:
        if elem.name == name:
            return elem

def get_node(model, name):
    for elem in model.graph.node:
        if elem.name == name:
            return elem

def get_output(model, name):
    for elem in model.graph.output:
        if elem.name == name:
            return elem
```

