#!/usr/bin/env python3
"""Create a tiny identity ONNX model with float input/output shape [1,d].
Usage: python scripts/make_tiny_onnx.py --dim 8 --out /tmp/tiny.onnx
"""
import argparse
try:
    import onnx
    from onnx import helper, TensorProto
except Exception as e:
    raise RuntimeError('onnx package is required to create an ONNX model') from e

parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=8)
parser.add_argument('--out', type=str, required=True)
args = parser.parse_args()

d = args.dim
input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, d])
output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, d])
node = helper.make_node('Identity', inputs=['input'], outputs=['output'], name='IdentityNode')
graph = helper.make_graph([node], 'identity_graph', [input], [output])
model = helper.make_model(graph)
onnx.checker.check_model(model)
onnx.save(model, args.out)
print(f'Wrote tiny ONNX identity model (dim={d}) to {args.out}')
