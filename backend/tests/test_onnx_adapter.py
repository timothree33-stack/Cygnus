import os
import pytest


def make_fake_session(return_vec):
    class FakeSession:
        def __init__(self, path, providers=None):
            self.path = path
        def get_inputs(self):
            class IN:
                def __init__(self):
                    self.name = 'input'
            return [IN()]
        def run(self, out_names, inp):
            # Simulate returning [[vector]]
            return [ [ return_vec ] ]
    return FakeSession


def test_onnx_adapter_loads_and_runs(monkeypatch, tmp_path):
    # Create a dummy model file path
    model_path = str(tmp_path / 'tiny.onnx')
    open(model_path, 'wb').close()

    # Patch onnxruntime.InferenceSession to our fake
    fake_vec = [0.1, 0.2, 0.3]
    fake_sess = make_fake_session(fake_vec)
    monkeypatch.setitem(__import__('sys').modules, 'onnxruntime', type('m', (), {'InferenceSession': fake_sess}))

    os.environ['EMBEDDER_TYPE'] = 'onnx'
    os.environ['EMBEDDER_ONNX_MODEL'] = model_path

    from backend.embeddings import get_embedder
    emb = get_embedder()
    v = emb.embed('hello onnx')
    assert isinstance(v, list)
    assert v == fake_vec


def test_onnx_scoring_integration(monkeypatch, tmp_path):
    # Ensure score_pair uses the ONNX embedder when EMBEDDER_TYPE=onnx
    fake_vec_a = [0.1, 0.1, 0.1]
    fake_vec_b = [0.8, 0.8, 0.8]

    def make_session_for(a_vec):
        class FakeSession:
            def __init__(self, path, providers=None):
                self.path = path
            def get_inputs(self):
                class IN:
                    def __init__(self):
                        self.name = 'input'
                return [IN()]
            def run(self, out_names, inp):
                return [ [ a_vec ] ]
        return FakeSession

    # monkeypatch InferenceSession factory to return different vectors depending on input text
    def FakeInferenceSession(path, providers=None):
        # simple switch: if path contains 'a' return a_vec else b_vec
        if 'a' in path:
            return make_session_for(fake_vec_a)(path)
        else:
            return make_session_for(fake_vec_b)(path)

    monkeypatch.setitem(__import__('sys').modules, 'onnxruntime', type('m', (), {'InferenceSession': FakeInferenceSession}))
    os.environ['EMBEDDER_TYPE'] = 'onnx'
    os.environ['EMBEDDER_ONNX_MODEL'] = str(tmp_path / 'model_a.onnx')

    from backend.mcp.hiemdall import score_pair
    sa, sb = score_pair('a','b', strategy='embed')
    assert isinstance(sa, int) and isinstance(sb, int)
    assert sa != sb
