import os
from typing import List

class ONNXEmbedder:
    """A lightweight ONNX embedder adapter.

    Notes:
    - Expects an ONNX model at a path defined by env `EMBEDDER_ONNX_MODEL` or passed model_path.
    - The adapter attempts to call session.run(None, {input_name: data}) and extract a float vector.
    - Real models may require preprocessing/tokenization; this adapter is a minimal wrapper.
    """
    def __init__(self, model_path: str = None, provider: str = None):
        self.model_path = model_path or os.environ.get('EMBEDDER_ONNX_MODEL')
        if not self.model_path:
            raise RuntimeError('EMBEDDER_ONNX_MODEL not set and no model_path provided')
        try:
            import onnxruntime as ort
        except Exception as e:
            raise RuntimeError('onnxruntime is required for ONNXEmbedder') from e
        # load session
        self.ort = ort
        # provider handling left minimal; default providers will be used
        try:
            if provider:
                self.session = ort.InferenceSession(self.model_path, providers=[provider])
            else:
                self.session = ort.InferenceSession(self.model_path)
        except Exception as e:
            raise RuntimeError(f'failed to load ONNX model: {e}')
        # Infer the first input name
        try:
            self.input_name = self.session.get_inputs()[0].name
        except Exception:
            self.input_name = None
        # Try to infer output embedding dim
        try:
            out0 = self.session.get_outputs()[0]
            shape = out0.shape
            # shape often [1, d]
            if shape and len(shape) >= 2 and isinstance(shape[1], int):
                self.embedding_dim = int(shape[1])
            else:
                self.embedding_dim = 8
        except Exception:
            self.embedding_dim = 8

    def _text_to_input_array(self, text: str, d: int):
        import hashlib
        import numpy as _np
        h = hashlib.sha256(text.encode('utf-8')).digest()
        vals = []
        i = 0
        while len(vals) < d:
            b = h[i % len(h)]
            vals.append((b / 255.0) * 2.0 - 1.0)
            i += 1
        return _np.array([vals], dtype=_np.float32)

    def embed(self, text: str) -> List[float]:
        import numpy as _np
        d = getattr(self, 'embedding_dim', 8)
        arr = self._text_to_input_array(text, d)
        inp = {self.input_name or 'input': arr}
        try:
            out = self.session.run(None, inp)
            if not out:
                return []
            vec = out[0]
            # vec may be (1,d) numpy array
            try:
                flat = _np.asarray(vec).reshape(-1)
                return [float(x) for x in flat]
            except Exception:
                # Fallback to list parsing
                if isinstance(vec, (list, tuple)) and len(vec) > 0 and isinstance(vec[0], (list, tuple)):
                    vec = vec[0]
                return [float(x) for x in vec]
        except Exception as e:
            raise RuntimeError(f'ONNX embedder inference failed: {e}')
