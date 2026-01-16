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

    def embed(self, text: str) -> List[float]:
        # Minimal wrapper: send raw text as input under inferred input name or 'input'
        inp = {self.input_name or 'input': [text]}
        try:
            out = self.session.run(None, inp)
            if not out:
                return []
            # Expect output to be list of embeddings for batch; take first row
            vec = out[0]
            # If it's nested like [[...]] take first element
            if isinstance(vec, (list, tuple)) and len(vec) > 0 and isinstance(vec[0], (list, tuple)):
                vec = vec[0]
            return [float(x) for x in vec]
        except Exception as e:
            raise RuntimeError(f'ONNX embedder inference failed: {e}')
