from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelManifest:
    name: str
    version: str
    source_url: str
    checksum: Optional[str] = None
    format: Optional[str] = None  # e.g., gguf, onnx, torch
    quantization: Optional[str] = None
    tokenizer: Optional[str] = None
    required_hw: Optional[str] = None  # e.g., "gpu", "cpu"
    license: Optional[str] = None

    def validate(self):
        if not self.name or not self.version or not self.source_url:
            raise ValueError("invalid ModelManifest: name/version/source_url required")
