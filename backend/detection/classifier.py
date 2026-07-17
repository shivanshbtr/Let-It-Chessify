"""
Chess OCR — Dual Square Classifier (Synthetic + Physical)
=============================================================
Loads BOTH trained ONNX models and exposes a single interface that
the caller selects via mode="synthetic" | "physical".

Synthetic: 13 classes (Empty + 6 white + 6 black)
Physical:  12 classes (NO Empty — see Section 21.3 of project log).
           Empty is injected at inference time via a confidence
           threshold (see confidence_threshold.py), not predicted
           by the model itself.
"""

import json
import numpy as np
import onnxruntime as ort
from pathlib import Path


DEFAULT_SYNTHETIC_LABEL_MAP = {
    "Empty": 0,
    "wK": 1, "wQ": 2, "wR": 3, "wB": 4, "wN": 5, "wP": 6,
    "bK": 7, "bQ": 8, "bR": 9, "bB": 10, "bN": 11, "bP": 12,
}

DEFAULT_PHYSICAL_LABEL_MAP = {
    "wK": 0, "wQ": 1, "wR": 2, "wB": 3, "wN": 4, "wP": 5,
    "bK": 6, "bQ": 7, "bR": 8, "bB": 9, "bN": 10, "bP": 11,
}


class _SingleClassifier:
    """Internal — one loaded ONNX session + its label map."""

    def __init__(self, model_path, label_map):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name   = self.session.get_inputs()[0].name
        self.idx_to_label = {v: k for k, v in label_map.items()}
        self.num_classes  = len(label_map)

        print(f"[Classifier] Loaded: {model_path.name} "
              f"({model_path.stat().st_size / 1e6:.3f} MB), "
              f"{self.num_classes} classes")

    def _preprocess_for_model(self, pil_img_gray_64):
        """
        Convert an already-preprocessed 64x64 grayscale PIL Image
        into the normalized float32 tensor the model expects.
        Matches training transform: [0,1] -> normalize(0.5,0.5) -> [-1,1]
        """
        arr = np.array(pil_img_gray_64, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        return arr[np.newaxis, np.newaxis, :, :]   # (1, 1, 64, 64)

    def classify_batch(self, crops_dict):
        """
        Args: crops_dict {square_index -> PIL Image, mode "L", 64x64}
        Returns: {square_index -> {"label": str, "confidence": float,
                                    "all_probs": {label: prob}}}
        """
        if not crops_dict:
            return {}

        square_indices = sorted(crops_dict.keys())
        batch = np.concatenate(
            [self._preprocess_for_model(crops_dict[idx]) for idx in square_indices],
            axis=0
        )

        logits = self.session.run(None, {self.input_name: batch})[0]   # (N, C)
        probs  = _softmax_batch(logits)

        results = {}
        for i, sq_idx in enumerate(square_indices):
            best_idx = int(np.argmax(probs[i]))
            results[sq_idx] = {
                "label":      self.idx_to_label[best_idx],
                "confidence": float(probs[i][best_idx]),
                "all_probs":  {self.idx_to_label[j]: float(probs[i][j])
                               for j in range(self.num_classes)},
            }
        return results


class DualClassifier:
    """
    Public interface. Holds both classifiers, loaded lazily on first use
    so a deployment that only ever uses one pipeline doesn't pay the
    load cost of both.
    """

    def __init__(self, synthetic_model_path, synthetic_label_map_path,
                 physical_model_path,  physical_label_map_path):
        self._synthetic_path      = synthetic_model_path
        self._synthetic_map_path  = synthetic_label_map_path
        self._physical_path       = physical_model_path
        self._physical_map_path   = physical_label_map_path
        self._synthetic = None
        self._physical  = None

    def _load_label_map(self, path, default):
        if path and Path(path).exists():
            with open(path) as f:
                return json.load(f)
        return default

    @property
    def synthetic(self):
        if self._synthetic is None:
            label_map = self._load_label_map(self._synthetic_map_path, DEFAULT_SYNTHETIC_LABEL_MAP)
            self._synthetic = _SingleClassifier(self._synthetic_path, label_map)
        return self._synthetic

    @property
    def physical(self):
        if self._physical is None:
            label_map = self._load_label_map(self._physical_map_path, DEFAULT_PHYSICAL_LABEL_MAP)
            self._physical = _SingleClassifier(self._physical_path, label_map)
        return self._physical

    def classify_batch(self, crops_dict, is_physical):
        clf = self.physical if is_physical else self.synthetic
        return clf.classify_batch(crops_dict)

    def is_loaded(self, is_physical):
        return self._physical is not None if is_physical else self._synthetic is not None


def _softmax_batch(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
