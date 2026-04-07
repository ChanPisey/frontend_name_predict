"""
Gender Prediction Service (Khmer First Names)
Updated for PyTorch 2.6-safe loading + support quantized checkpoints.
"""

import json
import gzip
import logging
import numpy as np
import torch
from pathlib import Path

from model import BiLSTMGender
from utils import cleanup_str, seg_kcc

logger = logging.getLogger(__name__)


class GenderPredictor:
    def __init__(
        self,
        model_path="optimized_gender_model.pt",
        kcc2idx_path="kcc2idx.json",
        fasttext_path="cc.km.300.vec.gz",
        device=None,
    ):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model hyperparameters (must match training configuration)
        self.EMBEDDING_DIM = 300
        self.HIDDEN_DIM = 128
        self.NUM_LAYERS = 5
        self.DROPOUT = 0.3
        self.MAX_LEN = 30

        self.model_path = Path(model_path)

        logger.info("Loading vocabulary from %s...", kcc2idx_path)
        with open(kcc2idx_path, "r", encoding="utf-8") as f:
            self.kcc2idx = json.load(f)
        logger.info("Vocabulary loaded: %d tokens", len(self.kcc2idx))

        logger.info("Loading FastText embeddings from %s...", fasttext_path)
        self.fasttext_embeddings = self._load_fasttext_vectors(fasttext_path)
        logger.info("FastText embeddings loaded: %d vectors", len(self.fasttext_embeddings))

        logger.info("Creating embedding matrix...")
        embedding_matrix = self._create_embedding_matrix()
        embedding_matrix_torch = torch.tensor(embedding_matrix, dtype=torch.float32)

        logger.info("Initializing model...")
        self.model = BiLSTMGender(
            vocab_size=len(self.kcc2idx),
            embedding_dim=self.EMBEDDING_DIM,
            hidden_dim=self.HIDDEN_DIM,
            num_layers=self.NUM_LAYERS,
            dropout=self.DROPOUT,
            pretrained_embeddings=embedding_matrix_torch,
        ).to(self.device)

        logger.info("Loading model weights from %s...", self.model_path)
        self._load_checkpoint_auto(self.model, self.model_path)
        self.model.eval()
        logger.info("Model ready for predictions!")

    # ── Checkpoint loading ────────────────────────────────────────────────────
    def _load_checkpoint_auto(self, model: torch.nn.Module, path: Path):
        """
        Loads float or quantized state_dict. Tries weights_only=True first (PyTorch 2.6 safe).
        """
        e1 = None

        def _try_load(weights_only: bool):
            return torch.load(path, map_location=self.device, weights_only=weights_only)

        try:
            state = _try_load(weights_only=True)
            self._apply_state_dict(model, state)
            return
        except Exception as err:
            e1 = err
            logger.warning("weights_only=True load failed: %s", e1)

        state = _try_load(weights_only=False)
        try:
            self._apply_state_dict(model, state)
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load checkpoint: {path}\n"
                f"weights_only=True error: {e1}\n"
                f"weights_only=False error: {e2}"
            )

    def _apply_state_dict(self, model: torch.nn.Module, state):
        if hasattr(state, "state_dict") and not isinstance(state, dict):
            state = state.state_dict()

        if not isinstance(state, dict):
            raise TypeError(f"Unsupported checkpoint type: {type(state)}")

        keys = list(state.keys())
        is_quantized = any(
            k.startswith("fc._packed_params") or k.startswith("lstm._all_weight_values")
            for k in keys
        )

        if is_quantized:
            if self.device.type != "cpu":
                raise RuntimeError(
                    "Quantized checkpoint detected. Run on CPU or use a float checkpoint."
                )
            logger.info("Quantized checkpoint detected. Quantizing model before loading weights...")
            qmodel = torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
            ).eval()
            qmodel.load_state_dict(state)
            self.model = qmodel
            return

        model.load_state_dict(state)

    # ── FastText + embedding ──────────────────────────────────────────────────
    def _load_fasttext_vectors(self, vec_path):
        embeddings = {}
        with gzip.open(vec_path, "rt", encoding="utf-8", errors="ignore") as f:
            next(f, None)  # skip header
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) < 2:
                    continue
                word = parts[0]
                try:
                    vector = np.array([float(v) for v in parts[1:]], dtype=np.float32)
                    if len(vector) == self.EMBEDDING_DIM:
                        embeddings[word] = vector
                except (ValueError, IndexError):
                    continue
        return embeddings

    def _create_embedding_matrix(self):
        embedding_matrix = np.zeros((len(self.kcc2idx), self.EMBEDDING_DIM), dtype=np.float32)
        for kcc, idx in self.kcc2idx.items():
            if kcc == "<PAD>":
                continue
            if kcc in self.fasttext_embeddings:
                embedding_matrix[idx] = self.fasttext_embeddings[kcc]
            else:
                embedding_matrix[idx] = np.random.randn(self.EMBEDDING_DIM).astype(np.float32) * 0.01
        return embedding_matrix

    # ── Prediction ────────────────────────────────────────────────────────────
    def predict(self, name_text: str) -> dict:
        cleaned = cleanup_str(name_text)
        kccs = seg_kcc(cleaned)

        if not kccs:
            return {
                "name": name_text,
                "error": "Unable to segment name into KCCs",
                "gender": None,
                "confidence": None,
                "probability": None,
                "kccs": None,
            }

        encoded = [self.kcc2idx.get(kcc, 1) for kcc in kccs]  # 1 = <UNK>
        actual_len = min(len(encoded), self.MAX_LEN)
        padded = (encoded + [0] * self.MAX_LEN)[:self.MAX_LEN]

        input_ids = torch.tensor([padded], dtype=torch.long).to(self.device)
        length = torch.tensor([max(1, actual_len)], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, length)
            probability = torch.sigmoid(logits).item()

        gender = "Female" if probability > 0.5 else "Male"
        confidence = probability if gender == "Female" else (1 - probability)

        return {
            "name": name_text,
            "gender": gender,
            "confidence": confidence * 100,
            "probability": probability,
            "kccs": kccs,
        }

    def batch_predict(self, names: list) -> list:
        return [self.predict(name) for name in names]
