"""
Gender Prediction Service (Khmer First Names)
Updated for PyTorch 2.6-safe loading + support quantized checkpoints.
"""

import json
import gzip
import numpy as np
import torch
from pathlib import Path

from model import BiLSTMGender
from utils import cleanup_str, seg_kcc


class GenderPredictor:
    def __init__(
        self,
        model_path="optimized_gender_model.pt",
        kcc2idx_path="kcc2idx.json",
        fasttext_path="cc.km.300.vec.gz",
        device=None,
    ):
        """
        Initialize the Gender Predictor

        Args:
            model_path: Path to the trained PyTorch checkpoint
                       - can be float state_dict
                       - or quantized state_dict (optimized_gender_model.pt)
            kcc2idx_path: Path to KCC vocabulary JSON
            fasttext_path: Path to FastText embeddings (.vec.gz file)
            device: torch device (cuda or cpu)
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model hyperparameters (must match training configuration)
        self.EMBEDDING_DIM = 300
        self.HIDDEN_DIM = 128
        self.NUM_LAYERS = 5
        self.DROPOUT = 0.3
        self.MAX_LEN = 30

        self.model_path = Path(model_path)

        # ---------------------------
        # Load vocabulary
        # ---------------------------
        print(f"Loading vocabulary from {kcc2idx_path}...")
        with open(kcc2idx_path, "r", encoding="utf-8") as f:
            self.kcc2idx = json.load(f)
        print(f"✓ Vocabulary loaded: {len(self.kcc2idx)} tokens")

        # ---------------------------
        # Load FastText embeddings
        # ---------------------------
        print(f"Loading FastText embeddings from {fasttext_path}...")
        self.fasttext_embeddings = self._load_fasttext_vectors(fasttext_path)
        print(f"✓ FastText embeddings loaded: {len(self.fasttext_embeddings)} vectors")

        # ---------------------------
        # Create embedding matrix
        # ---------------------------
        print("Creating embedding matrix...")
        embedding_matrix = self._create_embedding_matrix()
        embedding_matrix_torch = torch.tensor(embedding_matrix, dtype=torch.float32)

        # ---------------------------
        # Initialize float model
        # ---------------------------
        print("Initializing model...")
        self.model = BiLSTMGender(
            vocab_size=len(self.kcc2idx),
            embedding_dim=self.EMBEDDING_DIM,
            hidden_dim=self.HIDDEN_DIM,
            num_layers=self.NUM_LAYERS,
            dropout=self.DROPOUT,
            pretrained_embeddings=embedding_matrix_torch,
        ).to(self.device)

        # ---------------------------
        # Load checkpoint (PyTorch 2.6 safe + quantized support)
        # ---------------------------
        print(f"Loading model weights from {self.model_path}...")
        self._load_checkpoint_auto(self.model, self.model_path)

        self.model.eval()
        print("✓ Model loaded successfully and ready for predictions!")

    # ---------------------------------------------------------------------
    # Checkpoint loading (float or quantized) - PyTorch 2.6 safe
    # ---------------------------------------------------------------------
    def _load_checkpoint_auto(self, model: torch.nn.Module, path: Path):
        """
        Loads:
          - float state_dict -> loads directly
          - quantized state_dict -> quantize model first (CPU only), then load

        PyTorch 2.6: tries weights_only=True first, then weights_only=False (trusted only).
        """
        e1 = None

        def _try_load(weights_only: bool):
            return torch.load(path, map_location=self.device, weights_only=weights_only)

        # Try safe load
        try:
            state = _try_load(weights_only=True)
            self._apply_state_dict(model, state)
            return
        except Exception as err:
            e1 = err
            print(f"⚠️ weights_only=True load failed: {e1}")

        # Fallback: legacy unpickling (ONLY if you trust your checkpoint)
        state = _try_load(weights_only=False)
        try:
            self._apply_state_dict(model, state)
            return
        except Exception as e2:
            raise RuntimeError(
                f"❌ Failed to load checkpoint: {path}\n"
                f"weights_only=True error: {e1}\n"
                f"weights_only=False error: {e2}\n"
                f"Tip: Prefer saving float weights: torch.save(model.state_dict(), 'best_gender_model_state_dict.pt')"
            )

    def _apply_state_dict(self, model: torch.nn.Module, state):
        """
        Apply state to model. Detects quantized checkpoints and loads correctly.
        """
        # If a full model object was saved, convert to state_dict
        if hasattr(state, "state_dict") and not isinstance(state, dict):
            state = state.state_dict()

        if not isinstance(state, dict):
            raise TypeError(f"Unsupported checkpoint type: {type(state)}")

        keys = list(state.keys())

        # Detect quantized checkpoint signature keys
        is_quantized = any(
            k.startswith("fc._packed_params") or k.startswith("lstm._all_weight_values")
            for k in keys
        )

        if is_quantized:
            # Quantized dynamic models should run on CPU
            if self.device.type != "cpu":
                raise RuntimeError(
                    "This checkpoint is quantized (packed params). "
                    "Please run on CPU (set device='cpu') or use a float checkpoint."
                )

            print("Detected quantized checkpoint. Quantizing model before loading weights...")
            qmodel = torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
            ).eval()

            qmodel.load_state_dict(state)
            self.model = qmodel
            return

        # Normal float state_dict
        model.load_state_dict(state)
        return

    # ---------------------------------------------------------------------
    # FastText + embedding
    # ---------------------------------------------------------------------
    def _load_fasttext_vectors(self, vec_path):
        """Load FastText vectors from .vec.gz file"""
        embeddings = {}
        with gzip.open(vec_path, "rt", encoding="utf-8", errors="ignore") as f:
            # Skip header
            next(f, None)

            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) < 2:
                    continue

                word = parts[0]
                try:
                    vector = np.array([float(val) for val in parts[1:]], dtype=np.float32)
                    if len(vector) == self.EMBEDDING_DIM:
                        embeddings[word] = vector
                except (ValueError, IndexError):
                    continue

        return embeddings

    def _create_embedding_matrix(self):
        """Create embedding matrix from FastText vectors"""
        embedding_matrix = np.zeros((len(self.kcc2idx), self.EMBEDDING_DIM), dtype=np.float32)

        for kcc, idx in self.kcc2idx.items():
            if kcc == "<PAD>":
                continue

            if kcc in self.fasttext_embeddings:
                embedding_matrix[idx] = self.fasttext_embeddings[kcc]
            else:
                embedding_matrix[idx] = np.random.randn(self.EMBEDDING_DIM).astype(np.float32) * 0.01

        return embedding_matrix

    # ---------------------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------------------
    def predict(self, name_text):
        """
        Predict gender from a Khmer first name
        """
        self.model.eval()

        cleaned = cleanup_str(name_text)
        kccs = seg_kcc(cleaned)

        if not kccs:
            return {
                "name": name_text,
                "error": "Unable to segment name into KCCs",
                "gender": None,
                "confidence": None,
            }

        encoded = [self.kcc2idx.get(kcc, 1) for kcc in kccs]  # 1 is <UNK>

        actual_len = min(len(encoded), self.MAX_LEN)
        if len(encoded) < self.MAX_LEN:
            padded = encoded + [0] * (self.MAX_LEN - len(encoded))
        else:
            padded = encoded[: self.MAX_LEN]

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
    def batch_predict(self, names):
        return [self.predict(name) for name in names]