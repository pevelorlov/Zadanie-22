from __future__ import annotations

from threading import RLock


DEFAULT_MODEL = "intfloat/multilingual-e5-small"


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self._lock = RLock()

    @property
    def model(self):
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as error:
                    raise RuntimeError("Установите sentence-transformers для локальных эмбеддингов.") from error
                self._model = SentenceTransformer(self.model_name, device="cpu")
            return self._model

    @property
    def tokenizer(self):
        return self.model.tokenizer

    @property
    def dimensions(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str], batch_size: int = 16):
        import numpy as np
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        values = self.model.encode(
            [f"passage: {text}" for text in texts],
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(values, dtype=np.float32)

    def embed_query(self, query: str):
        import numpy as np
        values = self.model.encode(
            [f"query: {query}"],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(values[0], dtype=np.float32)
