from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"

_model = SentenceTransformer(MODEL_NAME)


def embed_text(text: str) -> list[float]:
    return _model.encode(
        text,
        normalize_embeddings=True,
    ).tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    return _model.encode(
        texts,
        normalize_embeddings=True,
    ).tolist()