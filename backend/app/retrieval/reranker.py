from sentence_transformers import CrossEncoder

model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank(query, chunks):
    pairs = [
        (query, chunk)
        for chunk in chunks
    ]

    scores = model.predict(pairs)

    ranked = sorted(
        enumerate(zip(chunks, scores)),
        key=lambda item: item[1][1],
        reverse=True,
    )

    return [
        (chunk, score, index)
        for index, (chunk, score) in ranked
    ]