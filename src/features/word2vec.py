from gensim.models import Word2Vec


def prepare_sentences(text_series):
    """
    Convert cleaned text into tokenized sentences.

    Example:
    "apple launches new iphone"
    ->
    ["apple", "launches", "new", "iphone"]
    """

    sentences = []

    for text in text_series:
        if not isinstance(text, str):
            text = ""

        tokens = text.split()

        if tokens:
            sentences.append(tokens)

    return sentences


def train_word2vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    epochs=10
):
    """
    Train Word2Vec model on the training corpus.
    """

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,
        epochs=epochs
    )

    return model


def document_vector(model, tokens):
    """
    Convert a document into a single vector
    by averaging its word vectors.
    """

    vectors = []

    for word in tokens:

        if word in model.wv:
            vectors.append(model.wv[word])

    if not vectors:
        return [0.0] * model.vector_size

    return sum(vectors) / len(vectors)


def create_avg_word2vec_vectors(model, text_series):
    """
    Convert every document into an AvgWord2Vec vector.
    """

    document_vectors = []

    for text in text_series:

        if not isinstance(text, str):
            text = ""

        tokens = text.split()

        vector = document_vector(model, tokens)

        document_vectors.append(vector)

    return document_vectors