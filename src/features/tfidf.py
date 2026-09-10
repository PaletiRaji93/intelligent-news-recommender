from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_vectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
):
    """
    Create and return a TF-IDF vectorizer.

    Parameters
    ----------
    max_features : int
        Maximum number of vocabulary features.

    ngram_range : tuple
        Use unigrams and bigrams by default.

    min_df : int
        Ignore terms appearing in fewer than min_df documents.

    max_df : float
        Ignore terms appearing in more than this proportion
        of documents.

    sublinear_tf : bool
        Apply logarithmic scaling to term frequency.
    """

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf
    )

    return vectorizer