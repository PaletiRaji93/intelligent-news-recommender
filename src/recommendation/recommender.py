import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class NewsRecommender:

    def __init__(self, vectorizer, news_data):
        self.vectorizer = vectorizer
        self.news_data = news_data.copy()

        # Convert all news articles into TF-IDF vectors
        self.tfidf_matrix = self.vectorizer.transform(
            self.news_data["clean_text"].fillna("")
        )

    def recommend(
        self,
        query,
        top_k=5,
        exclude_index=None
    ):
        """
        Recommend the most similar news articles.

        Parameters
        ----------
        query : str
            News article text or user query.

        top_k : int
            Number of recommendations to return.

        exclude_index : int, optional
            Index of the article to exclude from recommendations.

        Returns
        -------
        pandas.DataFrame
            Top recommended articles with similarity scores.
        """

        # -------------------------------------------------
        # Validate query
        # -------------------------------------------------

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "Query must be a non-empty string."
            )

        # -------------------------------------------------
        # Convert query into TF-IDF vector
        # -------------------------------------------------

        query_vector = self.vectorizer.transform([query])

        # -------------------------------------------------
        # Calculate cosine similarity
        # -------------------------------------------------

        similarity_scores = cosine_similarity(
            query_vector,
            self.tfidf_matrix
        )[0]

        # -------------------------------------------------
        # Exclude original article
        # -------------------------------------------------

        if exclude_index is not None:

            # Accept both native Python ints and numpy integer types
            # (e.g. values coming from DataFrame.iloc / .name), which
            # `isinstance(exclude_index, int)` alone would reject.
            if (
                isinstance(exclude_index, (int, np.integer))
                and 0 <= exclude_index < len(similarity_scores)
            ):
                similarity_scores[int(exclude_index)] = -1

        # -------------------------------------------------
        # Get top-k articles
        # -------------------------------------------------

        top_indices = np.argsort(
            similarity_scores
        )[::-1][:top_k]

        # -------------------------------------------------
        # Create recommendation DataFrame
        # -------------------------------------------------

        recommendations = self.news_data.iloc[
            top_indices
        ].copy()

        # Add similarity score
        recommendations["similarity_score"] = (
            similarity_scores[top_indices]
        )

        # Reset index
        recommendations.reset_index(
            drop=True,
            inplace=True
        )

        return recommendations
