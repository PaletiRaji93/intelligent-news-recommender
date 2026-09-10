import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


class NewsRecommender:

    def __init__(self, vectorizer, news_data):
        """
        Initialize the recommendation system.

        Parameters
        ----------
        vectorizer : fitted TF-IDF vectorizer
            The TF-IDF vectorizer trained on the training corpus.

        news_data : pandas DataFrame
            DataFrame containing the news articles.
        """

        self.vectorizer = vectorizer
        self.news_data = news_data.copy()

        # Convert all article text into TF-IDF vectors
        self.tfidf_matrix = self.vectorizer.transform(
            self.news_data["clean_text"].fillna("")
        )

    def recommend(self, query, top_k=5):
        """
        Recommend the most similar news articles.

        Parameters
        ----------
        query : str
            Input news article or text.

        top_k : int
            Number of recommendations.

        Returns
        -------
        pandas.DataFrame
            Recommended articles.
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "Query must be a non-empty string."
            )

        # Convert query into TF-IDF vector
        query_vector = self.vectorizer.transform([query])

        # Calculate cosine similarity
        similarity_scores = cosine_similarity(
            query_vector,
            self.tfidf_matrix
        )[0]

        # Get indices of highest similarity scores
        top_indices = np.argsort(
            similarity_scores
        )[::-1][:top_k]

        # Copy recommended articles
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