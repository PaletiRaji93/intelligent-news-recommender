from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def create_models():
    """
    Create the baseline classification models.
    """

    models = {

        "Naive Bayes": MultinomialNB(),

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=2.0
        ),

        "Linear SVM": LinearSVC(
            C=1.0
        )
    }

    return models