import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# --------------------------------------------------
# Download required NLTK resources
# --------------------------------------------------

def download_nltk_resources():
    """
    Download the NLTK resources required for preprocessing.
    """

    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("tokenizers/punkt", "punkt"),
    ]

    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package)


# Download resources when this module is imported
download_nltk_resources()


# --------------------------------------------------
# Initialize preprocessing objects
# --------------------------------------------------

STOP_WORDS = set(stopwords.words("english"))

LEMMATIZER = WordNetLemmatizer()


# --------------------------------------------------
# Basic text cleaning
# --------------------------------------------------

def clean_text(text):
    """
    Perform basic text cleaning.

    Steps:
    1. Convert to string
    2. Lowercase
    3. Remove HTML
    4. Remove URLs
    5. Remove email addresses
    6. Remove punctuation
    7. Remove numbers
    8. Normalize whitespace
    """

    if not isinstance(text, str):
        text = str(text)

    # Lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------------------------
# Tokenization
# --------------------------------------------------

def tokenize_text(text):
    """
    Tokenize cleaned text.
    """

    return nltk.word_tokenize(text)


# --------------------------------------------------
# Stopword removal
# --------------------------------------------------

def remove_stopwords(tokens):
    """
    Remove English stopwords.

    Note:
    Negation words such as 'not', 'no', and 'never'
    are preserved because they can carry meaning.
    """

    important_negations = {
        "no",
        "not",
        "never",
        "nor",
        "neither"
    }

    filtered_tokens = [
        token
        for token in tokens
        if (
            token not in STOP_WORDS
            or token in important_negations
        )
    ]

    return filtered_tokens


# --------------------------------------------------
# Lemmatization
# --------------------------------------------------

def lemmatize_tokens(tokens):
    """
    Lemmatize each token.
    """

    lemmatized_tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
    ]

    return lemmatized_tokens


# --------------------------------------------------
# Complete preprocessing pipeline
# --------------------------------------------------

def preprocess_text(text):
    """
    Complete NLP preprocessing pipeline.

    Raw text
        ↓
    Cleaning
        ↓
    Tokenization
        ↓
    Stopword removal
        ↓
    Lemmatization
        ↓
    Clean text
    """

    # Step 1: Basic cleaning
    text = clean_text(text)

    # Step 2: Tokenization
    tokens = tokenize_text(text)

    # Step 3: Stopword removal
    tokens = remove_stopwords(tokens)

    # Step 4: Lemmatization
    tokens = lemmatize_tokens(tokens)

    # Convert tokens back to text
    cleaned_text = " ".join(tokens)

    return cleaned_text


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    sample_text = """
    Apple has launched a NEW artificial intelligence
    technology! Visit https://example.com for more information.
    """

    print("Original text:")
    print(sample_text)

    print("\nProcessed text:")
    print(preprocess_text(sample_text))