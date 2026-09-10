from pathlib import Path
import pandas as pd


# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Dataset paths
TRAIN_PATH = BASE_DIR / "data" / "raw" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "raw" / "test.csv"


def load_train_data():
    """
    Load the AG News training dataset.
    """
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_PATH}"
        )

    df = pd.read_csv(TRAIN_PATH)

    return df


def load_test_data():
    """
    Load the AG News test dataset.
    """
    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test file not found: {TEST_PATH}"
        )

    df = pd.read_csv(TEST_PATH)

    return df


if __name__ == "__main__":

    train_df = load_train_data()
    test_df = load_test_data()

    print("=" * 60)
    print("TRAIN DATA")
    print("=" * 60)

    print(train_df.head())
    print("\nShape:", train_df.shape)
    print("\nColumns:", train_df.columns.tolist())

    print("\n" + "=" * 60)
    print("TEST DATA")
    print("=" * 60)

    print(test_df.head())
    print("\nShape:", test_df.shape)
    print("\nColumns:", test_df.columns.tolist())