import joblib
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from utils import preprocess


BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(
    BASE_DIR / "sentiment_reviews.csv",
    usecols=["text", "sentiment"]
)

df = df.dropna()
df["text"] = df["text"].str.strip()
df = df[df["text"] != ""]

labels = ["positive", "neutral", "negative"]
df = df[df["sentiment"].isin(labels)]
df = df.reset_index(drop=True)


X_train, X_test, Y_train, Y_test = train_test_split(
    df["text"], df["sentiment"], test_size=0.33, random_state=42
)


sentiment_pipeline = Pipeline([
    ("preprocessor", FunctionTransformer(preprocess, validate=False)),
    ("vectorizer", TfidfVectorizer()),
    ("classifier", LogisticRegression()),
])

sentiment_pipeline.fit(X_train, Y_train)

prediction = sentiment_pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(Y_test, prediction):.4f}")
print("\nClassification Report:\n", classification_report(Y_test, prediction))


joblib.dump(
    sentiment_pipeline,
    BASE_DIR / "sentiment_pipeline.joblib",
    compress=3
)

print("Pipeline saved as 'sentiment_pipeline.joblib'")
