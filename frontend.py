# app.py
import joblib
import streamlit as st

from utils import preprocess

st.set_page_config(
    page_title="Review Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)

@st.cache_resource
def load_sentiment_pipeline():
    return joblib.load("sentiment_pipeline.joblib")

try:
    pipeline = load_sentiment_pipeline()
except Exception as e:
    st.error(f"Error loading model pipeline: {e}")
    st.info("Make sure 'sentiment_pipeline.joblib' and 'utils.py' are in the same folder.")
    st.stop()

st.title("💬 Review Sentiment Analyzer")
st.write("Enter customer review or feedback below to predict whether the sentiment is **Positive**, **Neutral**, or **Negative**.")

user_text = st.text_area(
    "Customer Feedback / Review:",
    height=150,
    placeholder="Type or paste your review here (e.g., 'The delivery was late, but the product quality exceeded my expectations!')..."
)

if st.button("Analyze Sentiment", type="primary"):
    if not user_text.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        with st.spinner("Processing text and predicting sentiment..."):
            prediction = pipeline.predict([user_text])[0]
            probabilities = pipeline.predict_proba([user_text])[0]
            classes = pipeline.classes_

            max_prob = max(probabilities)
            confidence_pct = max_prob * 100

        st.divider()
        st.subheader("Results")

        col1, col2 = st.columns(2)

        with col1:
            if prediction == "positive":
                st.success(f"**Predicted Sentiment:** {prediction.upper()} 😄")
            elif prediction == "neutral":
                st.info(f"**Predicted Sentiment:** {prediction.upper()} 😐")
            else:
                st.error(f"**Predicted Sentiment:** {prediction.upper()} 😟")

        with col2:
            st.metric(
                label="Model Confidence",
                value=f"{confidence_pct:.2f}%"
            )

        st.write("### Probability Breakdown")

        prob_dict = dict(zip(classes, probabilities))

        for label in ["positive", "neutral", "negative"]:
            if label in prob_dict:
                prob_val = prob_dict[label]
                st.write(f"**{label.capitalize()}**: `{prob_val * 100:.1f}%`")
                st.progress(float(prob_val))