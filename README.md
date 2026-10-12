# Customer Sentiment Analysis using NLP and Machine Learning

## Project Overview

Customer Sentiment Analysis is a Natural Language Processing (NLP) and Machine Learning project that analyzes customer reviews and classifies them into three sentiment categories:

- **Positive** — Reviews expressing satisfaction, appreciation, or a favorable opinion.
- **Negative** — Reviews expressing dissatisfaction, complaints, or unfavorable opinions.
- **Neutral** — Reviews expressing factual information or opinions without a clearly positive or negative sentiment.

The project uses text preprocessing, TF-IDF (Term Frequency–Inverse Document Frequency) feature extraction, and Logistic Regression to predict the sentiment of customer reviews.

The trained model is saved as a reusable pipeline that can be loaded by a Python frontend for sentiment prediction.

## Features

- Text preprocessing using a custom preprocessing function.
- Removal of missing and empty reviews.
- Filtering of invalid sentiment labels.
- TF-IDF vectorization for converting text into numerical features.
- Logistic Regression for multiclass sentiment classification.
- Training and testing using an 80/20-style split? No — the current implementation uses a 67/33 train-test split.
- Model evaluation using accuracy, precision, recall, F1-score, and a classification report.
- Saving the complete preprocessing and prediction pipeline using Joblib.
- Reusable model for integration with a frontend interface.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Dataset loading and data cleaning |
| scikit-learn | Machine learning, TF-IDF, pipeline, and evaluation |
| spaCy | NLP preprocessing, if used by `utils.py` |
| Joblib | Saving and loading the trained pipeline |
| CSV | Storing customer review data |



## Machine Learning Workflow

The project follows these steps:

1. **Load the dataset:** Read the CSV file using Pandas.
2. **Clean the data:** Remove missing values, strip extra whitespace, and discard empty reviews.
3. **Filter sentiment labels:** Retain only positive, neutral, and negative examples.
4. **Split the dataset:** Use 67% of the data for training and 33% for testing.
5. **Preprocess text:** Apply the custom `preprocess()` function from `utils.py`.
6. **Extract features:** Convert the processed reviews into numerical TF-IDF vectors.
7. **Train the classifier:** Fit a Logistic Regression model to the training data.
8. **Evaluate performance:** Generate accuracy and a classification report using the test data.
9. **Save the model:** Export the complete pipeline using Joblib.
10. **Make predictions:** Load the saved pipeline in the frontend to classify new reviews.

## Model Architecture

Customer Reviews (CSV)
          |
          v
     Data Cleaning
          |
          v
   Train-Test Split
          |
          v
  Text Preprocessing
      (utils.py)
          |
          v
   TF-IDF Vectorizer
          |
          v
  Logistic Regression
          |
          v
 Predicted Sentiment
          |
          v
       Evaluation
          |
          v
 Save Pipeline (.joblib)
          |
          v
 Frontend Integration


## Installation and Setup

### 1. Install Python

Install Python 3.12 or another compatible Python version.


### 2. Navigate to the project directory

Open a terminal in the project folder:


cd "C:\Users\plama\OneDrive\Desktop\python\NLP\SentimentModel"


### 3. Create a virtual environment (recommended)


python -m venv .venv


Activate it in PowerShell:


.\.venv\Scripts\Activate.ps1


### 4. Install dependencies


python -m pip install pandas scikit-learn joblib spacy


If your `utils.py` uses spaCy's English language model, download it with:

python -m spacy download en_core_web_sm


## Training the Model

Run the training script:

python SentimentAnalysis.py


The script will:

- Load and clean the dataset.
- Train the TF-IDF and Logistic Regression pipeline.
- Print the model's accuracy.
- Display the classification report.
- Save the trained pipeline as `sentiment_pipeline.joblib`.

The saved pipeline includes the preprocessing transformer, TF-IDF vectorizer, and trained classifier.

## Model Evaluation

The model is evaluated using:

- **Accuracy:** Proportion of correctly predicted sentiments.
- **Precision:** Proportion of predicted examples of a class that are correct.
- **Recall:** Proportion of actual examples of a class that are correctly identified.
- **F1-score:** Harmonic mean of precision and recall.

The script prints the evaluation results to the terminal.

Actual performance depends on dataset quality, class distribution, duplicates, preprocessing, and how well the test data represents unseen reviews. No fixed accuracy is claimed in this documentation.



## Frontend Integration

The `frontend.py` file can provide a user interface where users enter a customer review and receive a predicted sentiment.

A typical interaction is:

1. The user enters a review.
2. The frontend passes the text to the loaded pipeline.
3. The pipeline preprocesses and transforms the text.
4. Logistic Regression predicts the sentiment.
5. The frontend displays the predicted category.


## Limitations

- Predictions depend on the quality and diversity of the training dataset.
- Sarcasm, mixed emotions, slang, and context-dependent language can be difficult to classify.
- TF-IDF primarily represents word importance and does not inherently understand deep semantic context.
- Logistic Regression performance may vary across domains, such as electronics, restaurants, and customer support.
- Duplicate or near-duplicate reviews can cause overly optimistic evaluation results if similar examples appear in both training and testing sets.
- The model's performance should be measured on unseen data before practical deployment.


