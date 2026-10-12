# utils.py
import spacy

nlp = spacy.load("en_core_web_sm")

negation_words = {
    "not", "no", "never", "neither", "nor", "none", "nobody", "nothing",
    "nowhere", "hardly", "barely", "scarcely", "without", "cannot", "can't",
    "couldn't", "didn't", "doesn't", "don't", "hadn't", "hasn't", "haven't",
    "isn't", "aren't", "wasn't", "weren't", "won't", "wouldn't", "shouldn't", "mustn't"
}

def preprocess(text_list):
    """Preprocesses a list or Series of text strings using spaCy."""
    processed_list = []
    for text in text_list:
        doc = nlp(str(text).lower())
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_punct
            and (not token.is_stop or token.text in negation_words)
        ]
        processed_list.append(" ".join(tokens))
    return processed_list