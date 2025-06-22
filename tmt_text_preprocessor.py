import re
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer

class TextPreprocessor:
    """
    Preprocesses a text corpus for neural network training, including:
    - Sentence splitting
    - Normalization (lowercasing, punctuation removal, etc.)
    - Filtering short/non-alphabetic sentences
    - Vectorization (TF-IDF)
    """
    def __init__(self, min_sentence_length: int = 3, remove_punctuation: bool = True):
        self.min_sentence_length = min_sentence_length
        self.remove_punctuation = remove_punctuation
        self.vectorizer: Optional[TfidfVectorizer] = None

    def split_sentences(self, text: str) -> List[str]:
        # Split on . ! ? and newlines
        sentences = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def normalize_sentence(self, sentence: str) -> str:
        s = sentence.lower()
        if self.remove_punctuation:
            s = re.sub(r'[^a-z0-9\s]', '', s)
        s = re.sub(r'\s+', ' ', s)
        return s.strip()

    def filter_sentences(self, sentences: List[str]) -> List[str]:
        filtered = []
        for s in sentences:
            norm = self.normalize_sentence(s)
            if len(norm.split()) >= self.min_sentence_length and any(c.isalpha() for c in norm):
                filtered.append(norm)
        return filtered

    def prepare_sequence_pairs(self, sentences: List[str]) -> Tuple[List[str], List[str]]:
        X, y = [], []
        for i in range(len(sentences) - 1):
            if sentences[i] and sentences[i+1]:
                X.append(sentences[i])
                y.append(sentences[i+1])
        return X, y

    def vectorize(self, X_sentences: List[str], y_sentences: List[str]) -> Tuple:
        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(X_sentences)
        y = self.vectorizer.transform(y_sentences)
        # Ensure X and y have the same shape
        min_len = min(X.shape[0], y.shape[0])
        X = X[:min_len, :]
        y = y[:min_len, :]
        X = X.toarray()
        y = y.toarray()
        min_rows = min(X.shape[0], y.shape[0])
        min_cols = min(X.shape[1], y.shape[1])
        X = X[:min_rows, :min_cols]
        y = y[:min_rows, :min_cols]
        return X, y

    def preprocess_text_for_sequence(self, text: str) -> Tuple:
        sentences = self.split_sentences(text)
        filtered = self.filter_sentences(sentences)
        if len(filtered) < 2:
            raise ValueError("Not enough valid sentences for sequence training.")
        X_sent, y_sent = self.prepare_sequence_pairs(filtered)
        return self.vectorize(X_sent, y_sent)

    def preprocess_text_for_qa(self, qa_pairs: List[Tuple[str, str]]) -> Tuple:
        # For Q/A, vectorize questions and answers
        questions = [self.normalize_sentence(q) for q, a in qa_pairs]
        answers = [self.normalize_sentence(a) for q, a in qa_pairs]
        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(questions).toarray()
        y = self.vectorizer.transform(answers).toarray()
        min_len = min(X.shape[0], y.shape[0])
        X = X[:min_len, :]
        y = y[:min_len, :]
        min_cols = min(X.shape[1], y.shape[1])
        X = X[:, :min_cols]
        y = y[:, :min_cols]
        return X, y

    def preprocess_corpus(self, text: str) -> Tuple:
        """
        Preprocess a raw text corpus into a trainable dataset (X, y) for general text modeling.
        This method splits, normalizes, filters, and vectorizes the text, returning X (features) and y (targets).
        Here, X is the vectorized form of each sentence except the last, and y is the vectorized form of the next sentence.
        """
        sentences = self.split_sentences(text)
        filtered = self.filter_sentences(sentences)
        if len(filtered) < 2:
            raise ValueError("Not enough valid sentences in corpus for training.")
        X_sent, y_sent = self.prepare_sequence_pairs(filtered)
        return self.vectorize(X_sent, y_sent)
