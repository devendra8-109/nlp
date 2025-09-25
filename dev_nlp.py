# ============================================
# 📌 NLP Phases for Fake vs Real Detection
# Using Naive Bayes at each step - REVISED FOR STANDARD PYTHON
# ============================================

# NOTE: The lines starting with '!' have been removed.
# You must install the required packages in your environment
# before running this script. Use the 'requirements.txt' file
# I previously provided with the following command in your terminal:
# pip install -r requirements.txt

import pandas as pd
import numpy as np
import nltk, re, string, spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK resources
# This should be done once on a new environment.
# You can uncomment these lines and run the script once if you haven't.
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

# Load spaCy English model
# The spacy model must also be downloaded separately.
# You can do this via: python -m spacy download en_core_web_sm
# The requirements.txt file will handle this if you have the URL.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Please download it by running 'python -m spacy download en_core_web_sm' in your terminal.")
    raise

# ============================
# Step 1: Load Dataset
# ============================
import pandas as pd

# The file path is assumed to be correct for your environment.
# Make sure 'merged_final.csv' is in the same directory as this script.
try:
    df = pd.read_csv("merged_final.csv")
    print("Dataset Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())
except FileNotFoundError:
    print("Error: 'merged_final.csv' not found.")
    print("Please make sure the file is in the same directory as this script.")
    exit()


X = df['statement']
y = df['BinaryTarget']

# ============================
# Helper: Train NB Model
# ============================
def train_nb(X_features, y, name):
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🔹 {name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    return acc

# ============================
# Phase 1: Lexical & Morphological Analysis
# (Tokenization + Stopword removal + Lemmatization)
# ============================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def lexical_preprocess(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and w not in string.punctuation]
    return " ".join(tokens)

X_lexical = X.apply(lexical_preprocess)
vec_lexical = CountVectorizer().fit_transform(X_lexical)
acc1 = train_nb(vec_lexical, y, "Lexical & Morphological Analysis")

# ============================
# Phase 2: Syntactic Analysis (Parsing)
# (Extract POS tags & dependency relations as features)
# ============================
def syntactic_features(text):
    if pd.isna(text):
        return ""
    doc = nlp(str(text))
    pos_tags = " ".join([token.pos_ for token in doc])
    return pos_tags

X_syntax = X.apply(syntactic_features)
vec_syntax = CountVectorizer().fit_transform(X_syntax)
acc2 = train_nb(vec_syntax, y, "Syntactic Analysis")

# ============================
# Phase 3: Semantic Analysis
# (Polarity & Subjectivity from TextBlob)
# ============================
def semantic_features(text):
    if pd.isna(text):
        return f"0.0 0.0"
    blob = TextBlob(str(text))
    return f"{blob.sentiment.polarity} {blob.sentiment.subjectivity}"

X_semantic = X.apply(semantic_features)
vec_semantic = TfidfVectorizer().fit_transform(X_semantic)
acc3 = train_nb(vec_semantic, y, "Semantic Analysis")

# ============================
# Phase 4: Discourse Integration
# (Sentence relations, connectives, length features)
# ============================
def discourse_features(text):
    if pd.isna(text):
        return "0"
    sentences = nltk.sent_tokenize(str(text))
    return f"{len(sentences)} {' '.join([s.split()[0] for s in sentences if len(s.split())>0])}"

X_discourse = X.apply(discourse_features)
vec_discourse = CountVectorizer().fit_transform(X_discourse)
acc4 = train_nb(vec_discourse, y, "Discourse Integration")

# ============================
# Phase 5: Pragmatic Analysis
# (Contextual features: interrogative, exclamatory, modality words)
# ============================

pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]

def pragmatic_features(text):
    features = []
    if pd.isna(text):
        return ""
    text_lower = str(text).lower()
    for w in pragmatic_words:
        count = text_lower.count(w)
        if count > 0:
            features.extend([w] * count)
    return " ".join(features)


X_pragmatic = X.fillna('').astype(str).apply(pragmatic_features)
non_empty_idx = X_pragmatic.str.strip() != ''
X_pragmatic_filtered = X_pragmatic[non_empty_idx]
y_filtered = y[non_empty_idx]

acc5 = 0.0
if X_pragmatic_filtered.empty:
    print("No meaningful pragmatic features found for vectorization.")
else:
    vec_pragmatic = CountVectorizer(stop_words=None).fit_transform(X_pragmatic_filtered)
    acc5 = train_nb(vec_pragmatic, y_filtered, "Pragmatic Analysis")

# ============================
# Final Results
# ============================
print("\n📊 Phase-wise Naive Bayes Accuracies:")
print(f"1. Lexical & Morphological: {acc1:.4f}")
print(f"2. Syntactic: {acc2:.4f}")
print(f"3. Semantic: {acc3:.4f}")
print(f"4. Discourse: {acc4:.4f}")
print(f"5. Pragmatic: {acc5:.4f}")
