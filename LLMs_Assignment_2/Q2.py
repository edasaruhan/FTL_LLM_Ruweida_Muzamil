import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch

# Load dataset
df = pd.read_csv("Corona_NLP_train.csv")

# Data cleaning function
def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove non-alphanumeric characters
    text = text.lower()               # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text

# Apply cleaning
df['text'] = df['text'].apply(clean_text)

# Encode labels for classification
le = LabelEncoder()
df['label'] = le.fit_transform(df['category'])  # Assuming 'category' column is present

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)



# Train Word2Vec model
sentences = [text.split() for text in X_train]
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, sg=1)

# Function to vectorize text using Word2Vec
def get_w2v_embeddings(text, model, size):
    words = text.split()
    vec = np.zeros(size)
    count = 0
    for word in words:
        if word in model.wv:
            vec += model.wv[word]
            count += 1
    return vec / count if count > 0 else vec

# Convert train and test sets
X_train_w2v = np.array([get_w2v_embeddings(text, w2v_model, 100) for text in X_train])
X_test_w2v = np.array([get_w2v_embeddings(text, w2v_model, 100) for text in X_test])


import gensim.downloader as api

# Load pre-trained GloVe embeddings
glove_vectors = api.load("glove-wiki-gigaword-100")

# Function to get GloVe embeddings
def get_glove_embeddings(text, model, size):
    words = text.split()
    vec = np.zeros(size)
    count = 0
    for word in words:
        if word in model:
            vec += model[word]
            count += 1
    return vec / count if count > 0 else vec

# Convert train and test sets
X_train_glove = np.array([get_glove_embeddings(text, glove_vectors, 100) for text in X_train])
X_test_glove = np.array([get_glove_embeddings(text, glove_vectors, 100) for text in X_test])



# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings
def get_bert_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Convert train and test sets
X_train_bert = np.array([get_bert_embeddings(text, tokenizer, bert_model) for text in X_train])
X_test_bert = np.array([get_bert_embeddings(text, tokenizer, bert_model) for text in X_test])
