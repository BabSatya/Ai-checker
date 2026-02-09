# ===============================
# feature_extractor.py (UPDATED)
# - No heavy model loads at import-time
# - NLTK downloads only if missing (quiet)
# - Lazy-load BERT + GPT-2 on first use
# ===============================

import re
import string
from collections import Counter
from typing import Tuple, Optional

import numpy as np
import nltk
import textstat
import torch
from nltk.corpus import stopwords

# ===============================
# AUTO-DOWNLOAD NLTK RESOURCES (only if missing)
# ===============================
def download_nltk_resources() -> None:
    resources = {
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    }

    for res, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK resource: {res}")
            nltk.download(res, quiet=True)

download_nltk_resources()

# After downloading, this will work
stop_words = set(stopwords.words("english"))

# ===============================
# GLOBAL CONFIG
# ===============================
MAX_TOKENS = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# Lazy-loaded Transformers (to avoid OOM at startup)
# ===============================
BERT_TOKENIZER = None
BERT_MODEL = None
GPT2_TOKENIZER = None
GPT2_MODEL = None

def get_bert():
    """
    Lazy-load BERT on first request.
    """
    global BERT_TOKENIZER, BERT_MODEL
    if BERT_MODEL is None or BERT_TOKENIZER is None:
        from transformers import AutoTokenizer, AutoModel  # import only when needed
        BERT_TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
        BERT_MODEL = AutoModel.from_pretrained("bert-base-uncased").to(device)
        BERT_MODEL.eval()
    return BERT_TOKENIZER, BERT_MODEL

def get_gpt2():
    """
    Lazy-load GPT-2 on first request.
    """
    global GPT2_TOKENIZER, GPT2_MODEL
    if GPT2_MODEL is None or GPT2_TOKENIZER is None:
        from transformers import GPT2Tokenizer, GPT2LMHeadModel  # import only when needed
        GPT2_TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")
        GPT2_MODEL = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        GPT2_MODEL.eval()
    return GPT2_TOKENIZER, GPT2_MODEL

# ===============================
# TEXT CLEANING
# ===============================
def clean_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?]", "", text)
    return text.strip()

# ===============================
# BERT EMBEDDING
# ===============================
def get_bert_embedding(text: str) -> np.ndarray:
    text = clean_text(text)

    if not text:
        return np.zeros(768, dtype=np.float32)

    bert_tokenizer, bert_model = get_bert()
    embeddings = []

    with torch.no_grad():
        # text-based chunking to avoid overflow
        for i in range(0, len(text), 2000):
            chunk_text = text[i:i + 2000]

            inputs = bert_tokenizer(
                chunk_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = bert_model(**inputs)

            cls_embedding = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_embedding.cpu().numpy())

    return np.mean(embeddings, axis=0).squeeze()

# ===============================
# STYLOMETRIC FEATURES
# ===============================
def stylometric_analysis(text: str) -> np.ndarray:
    text = text or ""

    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text.lower())
    words = [w for w in words if w.isalpha()]

    num_words = len(words)
    num_sentences = len(sentences)

    # ---- Sentence length stats ----
    sent_lengths = [len(nltk.word_tokenize(s)) for s in sentences if s.strip()]
    avg_sent_len = float(np.mean(sent_lengths)) if sent_lengths else 0.0
    sent_len_var = float(np.var(sent_lengths)) if len(sent_lengths) > 1 else 0.0

    # ---- Burstiness ----
    burstiness = (sent_len_var / avg_sent_len) if avg_sent_len > 0 else 0.0

    # ---- Stopword ratio ----
    stopword_ratio = (
        sum(1 for w in words if w in stop_words) / num_words
        if num_words > 0 else 0.0
    )

    # ---- Repetition metrics ----
    unigram_rep = (1 - (len(set(words)) / num_words)) if num_words > 0 else 0.0

    bigrams = list(nltk.bigrams(words))
    bigram_counts = Counter(bigrams)
    bigram_rep = (
        sum(1 for c in bigram_counts.values() if c > 1) / len(bigram_counts)
        if bigram_counts else 0.0
    )

    # ---- POS tag distribution ----
    pos_tags = nltk.pos_tag(words)
    pos_counts = Counter(tag for _, tag in pos_tags)

    noun_ratio = (
        sum(pos_counts[t] for t in ["NN", "NNS", "NNP", "NNPS"]) / num_words
        if num_words else 0.0
    )
    verb_ratio = (
        sum(pos_counts[t] for t in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]) / num_words
        if num_words else 0.0
    )
    adj_ratio = (
        sum(pos_counts[t] for t in ["JJ", "JJR", "JJS"]) / num_words
        if num_words else 0.0
    )

    # ---- Punctuation normalization ----
    punct_ratio = sum(1 for c in text if c in string.punctuation) / max(len(text), 1)

    # ---- Readability ----
    readability = float(textstat.flesch_reading_ease(text)) if text.strip() else 0.0

    # ---- Perplexity (GPT-2) ----
    gpt2_tokenizer, gpt2_model = get_gpt2()

    inputs = gpt2_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = gpt2_model(**inputs, labels=inputs["input_ids"])
        perplexity = float(torch.exp(outputs.loss).item())

    return np.array([
        num_words,
        num_sentences,
        avg_sent_len,
        sent_len_var,
        burstiness,
        stopword_ratio,
        unigram_rep,
        bigram_rep,
        noun_ratio,
        verb_ratio,
        adj_ratio,
        punct_ratio,
        readability,
        perplexity
    ], dtype=np.float32)

# ===============================
# FINAL FEATURE VECTOR
# ===============================
def extract_features(text: str) -> np.ndarray:
    """
    FINAL FEATURE VECTOR
    Shape = (768 + 14,) = (782,)
    MUST MATCH TRAINING
    """
    bert_features = get_bert_embedding(text)
    style_features = stylometric_analysis(text)

    return np.concatenate([bert_features, style_features]).astype(np.float32)
