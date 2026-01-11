"""
NLP Text Processing
Author: Shengjie Wu
Description:
This script performs a complete NLP preprocessing pipeline on a text file:
1. Text cleaning and normalization
2. Tokenization and stopword removal using NLTK
3. Word frequency analysis (Top-10 words)
4. Performance comparison of tokenization frameworks:
   - NLTK
   - TextBlob
   - spaCy
"""

# =====================================================
# 1. Installation & Resource Download (RUN ONCE)
# =====================================================
# The following commands should be executed in the terminal
# before running this script:
#
# pip install nltk textblob spacy
# python -m spacy download en_core_web_sm

import nltk

# Download required NLTK resources (only needed the first time)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


# =====================================================
# 2. Data Verification
# =====================================================
# Check current working directory and available files
import os

print("Current working directory:")
print(os.getcwd())

print("\nFiles in the current directory:")
print(os.listdir())


# =====================================================
# 3. Read Input Text
# =====================================================
# Read the raw text file (Alice in Wonderland)
with open('alice29.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# =====================================================
# 4. Text Cleaning and Normalization
# =====================================================
import re

# Convert all characters to lowercase
text = text.lower()

# Remove all non-alphabetic characters
# Keep whitespace (\s) to preserve word boundaries
text = re.sub(r'[^a-z\s]', ' ', text)

# Replace multiple spaces and line breaks with a single space
text = re.sub(r'\s+', ' ', text).strip()

# Save the cleaned text to file
with open('cleaned.txt', 'w', encoding='utf-8') as f:
    f.write(text)


# =====================================================
# 5. Tokenization and Word Frequency Analysis
# =====================================================
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Load English stopwords
stop_words = set(stopwords.words('english'))

# Tokenize the cleaned text
words = word_tokenize(text)

# Remove stopwords from token list
words = [w for w in words if w not in stop_words]

# Save all tokens (one word per line)
with open('words.txt', 'w', encoding='utf-8') as f:
    for w in words:
        f.write(w + '\n')

# Compute word frequencies
freq = Counter(words)

# Extract the top 10 most frequent words
top10 = freq.most_common(10)

# Save top-10 frequent words
with open('top10words.txt', 'w', encoding='utf-8') as f:
    for w, c in top10:
        f.write(f"{w}: {c}\n")

print("\nTop 10 most frequent words:")
print(top10)


# =====================================================
# 6. Tokenization Performance Comparison
# =====================================================
import timeit
import statistics
from textblob import TextBlob
import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

RUNS = 10  # Number of benchmark repetitions


def nltk_tok():
    """Tokenization using NLTK"""
    word_tokenize(text)


def tb_tok():
    """Tokenization using TextBlob"""
    TextBlob(text).words


def spacy_tok():
    """Tokenization using spaCy"""
    nlp(text)


def bench(fn):
    """
    Benchmark a tokenization function.
    
    Parameters:
        fn (callable): Tokenization function
    
    Returns:
        mean_time (float): Mean execution time (seconds)
        std_time (float): Standard deviation of execution time
    """
    times = timeit.repeat(fn, number=1, repeat=RUNS)
    return statistics.mean(times), statistics.stdev(times)


# Run performance benchmarks
results = {
    'NLTK': bench(nltk_tok),
    'TextBlob': bench(tb_tok),
    'spaCy': bench(spacy_tok)
}

# Save performance comparison results
with open('time_compares.txt', 'w', encoding='utf-8') as f:
    f.write("Framework\tMean(s)\tStd(s)\n")
    for framework, (mean_t, std_t) in results.items():
        f.write(f"{framework}\t{mean_t:.4f}\t{std_t:.4f}\n")

print("\nTokenization performance comparison:")
print(results)
