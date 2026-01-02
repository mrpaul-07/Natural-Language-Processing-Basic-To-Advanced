import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

corpus = [
    "Welcome To NLP Tutorial",
    "Welcome To Python Tutorial",
    "I Love Python & NLP",
    "Happy Learning Tutorial"
]

# ==============================
# Bag of Words (BoW)
# ==============================

bow = CountVectorizer()
bow_matrix = bow.fit_transform(corpus)

print('Bow Results: ')
print(pd.DataFrame(bow_matrix.toarray(), columns=bow.get_feature_names_out()))
print()

# ==============================
# Bigrams (N-grams)
# ==============================

bigram =  CountVectorizer(ngram_range=(2,2))
bigram_matrix = bigram.fit_transform(corpus)

print('Bigram Results: ')
print(pd.DataFrame(bigram_matrix.toarray(), columns=bigram.get_feature_names_out()))
print()
# ==============================
# TF-IDF
# ==============================

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(corpus)

print('TF-IDF Results: ')
print(pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out()))