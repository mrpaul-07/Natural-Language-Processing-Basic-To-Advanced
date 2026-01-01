# ===================================
# ১. Lowercasing (সব ছোট হাতের করা)
# ===================================

text = "Welcome To NLP Learning"
lower_text = text.lower()
print(lower_text)

# =============================================
# ২. Punctuation Removal (দাড়ি-কমা বাদ দেওয়া)
# =============================================

import string
text = "Weclome! To NLP Learing....."
lower_text = text.lower()
final_text = lower_text.translate(str.maketrans('', '', string.punctuation))
print(final_text)

# ===================================
# Tokenization (শব্দ আলাদা করা)
# ===================================

text = "Welcome To NLP Learning."
final = text.translate(str.maketrans('', '', string.punctuation))
token = final.split()
print(token)


# ===================================
# Stopwords Removal
# ===================================

import nltk
from nltk.corpus import stopwords

text = "Better When you land on a sample web page or open an email template and see content beginning with lorem ipsum, the page creator placed that apparent gibberish there on purpose."
final = text.translate(str.maketrans('', '', string.punctuation))
token = final.split()
stopwords = set(stopwords.words('english'))

clean_text = []
for w in token:
    if w not in stopwords:
        clean_text.append(w)

print('Original Words: ',token)
print('Cleaned Words: ',clean_text,'\n')


# ===================================
# Stemming Code
# ===================================

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

print('\n\nStemming Results:')
for w in clean_text:
    root = stemmer.stem(w)
    print(w, '--->', root)


# ===================================
# Lemmatization Code
# ===================================

from nltk.stem import WordNetLemmatizer

lemma = WordNetLemmatizer()

print('\nLemmatization Result:')
for w in clean_text:
    root = lemma.lemmatize(w)
    print(w, '--->', root)