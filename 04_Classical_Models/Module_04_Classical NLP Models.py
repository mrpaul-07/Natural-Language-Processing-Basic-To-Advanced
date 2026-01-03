import pandas as pd
import string
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "dataset.csv")
df = pd.read_csv(file_path)

# print(df.info())
# print(df.head(5))
# print(df.isnull().sum()) [No NUll Values Here]
duplicate = df[df.duplicated()]
# print(duplicate.count()) [Existing Duplicated Values]
df.drop_duplicates(inplace = True)
duplicate = df[df.duplicated()]
# print(duplicate.count()) [No Duplicated Values]
# print(df['category'].unique())
category_map = {
    'not-spam':0, 'spam':1
}
df['category'] = df['category'].map(category_map)

x = df['email']
y= df['category']

def clean_texts(text):
    text = text.lower()
    f_text = text.translate(str.maketrans('', '', string.punctuation))
    return f_text

X = x.apply(clean_texts)
Y = y
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    Y, 
    test_size=0.2, 
    random_state=42
)

tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)


# --------------------
# Naive Bayes
# --------------------
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)

# ----------------------
# Logistic Regression
# ----------------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# -------------------------
# Support Vector Machine
# -------------------------
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

# -----------
# Evaluate
# -----------
def evaluate(y_test, y_pred):
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred)
    }

# Train prediction
lr_train_pred = lr.predict(X_train)
lr_test_pred  = lr.predict(X_test)
svm_train_pred = svm.predict(X_train)
svm_test_pred  = svm.predict(X_test)
nb_train_pred = nb.predict(X_train)
nb_test_pred  = nb.predict(X_test)
# Test prediction
lr_train_metrics = evaluate(y_train, lr_train_pred)
lr_test_metrics  = evaluate(y_test, lr_test_pred)
svm_train_metrics = evaluate(y_train, svm_train_pred)
svm_test_metrics  = evaluate(y_test, svm_test_pred)
nb_train_metrics = evaluate(y_train, nb_train_pred)
nb_test_metrics  = evaluate(y_test, nb_test_pred)

# -------------
# Matrix's
# -------------
results = pd.DataFrame({
    "Model": [
        "Logistic Regression (Train)", "Logistic Regression (Test)",
        "SVM (Train)", "SVM (Test)",
        "Naive Bayes (Train)", "Naive Bayes (Test)"
    ],
    "Accuracy": [
        lr_train_metrics["Accuracy"], lr_test_metrics["Accuracy"],
        svm_train_metrics["Accuracy"], svm_test_metrics["Accuracy"],
        nb_train_metrics["Accuracy"], nb_test_metrics["Accuracy"]
    ],
    "Precision": [
        lr_train_metrics["Precision"], lr_test_metrics["Precision"],
        svm_train_metrics["Precision"], svm_test_metrics["Precision"],
        nb_train_metrics["Precision"], nb_test_metrics["Precision"]
    ],
    "Recall": [
        lr_train_metrics["Recall"], lr_test_metrics["Recall"],
        svm_train_metrics["Recall"], svm_test_metrics["Recall"],
        nb_train_metrics["Recall"], nb_test_metrics["Recall"]
    ],
    "F1-Score": [
        lr_train_metrics["F1"], lr_test_metrics["F1"],
        svm_train_metrics["F1"], svm_test_metrics["F1"],
        nb_train_metrics["F1"], nb_test_metrics["F1"]
    ]
})

print(results.round(4))