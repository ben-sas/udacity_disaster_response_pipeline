
# # ML Pipeline Preparation

# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle
import sys

from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import string

from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import hamming_loss

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import classification_report

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# load data from database
database_filepath = sys.argv[1]
engine = create_engine(f'sqlite:///{database_filepath}')

df = pd.read_sql_table("Messages_Categories", con=engine)

# Optional: Limit number of included rows for testing purposes
df = df[:500]

X = df["message"]
Y = df.iloc[:,4:].astype("int")


# ### 2. Write a tokenization function to process your text data

def tokenize_single(text):
    """
    Tokenize text into sentences and then single words. Stopwords are also removed.
    """
    
    message = text.lower()
    
    # Remove punctuation
    # https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
    message = message.translate(str.maketrans('', '', string.punctuation))
        
    # Tokenize into sentences and then words
    sentence_list = sent_tokenize(message)
        
    try:
        words_tokenized = [word_tokenize(sentence) for sentence in sentence_list][0]
    except:
        words_tokenized = [""]
        print(Exception)
        print(f"Text '{text}' replaced with empty string.")

    # Remove stopwords
    words_tokenized = [w for w in words_tokenized if w not in stopwords.words("english")]
    
    
    return words_tokenized


def lemmatize_single(text):
    """
    This function takes a piece of text, tokenizes by words via the tokenize() function and then lemmatizes each token.
    The output is a list of lemmatized tokens per corpus.
    """
    # tokenize text
    message_tokenized = tokenize_single(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    words_lemmatized = [lemmatizer.lemmatize(token) for token in message_tokenized]
    
    # Apply porter stemmer
    ps = PorterStemmer()
    words_stemmed = [ps.stem(word) for word in words_lemmatized]
    
    return words_stemmed


# ### 3. Build a machine learning pipeline

categories_colnames = list(Y.columns)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)


pipeline_modular = Pipeline([
        ('vect', CountVectorizer(tokenizer=lemmatize_single)),
#         ('tfidf', TfidfVectorizer(lowercase=False)),
        ('tfidf', TfidfTransformer()),
        # ('clf', MultiOutputClassifier(RandomForestClassifier())),
        # ('clf', MultiOutputClassifier(MultinomialNB(fit_prior=True, class_prior=None))),
        ('clf', MultiOutputClassifier(SGDClassifier())),
        # ('clf', MLPClassifier(random_state=1, max_iter=300)),
    ])


# ### 4. Train pipeline
pipeline_modular.fit(X_train, Y_train)
prediction_modular = pipeline_modular.predict(X_test)

# ### 5. Test your model
print(classification_report(Y_test, prediction_modular, target_names=categories_colnames))
print(f"Hamming loss: {hamming_loss(Y_test, prediction_modular)}")


# ### 9. Export your model as a pickle file

# filename_model = "ML_model.sav"
filename_model = sys.argv[2] #"models/ML_model.pkl"


file_model = open(filename_model, "wb")
pickle.dump(pipeline_modular, file_model)
file_model.close()

print(f"ML pipeline completed. Model exported as picke file '{filename_model}'")