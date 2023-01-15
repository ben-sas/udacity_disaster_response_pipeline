
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
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

import string

from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

import sys


def load_data(database_filepath):
    """ Load data from database and return the messages and categories as separate data frames."""
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("Messages_Categories", con=engine)

    # Optional: Limit number of included rows for testing purposes
    # df = df[:500]

    X = df["message"]
    Y = df.iloc[:,4:].astype("int")
    category_names = list(Y.columns)

    return X, Y, category_names


def tokenize_single(text):
    """Tokenize text into sentences and then single words. Stopwords are also removed."""
    message = text.lower()
    
    # Remove punctuation
    # https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
    message = message.translate(str.maketrans('', '', string.punctuation))
        
    # Tokenize into sentences and then words
    sentence_list = sent_tokenize(message)
        
    try:
        words_tokenized = [word_tokenize(sentence) for sentence in sentence_list][0]
    except:
        # Some strings lead to errors and are replaced with empty strings (e.g. many whitespaces without meaning)
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
    

def build_model():
    """Build the model pipeline."""
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=lemmatize_single)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(SGDClassifier())),
        ])
    
    # Note, including more parameters significantly increases the time needed to train the model, therefore a small selection has been made.
    gridsearch_parameters = {
        'clf__estimator__loss': ["hinge", "log_loss", "log", "modified_huber"] #, "squared_hinge", "perceptron", "squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
        "clf__estimator__alpha" : [0.00005, 0.0001, 0.0005], #[0.00001, 0.00005, 0.0001, 0.0005, 0.001]
        # "clf__estimator__max_iter" : [500, 1000], #[500, 1000, 2000]
        # "clf__estimator__learning_rate" : ["optimal", "adaptive"], #["constant", "optimal", "adaptive"]
        # "clf__estimator__class_weight" : [None, "balanced"]
    }

    grid_search_pipeline = GridSearchCV(pipeline, param_grid=gridsearch_parameters)

    return grid_search_pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the model by predicting the categories and printing the full classification report and hamming loss."""
    Y_predicted = model.predict(X_test)

    # model_classification_report = classification_report(Y_test, Y_predicted, target_names=category_names)
    # model_hamming_loss = hamming_loss(Y_test, Y_predicted)
    print(classification_report(Y_test, Y_predicted, target_names=category_names))
    print(f"Hamming loss: {hamming_loss(Y_test, Y_predicted)}")

    # grid_search_results = pd.DataFrame(model.cv_results_)
    # grid_search_results

    # model_classification_report.to_csv("model_classification_report.csv")


def save_model(model, model_filepath):
    """Save the model as a pickle file."""
    file_model = open(model_filepath, "wb")
    pickle.dump(model, file_model)
    file_model.close()


def main():
    """Run the whole pipeline."""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print(f"ML pipeline completed. Model exported as picke file '{model_filepath}'")

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()


