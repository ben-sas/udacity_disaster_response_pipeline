import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine

import string


app = Flask(__name__)

# def lemmatize_single(text):
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()

#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)

#     return clean_tokens


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
    clean_tokens = [ps.stem(word) for word in words_lemmatized]
    
    return clean_tokens



# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages_Categories', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Create table for top 10 categories visualization
    df_categories = pd.DataFrame(df.iloc[:, 4:].sum()).reset_index()
    df_categories.columns = ["category", "count"]
    df_categories.sort_values(by="count", inplace=True, ascending=False)
    df_categories_top10 = df_categories.head(10)

    # Show number of categories per message
    df_count_cat_per_message = pd.DataFrame(df.iloc[:, 4:].sum(axis=1)).reset_index()
    df_count_cat_per_message.columns = ["num_messages", "num_categories"]
    df_count_cat_per_message = df_count_cat_per_message.groupby("num_categories").count().reset_index()
    df_count_cat_per_message
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=df_categories_top10["category"],
                    y=df_categories_top10["count"]
                )
            ],

            'layout': {
                'title': 'Top 10 Categories in Training Data Set',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=df_count_cat_per_message["num_categories"],
                    y=df_count_cat_per_message["num_messages"]
                )
            ],

            'layout': {
                'title': 'Distribution of Category Labels per Message',
                'yaxis': {
                    'title': "Messages"
                },
                'xaxis': {
                    'title': "Categories per Message"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()