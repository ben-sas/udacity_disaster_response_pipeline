# Udacity Data Science Project: Disaster Response Pipeline

This project is part of the Udacity Data Scientist Nanodegree. It encompasses an ETL pipeline to load and prepare the data, a machine learning pipeline for modeling and a web app with visualizations and interactive classification of user-submitted messages.


# Data

The data is comprised of two separate files:

- *data/messages.csv* contains messages submitted by people in distress after disasters (e.g. storms, fires, earthquakes). All messages are already translated to English
- *data/categories.csv* contains the respective categories for classifying the data (e.g. fire, cold, security)


# Libraries

The following libraries are required to run the code and need to be installed in the virtual environment:

import pandas as pd
    import numpy as np
    from sqlalchemy import create_engine
    import pickle
    import sys
    import nltk
    import string
    import json
    import plotly
    from flask import Flask
    import joblib
    import string


<br>


# Description of files in repository

File Name	  				| File Description
-------------				        | -------------
README.md				    | Readme file.
app/run.py 				    | Web app that uses the trained model.
app/templates/go.html                       | HTML file for the classification results based on user input.
app/templates/master.html                   | Main web app HTML.
data/categories.csv                         | Categories data set.
data/messages.csv                           | Messages data set.
data/DisasterResponse.db                    | SQLite database file (output of the ETL pipeline).
data/grid_search_results.csv                | Results of a grid search for different classification algorithms.
data/process_data.py                        | Script to run the ETL pipeline.
helper_code/ETL_Pipeline.ipynb              | Jupyter Notebook with ETL pipeline for testing.
helper_code/ML_pipeline.ipynb               | Jupyter Notebook with ML pipeline for testing.
models/classifier.pkl                       | Pickle file containing the ML pipeline output for the web app.
models/train_classifier.py                  | ML pipeline script.

<br>


# How to run the pipelines and web app

1. Download the necessary NLTK resources when you run the code for the first time:

        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

2. Run the ETL pipeline and specify input and output files:
        
        python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db

3. Run the ML pipeline and specify input and output files:

        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

4. Start the web app & access the launched application in your browser via the IP address displayed in the command line editor

        cd app
        python run.py
