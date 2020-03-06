import flask
import os
import requests
from flask import request, jsonify, render_template, sessions, make_response
from nltk.corpus import stopwords
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import csv
import NLP as nlp
import pandas as pd
from textblob import TextBlob

# import flask_restful import Resource
import re
from bs4 import BeautifulSoup
# nltk.download('stopwords')
# nltk.download('vader_lexicon')

app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.config["JSON_SORT_KEYS"] = False
vader = SentimentIntensityAnalyzer()

# def dict_factory(cursor, row):
#     d = {}
#     for idx, col in enumerate(cursor.description):
#         d[col[0]] = row[idx]
#     return d

@app.route('/', methods=['GET', 'POST'])
def index():
    errors = []
    results = {}
    if request.method == "POST":
        print("method")
        # get text that the user has entered
        try:
            if 'tab' in request.form:
                processType = request.form['tab']
                if processType == 'single':
                    text = request.form['text']
                elif processType == 'batch':
                    file = request.files['file']
                else:
                    errors.append(
                        "Unable to get entered text. Please make sure it's valid and try again."
                    )
            if 'output' in request.form:
                output = request.form['output']
            else:
                errors.append(
                        "Unable to generate output. Please make sure it's valid and try again."
                    )
        except:
            errors.append(
                "Unable to get entered text. Please make sure it's valid and try again."
            )
            return render_template('index.html', errors=errors)
        if processType == 'single':

            # Data Cleansing
            # clean_text = text.apply(lambda x: ' '.join([word.lower() for word in x.split()]))
            clean_text = text.lower()
            cachedStopWords = stopwords.words("english")

            # clean_text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
            clean_text = ' '.join([word for word in clean_text.split() if word not in cachedStopWords])
            # regURL = re.compile(r"http.?://[^\s]+[\s]?")
            # clean_text = clean_text.replace(regURL, "")
            # Sentiment

            polarity = TextBlob(clean_text).sentiment.polarity
            subjectivity = TextBlob(clean_text).subjectivity
            # sentiment = TextBlob(clean_text).sentiment.polarity
            if polarity > 0:
                sentiment = 'Positive'
            elif polarity < 0:
                sentiment = 'Negative'
            elif polarity == 0:
                sentiment = 'Neutral'

            # From VADER (Valence Aware Dictionary And sEntiment Reasoner)
            scores = vader.polarity_scores(clean_text)
            results = { 'Text Entered' : text,
                        'Sentiment' : sentiment,
                        'Polarity': polarity,
                        'Subjectivity' : subjectivity,
                        'SentimentScore' : scores}
            return jsonify(results)

        if processType == 'batch':
            df = pd.read_csv(request.files.get('file'))
            new_df = nlp.caseChange(df, column= 'Text')
            new_df = nlp.sentiment(new_df, column='clean_text')

            # Drop the clean_text column
            new_df = new_df.drop(['clean_text'], axis= 1)

            if output == 'json':
                results = new_df.set_index('ID').T.to_dict('list')
                return jsonify(results)
            elif output == 'csv':
                # Downloan CSV File
                resp = make_response(new_df.to_csv(index = False))
                resp.headers["Content-Disposition"] = "attachment; filename=export.csv"
                resp.headers["Content-Type"] = "text/csv"
                return resp
    return render_template('index.html', errors = errors, results = results)

if __name__ == '__main__':
    app.run()

