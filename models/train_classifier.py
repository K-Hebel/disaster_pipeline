import sys

# import libraries
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

import sqlite3 as sql
from sklearn.multioutput import MultiOutputClassifier
import pickle

import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])


def load_data(database_filepath):
    # load data from database
    conn = sql.connect(database_filepath)
    df = pd.read_sql("SELECT * FROM {}".format( database_filepath[5:-3]),conn)

    #Separate Feature data (X) from target data (y)
    X = df['message']
    y =  df.drop(['message', 'original', 'id','genre'], axis=1)

    #Create variable for target labels
    category_names = y.columns.values

    return X,y,category_names

def tokenize(text):
    #normalize text: remove punctuation and make lower case
    text= re.sub(r'[^a-zA-Z0-9]',' ', text.lower())
    tokens = word_tokenize(text)

    #remove stopwords and lemmatize tokens
    lemmatizer=WordNetLemmatizer()

    clean_tokens =[lemmatizer.lemmatize(word) for word in tokens]

    return clean_tokens


def build_model():
    #Instansiate the ML pipeline
    model = Pipeline([('vect', CountVectorizer(tokenizer=tokenize, stop_words='english', max_df=0.5, ngram_range=(1,1))),
                    ('tfidf', TfidfTransformer(smooth_idf=False)),
                    ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    return model

def evaluate_model(model, X_test, y_test, category_names):
    #Predict with ML pipeline
    y_pred= model.predict(X_test)

    return y_pred

def save_model(model, model_filepath):

    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
