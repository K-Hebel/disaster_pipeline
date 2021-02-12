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
    ''' Loads SQLite datebase and separates feature and target database
        X = dataframe of Messages
        y = dataframe of 36 message categories wtih binary data
        category_names = y column names

        return X, y and category_names
    '''

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
    ''' Process each message by :
        1- Normalizing it (lower case and strip punctuation and stopwords)
        2- Creating tokens
        3- Lemmatizing each word in the clean_tokens

        return clean_tokens
    '''
    #normalize text: remove punctuation and make lower case
    text= re.sub(r'[^a-zA-Z0-9]',' ', text.lower())
    tokens = word_tokenize(text)

    #remove stopwords and lemmatize tokens
    lemmatizer=WordNetLemmatizer()

    clean_tokens =[lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]

    return clean_tokens


def build_model():
    ''' Build machine learning pipeline using the MultiOutputClassifier given the list of target variables/
        category names

        return model
    '''
    #Instansiate the ML pipeline
    model = Pipeline([('vect', CountVectorizer(tokenizer=tokenize, max_df=0.5, ngram_range=(1,2))),
                    ('tfidf', TfidfTransformer(smooth_idf=False)),
                    ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    return model

def evaluate_model(model, X_test, y_test, category_names):
    ''' Evalute the model and optimize model parameters using GridSearchCV
        print classification report for each category
        print GridSearch best_params_
    '''

    from sklearn.model_selection  import GridSearchCV

    #Predict with ML pipeline and print classification report
    y_pred= model.predict(X_test)
    for i in range(len(category_names)):
      print(category_names[i])
      print(classification_report(y_test[category_names[i]], y_pred[:, i]))

    param_grid= {
        'vect__ngram_range': [(1,1),(1,2)]
    }

    cv=GridSearchCV(pipeline, param_grid, cv=2)
    cv.fit(X_train,y_train)
    print('GridSearch Best Parameter:  ', cv.best_params_)


def save_model(model, model_filepath):
    ''' Save model as a pickle file
    '''

    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    ''' Main program from which all functions are implemented
    '''
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
