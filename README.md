# Disaster Response Pipeline Project
## Project Summary:
In the event of a large scale emergency or disaster, communicating quickly and efficiently with aid groups, first responders and other agencies is critical. However during these times, it can be very difficult for emergency workers to filter all the information they receive.  This is a web app where emergency workers can input a new message and get classification results in several categories.

This app may subsequently be used to automatically forward messages onward based on the messages' classification tags.  This web app also displays visualizations of the training data used to create the Machine Learning model.



### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Files in Repository:
The following files are contained in this repository:
- **README.md** - contains  a summary of the project and instructions on how to run the web application

- **data** (folder)
    - **disaster_categories.csv** - contains the text classification categories
    - **disaster_messages.csv** - Contains emergency messages and message source ('genre')
    - **process_data.py** - Cleans and merges datasets, then saves to a SQLite3 database
    - **DisasterResponse.db** - saved database

- **models** (folder)
    - **train_classifier.py** - Loads data from the SQLite3 database, builds a text processing and machine learning pipeline,  tunes model using GridSearchCV and saves the final model as a pickle file
    - **classifier.pkl** - saved model    
    
- **app** (folder)
    - **run.py** - This file contains code for the Flash web app as well as the plotly graph visualization seen on the landing page.
    - **templates** (folder - holds html pages for app)
        - **go.html**
        - **master.html**
