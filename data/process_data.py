import sys
import pandas as pd
import numpy as np
from sqlite3 as sql

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories =pd.read_csv(categories_filepath)
    # Merge datasets
    df = messages.merge(categories, on='id')

    return df

def clean_data(df):
    # Create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';',expand=True)
    categories.columns=categories.iloc[0].str.strip('- 10')

    # Set each value to be the last character of the string
    for column in categories.columns:
        categories[column]= categories[column].str[-1].astype('int32')

    # Drop the 'categories' column from `df`
    # Drop "original" and 'genre' columns as they are not part of the
    # analysis
    df=df.drop(['categories','original', 'genre'] , axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df=df.drop_duplicates()

    # Replace the '2' value in the 'related' column with '1'
    # for 'related' column, replace the 2 value with 1
    df['related'] =df['related'].replace(2,1)

    return df

def save_data(df, database_filepath):
    # Save clean data to sqlite database
    conn = sql.connect(database_filepath)
    df.to_sql(df, conn, if_exists='replace', index= False)




def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
