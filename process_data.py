# # ETL Pipeline Preparation


# ### 1. Import libraries and load datasets.
import pandas as pd
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    """
    Load & merge data sets based on provided file names.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")
    # df_original = df
    return df


def clean_data(df):
    # ### 3. Split `categories` into separate category columns.

    # create a dataframe of the 36 individual category columns
    # categories = categories_original["categories"].str.split(";", expand=True)
    categories = df["categories"].str.split(";", expand=True)

    # Add original id for merging with messages data set
    categories = pd.concat([df["id"], categories], axis=1)

    # Rename columns based on first row
    first_row_list = categories.iloc[0].tolist()
    first_row_list[0] = "id--"
    category_colnames = [cat[:-2] for cat in first_row_list]
    categories.columns = category_colnames

    # ### 4. Convert category values to just numbers 0 or 1.
    for column in categories.iloc[:,1:]:
        # set each value to be the last character of the string & convert to int
        categories[column] = categories[column].str[-1:]
        categories[column] = categories[column].astype("int")

    # ### 5. Replace `categories` column in `df` with new category columns.
    df = df.drop("categories", axis=1)
    df = pd.merge(df, categories, how="inner", on="id")

    # ### 6. Remove duplicates.
    print(f"Initial duplicates in data set: {df.duplicated().sum()}")
    df.drop_duplicates(inplace=True)
    print(f"Duplicates after cleaning: {df.duplicated().sum()}\n")

    # Check & drop NAs in category label columns
    df = df.dropna(subset=["related"])
    print(f"N/As in data set after cleaning: \n{df.isna().sum()} \n")

    # Convert to int
    df.iloc[:,4:] = df.iloc[:,4:].astype("int")

    df = remove_non_binary(df)

    return df


def remove_non_binary(df):
    """
    Ensure that each category column is binary. Rows with values not 0/1 are removed. Columns with fewer than two values are removed.
    """    
    columns_to_drop = []
    print("Non-binary category columns are being cleaned...")
    for col in list(df.columns)[4:]:
#         print(set(df[col].unique()))
        print(col)
        column_values = set(df[col].unique())
        
        if len(column_values) < 2:
            columns_to_drop.append(col)
            print(f"Column '{col}' contains fewer than two unique values and has been dropped.")
            
        elif column_values.issubset(set([0, 1])) == False:
            print(f"Column '{col}' contains the following values: {df[col].unique()}. \n Rows containing values other than 0 & 1 are removed.")
            # Remove values that are not 0 or 1
            df = df.loc[df["related"].isin([0,1]) == True]
        
    df = df.drop(columns_to_drop, axis = 1)
    print(f"\n")  
    return df



def save_data(df, database_filename):
    """
    Save cleaned data frame as SQLite database file.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('Messages_Categories', engine, index=False, if_exists="replace")  


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
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()



print("ETL pipeline run completed.")