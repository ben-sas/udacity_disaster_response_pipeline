# # ETL Pipeline Preparation
# ### 1. Import libraries and load datasets.

import pandas as pd
from sqlalchemy import create_engine
import sys

# Load data sets

## Code with input prompt via terminal
# categories_file = input("Enter filename of categories data: ")
# messages_file = input("Enter filename of messages data: ")
# messages = pd.read_csv(messages_file)
# categories_original = pd.read_csv(categories_file)

## Version with direct input when script is started
messages = pd.read_csv(sys.argv[1])
categories_original = pd.read_csv(sys.argv[2])


# ### 2. Merge datasets.
df = pd.merge(messages, categories_original, on="id")
df_original = df

# ### 3. Split `categories` into separate category columns.

# create a dataframe of the 36 individual category columns
categories = categories_original["categories"].str.split(";", expand=True)
# Add original id for merging with messages data set
categories = pd.concat([categories_original["id"], categories], axis=1)

# Rename columns based on first row
# select the first row of the categories dataframe
first_row_list = categories.iloc[0].tolist()
first_row_list[0] = "id--"

category_colnames = [cat[:-2] for cat in first_row_list]
categories.columns = category_colnames

# ### 4. Convert category values to just numbers 0 or 1.

for column in categories.iloc[:,1:]:
    # set each value to be the last character of the string
    categories[column] = categories[column].str[-1:]
    
    # convert column from string to numeric
    categories[column] = categories[column].astype("int")

# ### 5. Replace `categories` column in `df` with new category columns.

df = df.drop("categories", axis=1)
df = pd.merge(df, categories, how="inner", on="id")


# ### 6. Remove duplicates.
# - Check how many duplicates are in this dataset.
# - Drop the duplicates.
# - Confirm duplicates were removed.

# check number of duplicates
df.duplicated().sum()

# drop duplicates
df.drop_duplicates(inplace=True)

# %%
# Check & drop NAs in category label columns
df = df.dropna(subset=["related"])
df.isna().sum()

# Convert to int
df.iloc[:,4:] = df.iloc[:,4:].astype("int")

# Check that values are binary & remove non-binary entries (rows)
categories_colnames = list(categories.columns)

def remove_non_binary(df):
    
    columns_to_drop = []
    print("Non-binary category columns are being cleaned...")
    for col in categories_colnames[1:]:
#         print(set(df[col].unique()))
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

df = remove_non_binary(df)

# ### 7. Save the clean dataset into an sqlite database.

engine = create_engine('sqlite:///UdacityDisasterResponse.db')
df.to_sql('Messages_Categories', engine, index=False, if_exists="replace")

print("ETL pipeline run completed.")