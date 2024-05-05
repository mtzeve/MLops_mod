# %%
from dotenv import load_dotenv
import os 

# %%
#!pip install great_expectations==0.18.12

# %%
# Import necessary libraries
import pandas as pd               # For data manipulation using DataFrames
import numpy as np                # For numerical operations
import matplotlib.pyplot as plt   # For data visualization
import os                         # For operating system-related tasks
import joblib                     # For saving and loading models
import hopsworks                  # For getting access to hopsworks



# Import specific modules from scikit-learn
from sklearn.preprocessing import StandardScaler, OneHotEncoder   # For data preprocessing
from sklearn.metrics import accuracy_score                        # For evaluating model accuracy

# %%
#from alpha_vantage.timeseries import TimeSeries
#import pandas as pd

#load_dotenv()

#api_key = os.environ.get('stocks_api') # Replace this with your actual API key
#ts = TimeSeries(key=api_key, output_format='pandas')

# Fetch daily adjusted stock prices; adjust the symbol as needed
#data, meta_data = ts.get_daily(symbol='TSLA', outputsize='full')

#print(data.head())

# %%
#data.info()

# %%
#meta_data

# %%
# Define your file path and name
#file_path = 'TSLA_stock_price.csv'  # Customize the path and filename

# Save the DataFrame to CSV
#stock_data.to_csv(file_path)

#print(f"Data saved to {file_path}")


# %%
# Load and display the data from CSV to confirm
tsla_df = pd.read_csv('TSLA_stock_price.csv')
print(tsla_df.head())
    

# %%
api_key = os.environ.get('hopsworks_api')
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()

# %%
import re 

# %%
def clean_column_name(name):
    # Remove all non-letter characters
    cleaned_name = re.sub(r'[^a-zA-Z]', '', name)
    return cleaned_name


# %%
tsla_df

# %%
# Assuming 'tsla_df' is your DataFrame
tsla_df.columns = [clean_column_name(col) for col in tsla_df.columns]


# %%
print(tsla_df.columns)


# %%
import pandas as pd

# Assuming tsla_df is your pandas DataFrame
# Convert the "date" column to timestamp
tsla_df['date'] = pd.to_datetime(tsla_df['date'])


# %%
# Define a feature group
tesla_fg = fs.get_or_create_feature_group(
    name="tesla_stock",
    description="Tesla stock dataset from alpha vantage",
    version=3,
    primary_key=["ticker"],
    event_time=['date'],
    online_enabled=False,
)

# %%
tesla_fg.insert(tsla_df, write_options={"wait_for_job" : False})

# %%
news_df = pd.read_csv('news_articles_ema.csv')


# %%
news_df_updated = news_df.drop(columns=['exp_mean_7_days'])

# %%
news_df_updated['date'] = pd.to_datetime(news_df_updated['date'])

# %%
news_sentiment_fg = fs.get_or_create_feature_group(
    name='news_sentiment_updated',
    description='News sentiment from Polygon',
    version=2,
    primary_key=['ticker'],
    event_time=['date'],
    online_enabled=False,
)

# %%
news_sentiment_fg.insert(news_df_updated)


