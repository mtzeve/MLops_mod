# %%
from dotenv import load_dotenv
import os 

# %%
from alpha_vantage.timeseries import TimeSeries
import pandas as pd

load_dotenv()

api_key = os.environ.get('stocks_api') # Replace this with your actual API key
ts = TimeSeries(key=api_key, output_format='pandas')

# Fetch daily adjusted stock prices; adjust the symbol as needed
data, meta_data = ts.get_daily(symbol='TSLA', outputsize='full')

print(data.head())

# %%


# %%
data.info()

# %%
meta_data

# %%
# Define your file path and name
file_path = '/Users/manos/Documents/BDS/MLops_mod/TSLA_stock_price.csv'  # Customize the path and filename

# Save the DataFrame to CSV
data.to_csv(file_path)

print(f"Data saved to {file_path}")


# %%
# Load and display the data from CSV to confirm
tsla_df = pd.read_csv(file_path)
print(tsla_df.head())


# %%
import hopsworks

project = hopsworks.login()
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
# Define a feature group
tesla_fg = fs.get_or_create_feature_group(
    name="tsla_stock",
    description="Tesla stock dataset from alpha vantage",
    version=1,
    primary_key=["date"],
    online_enabled=True,
)

# %%
tesla_fg.insert(tsla_df, write_options={"wait_for_job" : False})

# %%
# Create feature group for historical news data
news_df = pd.read_csv('/Users/manos/Documents/BDS/MLops_mod/news_articles.csv')

news_sentiment_fg = fs.get_or_create_feature_group(
    name='news_sentiment',
    description='News sentiment from Polygon',
    version=1,
    primary_key=['date'],
    online_enabled=True,
)

news_sentiment_fg.insert(news_df)


