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
from feature_pipeline import tesla_fg
from feature_pipeline import news_sentiment_fg

# %%
from dotenv import load_dotenv
import os

load_dotenv()

# %%
api_key = os.environ.get('hopsworks_api')
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()

# %%
def create_stocks_feature_view(fs, version):

    # Loading in the feature groups
    tesla_fg = fs.get_feature_group('tesla_stock', version=1)
    news_sentiment_fg = fs.get_feature_group('news_sentiment_updated', version=1)

    # Define the query
    ds_query = tesla_fg.select(['date', 'open', 'ticker'])\
        .join(news_sentiment_fg.select(['sentiment']))

    # Create the feature view
    feature_view = fs.create_feature_view(
        name='tesla_stocks_fv',
        query=ds_query,
        labels=['open']
    )

    return feature_view, tesla_fg

# %%
try:
    feature_view = fs.get_feature_view("tesla_stocks_fv", version=1)
    tesla_fg = fs.get_feature_group('tesla_stock', version=1)
except:
    feature_view, tesla_fg = create_stocks_feature_view(fs, 1)

# %%
def fix_data_from_feature_view(df,start_date,end_date):
    df = df.sort_values("date")
    df = df.reset_index()
    df = df.drop(columns=["index"])

    # Create a boolean mask for rows that fall within the date range
    mask = (pd.to_datetime(df['date']) >= pd.to_datetime(start_date)) & (pd.to_datetime(df['date']) <= pd.to_datetime(end_date))
    len_df = np.shape(df)
    df = df[mask] # Use the boolean mask to filter the DataFrame
    print('From shape {} to {} after cropping to given date range: {} to {}'.format(len_df,np.shape(df),start_date,end_date))

    # Get rid off all non-business days
    isBusinessDay, is_open = extract_business_day(start_date,end_date)
    is_open = [not i for i in is_open] # Invert the mask to be able to drop all non-buisiness days

    filtered_df = df.drop(df[is_open].index) # Use the mask to filter the rows of the DataFrame
    print('From shape {} to {} after removing non-business days'.format(np.shape(df),np.shape(filtered_df)))
    print(filtered_df)
    
    return filtered_df

# %%
#def create_stocks_feature_view(fs, version):

    #Loading in the feature groups
#    tesla_fg = fs.get_feature_group('tesla_stock', version = 3)
#    news_sentiment_fg = fs.get_feature_group('news_sentiment_updated', version = 2)

#    ds_query = tesla_fg.select(['date','open', 'ticker'])\
#        .join(news_sentiment_fg.select_except(['ticker','time', 'amp_url', 'image_url']))
    
#    return (fs.create_tesla_feature_view(
#        name = 'tsla_stocks_fv',
#        query = ds_query,
#        labels=['ticker']
#    ), tesla_fg)

# %%
#try:
#    feature_view = fs.get_feature_view("tsla_stocks_fv", version=1)
#    tesla_fg = fs.get_feature_group('tesla_stock', version=3)
#except:
#    feature_view, tesla_fg = create_stocks_feature_view(fs, 1)


