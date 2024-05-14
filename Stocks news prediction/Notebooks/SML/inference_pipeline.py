# %%
import pandas as pd 
import hopsworks 
from datetime import datetime, timedelta
from SML.training_pipeline_new import model_dir
import numpy as np



#Making the notebook able to fetch from the .env file
from dotenv import load_dotenv
import os

load_dotenv()

# %%
api_key = os.environ.get('hopsworks_api')
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()
mr = project.get_model_registry() 

# %%
start_date = datetime.now() - timedelta(hours=48)
print(start_date.strftime("%Y-%m-%d"))

# %%
end_date = datetime.now() - timedelta(hours=24)
print(end_date.strftime("%Y-%m-%d"))

# %%
feature_view = fs.get_feature_view('tesla_stocks_fv', 5)
feature_view.init_batch_scoring(training_dataset_version=1)

# %%
print(feature_view.get_batch_query())

# %%
# we had problems fetching the data from fv with get_batch_data function, tried everything and it just doesnt work 
tesla_df_b = feature_view.get_batch_data(start_time=start_date, end_time=end_date)

# %%
tesla_df_b.head()

# %%
from sklearn.preprocessing import OneHotEncoder

# %%
#OneHotEncoding the tesla_df_b column 'ticker'

tickers = tesla_df_b[['ticker']]

# Initializing OneHotEncoder
encoder = OneHotEncoder()

# Fitting and transforming the 'ticker' column
ticker_encoded_test = encoder.fit_transform(tickers)

# Converting the encoded column into a DataFrame
ticker_encoded_df_test = pd.DataFrame(ticker_encoded_test.toarray(), columns=encoder.get_feature_names_out(['ticker']))

# Concatenating the encoded DataFrame with the original DataFrame
tesla_df_b = pd.concat([tesla_df_b, ticker_encoded_df_test], axis=1)

# Dropping the original 'ticker' column
tesla_df_b.drop('ticker', axis=1, inplace=True)

# %%
# As X_train['date'] column exists and is in datetime format, we're converting it
tesla_df_b['year'] = tesla_df_b['date'].dt.year
tesla_df_b['month'] = tesla_df_b['date'].dt.month
tesla_df_b['day'] = tesla_df_b['date'].dt.day

# Dropping the original date column
tesla_df_b.drop(columns=['date'], inplace=True)

# Converting dataframe to numpy array
tesla_df_b_array = tesla_df_b.to_numpy()

# Reshaping the array to have a shape suitable for LSTM
tesla_df_b_array = np.expand_dims(tesla_df_b_array, axis=1)

# %%
import joblib

the_model = mr.get_model("stock_pred_model", version=28)
model_dir = the_model.download()

model = joblib.load(model_dir + "/stock_prediction_model.pkl")

# %%
predictions = model.predict(tesla_df_b_array)

# %%
predictions 

# %%
import numpy as np

# Our predictions array
predictions = np.array(predictions, dtype=np.float32)

# Changing the format of the predicted value to correspond with format of "open"
predictions = predictions[0][0]*100
print(predictions)


# %%
tesla_df_b['predictions'] = predictions.tolist()

# %%
# Assuming you have 'year', 'month', and 'day' columns in your DataFrame
tesla_df_b['date'] = pd.to_datetime(tesla_df_b[['year', 'month', 'day']])

# Now you can drop the 'year', 'month', and 'day' columns if you want
tesla_df_b.drop(columns=['year', 'month', 'day'], inplace=True)

# %%
tesla_df_b['date'] = pd.to_datetime(tesla_df_b['date'])

# %%
tesla_df_b

# %%
# Convert the encoded DataFrame back to numpy array
ticker_encoded_array = ticker_encoded_df_test.to_numpy()

# Inverse transform the encoded array to retrieve the original values
original_tickers = encoder.inverse_transform(ticker_encoded_array)

# Convert the original_tickers array to a DataFrame
original_tickers_df = pd.DataFrame(original_tickers, columns=['ticker'])

# Concatenate the original ticker column with the remaining columns from tesla_df_b
tesla_df_b = pd.concat([tesla_df_b.drop(columns=['ticker_TSLA']), original_tickers_df], axis=1)


# %%
tesla_df_b.head()

# %%
#from sklearn.preprocessing import MinMaxScaler

# Flatten the list of lists into a single list
#flat_predictions_scaled = [item for sublist in predictions_scaled for item in sublist]

# Initialize the MinMaxScaler
#scaler = MinMaxScaler()

# Fit the scaler to the scaled predictions
#scaler.fit(flat_predictions_scaled)

# Inverse transform the scaled predictions to get the original values
#predictions_unscaled = scaler.inverse_transform(flat_predictions_scaled)

# Update the 'predictions' column with the unscaled values
#tesla_df_b['predictions'] = predictions_unscaled

# %%
api_key = os.environ.get('hopsworks_api')
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()

# %%
results_fg = fs.get_or_create_feature_group(
    name= 'stock_prediction_results',
    version = 4,
    description = 'Predction of TSLA open stock price',
    primary_key = ['ticker'],
    event_time = ['date'],
    online_enabled = False,
)

# %%
#Inserting the stock data into the stocks feature group
results_fg.insert(tesla_df_b, write_options={"wait_for_job" : False})


