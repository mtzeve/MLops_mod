import streamlit as st
import os
import hopsworks
from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import joblib

def login_hopsworks(api_key):
    project = hopsworks.login(api_key_value=api_key)
    return project

def get_feature_data(fs, start_date, end_date):
    feature_view = fs.get_feature_view('tesla_stocks_fv', 3)
    feature_view.init_batch_scoring(training_dataset_version=1)
    
    try:
        tesla_df_b = feature_view.get_batch_data(start_time=start_date, end_time=end_date)
        return tesla_df_b
    except Exception as e:
        st.error(f"Error fetching batch data: {e}")
        st.stop()

def preprocess_data(df):
    tickers = df[['ticker']]
    encoder = OneHotEncoder()
    ticker_encoded_test = encoder.fit_transform(tickers)
    ticker_encoded_df_test = pd.DataFrame(ticker_encoded_test.toarray(), columns=encoder.get_feature_names_out(['ticker']))
    df = pd.concat([df, ticker_encoded_df_test], axis=1)
    df.drop('ticker', axis=1, inplace=True)
    
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df.drop(columns=['date'], inplace=True)
    
    return df, encoder

def load_model(mr):
    the_model = mr.get_model("stock_pred_model", version=3)
    model_dir = the_model.download()
    model = joblib.load(model_dir + "/stock_prediction_model.pkl")
    return model

def make_predictions(model, df):
    df_array = df.to_numpy()
    df_array = np.expand_dims(df_array, axis=1)
    predictions = model.predict(df_array)
    predictions = np.array(predictions, dtype=np.float32)
    predictions = predictions[0][0] * 100
    df['predictions'] = predictions.tolist()
    return df

def reconstruct_date_column(df):
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df.drop(columns=['year', 'month', 'day'], inplace=True)
    return df

def inverse_transform_tickers(df, encoder):
    ticker_encoded_df_test = df.filter(like='ticker_')
    ticker_encoded_array = ticker_encoded_df_test.to_numpy()
    original_tickers = encoder.inverse_transform(ticker_encoded_array)
    original_tickers_df = pd.DataFrame(original_tickers, columns=['ticker'])
    df = pd.concat([df.drop(columns=ticker_encoded_df_test.columns), original_tickers_df], axis=1)
    return df

def print_fancy_header(text, font_size=24):
    res = f'<span style="color:#ff5f27; font-size: {font_size}px;">{text}</span>'
    
def main():
    st.title("Stock Predictions")
    st.write("Predictions for stocks:")

    # Initialize Hopsworks
    api_key = os.environ.get('hopsworks_api')
    project = login_hopsworks(api_key)
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    # Define date range
    start_date = datetime.now() - timedelta(hours=48)
    end_date = datetime.now() - timedelta(hours=24)

    # Fetch and preprocess feature data
    tesla_df_b = get_feature_data(fs, start_date, end_date)
    tesla_df_b, encoder = preprocess_data(tesla_df_b)

    # Load the model and make predictions
    model = load_model(mr)
    tesla_df_b = make_predictions(model, tesla_df_b)

    # Reconstruct the date column and inverse transform tickers
    tesla_df_b = reconstruct_date_column(tesla_df_b)
    tesla_df_b = inverse_transform_tickers(tesla_df_b, encoder)

    # Display the dataframe and plot the predictions
    selected_ticker = st.selectbox('Select Ticker', tesla_df_b['ticker'])

# Filter the DataFrame based on the selected ticker
    filtered_df = tesla_df_b[tesla_df_b['ticker'] == selected_ticker]

# Display the filtered DataFrame
    st.dataframe(filtered_df)
    #st.dataframe(tesla_df_b)
    #st.line_chart(tesla_df_b.set_index('date')['predictions'])

    # Additional information
    st.write("Model used: stock_pred_model version 29")

if __name__ == "__main__":
    main()
