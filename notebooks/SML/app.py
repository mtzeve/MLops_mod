from datetime import datetime, timedelta
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from matplotlib import pyplot
import warnings
import os
import hopsworks
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

import folium
from streamlit_folium import st_folium
import json

import asyncio
import nest_asyncio

# Apply nest_asyncio to the current event loop
nest_asyncio.apply(asyncio.get_event_loop())

start_date = datetime.now() - timedelta(hours=48)
end_date = datetime.now() - timedelta(hours=24)


warnings.filterwarnings("ignore")

api_key = os.getenv('HOPSWORKS_API_KEY')
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()

@st.cache_data()
def retrieve_dataset(_fv, start_date, end_date):
    st.write(36 * "-")
    print_fancy_header('\n💾 Dataset Retrieving...')
    batch_data = fv.get_batch_data(start_time = start_date, end_time = end_date)
    return batch_data


@st.cache_data()
def get_feature_view():
    fv = fs.get_feature_view("tesla_stocks_fv", 5)
    return fv


@st.cache_data()
def get_model(_project = project):
    mr = project.get_model_registry()
    model = mr.get_model("stock_pred_model", version = 10)
    model_dir = model.download()
    return joblib.load(model_dir + "/stock_prediction_model.pkl")
#
#
def print_fancy_header(text, font_size=24):
    res = f'<span style="color:#ff5f27; font-size: {font_size}px;">{text}</span>'
    st.markdown(res, unsafe_allow_html=True)
#
#def transform_preds(predictions):
#    return ['Fraud' if pred == 1 else 'Not Fraud' for pred in predictions]    

progress_bar = st.sidebar.header('⚙️ Working Progress')
progress_bar = st.sidebar.progress(0)
#st.title('🆘 Fraud transactions detection 🆘')

#st.write(36 * "-")
#print_fancy_header('\n📡 Connecting to Hopsworks Feature Store...')

#st.write(36 * "-")
#print_fancy_header('\n🤖 Connecting to Model Registry on Hopsworks...')
model = get_model(project)
st.write(model)
st.write("✅ Connected!")

progress_bar.progress(40)

st.write(36 * "-")
print_fancy_header('\n✨ Fetch batch data and predict')
fv = get_feature_view()


if st.button('📊 Make a prediction'):
    batch_data = retrieve_dataset(_fv, start_date, end_date)
    st.write("✅ Retrieved!")
    #progress_bar.progress(55)
    #predictions = model.predict(batch_data)
    #predictions = transform_preds(predictions)
    #batch_data_to_explore = batch_data.copy()
    #batch_data_to_explore['fraud'] = predictions
    #explore_data(batch_data_to_explore)

st.button("Re-run")
