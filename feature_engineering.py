# %%
import requests
import pandas as pd
import json
import datetime
import numpy as np
from datetime import timedelta 

# %%
def getNews(api_key,endpoint,ticker,from_date,to_date,num=1000):
    # Set the parameters for the request
    params = {
        "api_token": api_key,
        "s": ticker,
        "from": from_date, 
        "to": to_date,
        "limit": num,
    }
    
    # Make the request to the API
    response = requests.get(endpoint, params=params)
    
    # Print the response from the API
    print(response.json())

    #Return a Pandas dataframe from the response
    return pd.DataFrame(response.json())

# %%



