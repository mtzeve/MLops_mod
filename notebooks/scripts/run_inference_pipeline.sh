#!/bin/bash

set -e

cd ./notebooks/SML

python historical_news.py

python historical_stock.py

python news_preprocessing.py

python stock_preprocessing.py

python feature_pipeline.py

python feature_view.py

python training_pipeline.py

python inference_pipeline.py
