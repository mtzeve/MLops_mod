#!/bin/bash

set -e

cd /Stocks_news_prediction/Notebooks/SML

python feature_pipeline.py

python inference_pipeline.py
