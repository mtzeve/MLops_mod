#!/bin/bash

set -e

cd /Stocks_news_prediction/Notebooks
chmod +x ./Stocks_news_prediction/Notebooks/run_inference_pipeline.sh


jupyter nbconvert --to notebook --execute 8_inference_pipeline.ipynb
