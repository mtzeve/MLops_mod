#!/bin/bash

set -e

cd /Stocks_news_prediction/Notebooks

jupyter nbconvert --to notebook --execute 8_inference_pipeline.ipynb