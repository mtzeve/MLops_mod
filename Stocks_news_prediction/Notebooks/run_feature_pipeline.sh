#!/bin/bash

set -e

cd /Stocks_news_prediction/Notebooks

jupyter nbconvert --to notebook --execute 5_feature_pipeline.ipynb