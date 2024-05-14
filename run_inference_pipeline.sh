#!/bin/bash

set -e

cd Mlops_mod/Stock_news_prediction/Notebooks

jupyter nbconvert --to notebook --execute 8_inference_pipeline.ipynb