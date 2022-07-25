#!/bin/sh
python3 generate_csv.py
python3 process_data.py
python3 model_predict.py
python3 upload_ml_prices.py