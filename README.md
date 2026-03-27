Overview
A two-part Python toolkit for analyzing Indian equity stocks using minute-level OHLCV data. Includes classical time-series decomposition for exploratory analysis and a TensorFlow LSTM model for next-day closing price prediction.

Features
Resamples raw minute-level data to daily OHLCV aggregates
Additive time-series decomposition (trend / seasonal / residual) via statsmodels
Stacked 2-layer LSTM with dropout, trained on a 30-day sliding window
Evaluates predictions with RMSE, MAE, and MAPE
Configurable to run on a single stock or the entire dataset

Output
Decomposition plots (observed, trend, seasonal, residual) for each stock
Actual vs. predicted close price chart on the test set
Training/validation loss curve
Summary table of RMSE, MAE, MAPE across all processed stocks
