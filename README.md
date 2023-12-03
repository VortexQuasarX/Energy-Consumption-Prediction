# Energy Consumption Prediction Project

## Overview

This project focuses on predicting energy consumption using time series data. The dataset used contains information about global active power, global reactive power, voltage, global intensity, and sub-metering readings. The primary goal is to develop a predictive model using Long Short-Term Memory (LSTM) neural networks to forecast energy consumption accurately.

## Dependencies

- [pyspark](https://pypi.org/project/pyspark/): PySpark library for Apache Spark.
- [numpy](https://numpy.org/): NumPy library for numerical operations.
- [pandas](https://pandas.pydata.org/): Pandas library for data manipulation and analysis.
- [matplotlib](https://matplotlib.org/): Matplotlib library for data visualization.
- [scikit-learn](https://scikit-learn.org/): Scikit-learn library for machine learning.
- [keras](https://keras.io/): Keras library for building neural networks.

## Getting Started

1. Open the Jupyter Notebook
2. Mount Google Drive to access the dataset.
3. Install required libraries using `!pip install pyspark` and other necessary libraries.
4. Run the notebook cells sequentially.

## Data Analysis

- Check for NULL values in the dataset.
- Display average values.
- Resample data over different time intervals (month, day, hour).
- Visualize resampled data.

## LSTM Model

- Preprocess data for LSTM.
- Build an LSTM model using Keras.
- Train the model and evaluate performance.
- Visualize actual vs. predicted values.

## Results

The LSTM model demonstrates promising results in predicting energy consumption. Key findings and metrics include:

### Root Mean Squared Error (RMSE)

The model achieves an RMSE of [insert RMSE value], indicating [provide context for the RMSE value].

### Visualizations

![Actual vs. Predicted Energy Consumption](/visualization.png)

The plot above compares the actual and predicted energy consumption over a specific time period. The model captures [highlight any notable patterns or trends].

## Acknowledgments

Special thanks to Harish ,Sriharsh for their contributions.

