# Time-Series Linear Regression

This repository contains a comprehensive project focused on time series analysis and linear regression modeling using financial data. The primary objective is to develop predictive models to forecast financial trends and prices over time.

## Project Overview

Time series analysis is crucial in understanding and forecasting financial data. This project utilizes a dataset of historical financial data and applies linear regression models to predict future trends. The project is structured to facilitate ease of understanding and replication of the results.

### Key Features

- **Data Retrieval**: Custom functions are used to fetch historical financial data from MetaTrader 5.
- **Data Processing**: The data is processed and transformed to prepare it for modeling, including steps such as standardization and feature engineering.
- **Model Development**: Linear regression models are developed and trained on the processed data.
- **Model Evaluation**: The performance of the models is evaluated using appropriate metrics to ensure robustness and accuracy.
- **Visualization**: Key results and trends are visualized to provide insights into the data and model performance.

## Repository Structure

The repository is organized as follows:

- `data.py`: Contains the custom function `get_rates` for fetching historical data from MetaTrader 5.
- `data.py`: Contains the custom function `add_shifted_columns` for adding additional columns with older values to the DataFrame
- `LinearRegression Model.ipynb`: Jupyter Notebook containing the main code for data processing, model development, evaluation, and visualization.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `LICENSE`: The MIT License under which this project is distributed.
- `README.md`: This file.

## Installation

To run this project, you need to have Python installed on your machine along with the following packages:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `MetaTrader5`
