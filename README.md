# Time-Series Linear Regression

This repository is a comprehensive project focused on time series analysis and linear regression modeling using financial data. The main objective is to develop predictive models to forecast financial trends and prices over time.

## Project Overview

Time series analysis is crucial for understanding and forecasting financial data. This project utilizes a dataset of historical financial data and applies linear regression models to predict future trends. The project is structured to facilitate ease of understanding and replication of the results.

### Key Features

- **Data Retrieval**: Custom functions are used to fetch historical financial data from MetaTrader 5.
- **Data Processing**: The data is processed and transformed to prepare it for modeling, including steps such as standardization and feature engineering.
- **Model Development**: Linear regression models are developed and trained on the processed data.
- **Model Evaluation**: The performance of the models is evaluated using appropriate metrics to ensure robustness and accuracy.
- **Visualization**: Key results and trends are visualized to provide insights into the data and model performance.

## Repository Structure

- **`data.py`**: Contains functions for data retrieval and manipulation, including:
  - `get_rates`: Fetches historical price data for a given symbol and timeframe.
  - `add_shifted_columns`: Adds lagged versions of columns to a DataFrame to create features for time series prediction.
  - `split_data`: Splits data into training and testing sets.

- **`backtest.py`**: Provides functions for backtesting financial strategies, including:
  - `compute_strategy_returns`: Calculates returns for a given strategy.
  - `plot_returns`: Visualizes the cumulative returns of a strategy.
  - `vectorize_backtest_returns`: Computes and prints financial metrics such as Sortino, Beta, and Alpha ratios.
  - `compute_model_accuracy`: Calculates and visualizes the accuracy of model predictions.
  - `strategy_drawdown`: Computes and visualizes the drawdown of the strategy.

- **`LinearRegression Model.ipynb`**: Jupyter Notebook containing the main code for data processing, model development, evaluation, and visualization.

- **`.gitignore`**: Specifies files and directories to be ignored by Git.

- **`LICENSE`**: The MIT License under which this project is distributed.

- **`README.md`**: This file.

## Installation

To run this project, you need to have Python installed on your machine along with the following packages:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `MetaTrader5`

## Getting Started

1. **Clone the Repository**: Clone this repository to your local machine using `git clone https://github.com/YarosInter/Time-Series-LinearRegression.git`.

2. **Install Dependencies**: Ensure all required Python packages are installed.

3. **Run the Notebook**: Open `LinearRegression Model.ipynb` in Jupyter Notebook to explore the model and results.

4. **Data Retrieval**: Use `data.py` to fetch and preprocess financial data.

5. **Backtesting**: Leverage `backtest.py` to evaluate your strategy's performance.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or new features.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

