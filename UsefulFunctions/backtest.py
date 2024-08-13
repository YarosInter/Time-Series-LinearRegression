import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import MetaTrader5 as mt5


def compute_strategy_returns(y_test, y_pred):
    """
    Computes the strategy returns by comparing real percent changes with model predictions.

    This function creates a DataFrame that includes the actual percent changes (`y_test`) and the model's
    predicted values (`y_pred`). It then calculates the directional positions based on the actual and predicted
    values, and computes the returns by multiplying the real percent change by the predicted position.

    Args:
        y_test (Series or DataFrame): The actual percent changes (real values).
        y_pred (Series or DataFrame): The predicted values from the model.

    Returns:
        DataFrame: A DataFrame containing the actual percent changes, predicted values, 
                   real positions, predicted positions, and computed returns.
    """

    # Initialize a DataFrame with the actual percent changes and add the model's predictions
    df = pd.DataFrame(y_test)
    df["y_pred"] = y_pred

    # Add columns for the real and predicted directional positions
    df["real_position"] = np.sign(df_with_outcomes["pct_change"])
    df["pred_position"] = np.sign(df_with_outcomes["prediction"])

    # Calculate the strategy returns by multiplying the actual percent change by the predicted position
    # Note: Predictions are based on the previous bar's data, so no additional shift is needed
    df["returns"] = df_with_outcomes["pct_change"] * df_with_outcomes["pred_position"]

    return df



def plot_returns(returns_serie):
    """
    Plots the cumulative percentage returns from a trading strategy.

    This function takes a series of trading strategy returns, computes the cumulative sum, and 
    plots it as a percentage. The plot visualizes the profit and loss (P&L) over time.

    Args:
        returns_serie (pandas.Series): A series containing the returns from the trading strategy.

    Returns:
        None: The function generates and displays a plot.
    """
    
    (np.cumsum(returns_serie)*100).plot(figsize=(15,6))
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('P&L in %', fontsize=20)
    plt.title('Returns From Strategy', fontsize=20)
    plt.show()


def vectorize_backtest_returns(returns, anualization_factor, benchmark_asset=".US500Cash", mt5_timeframe=mt5.TIMEFRAME_H1):
    """
    Computes and prints the Sortino Ratio, Beta Ratio, and Alpha Ratio for a given set of returns serie.
    
    Parameters:
    - returns: Series of returns from a strategy.
    - anualization_factor: Factor used to annualize returns.
    - benchmark_asset: The benchmark asset for comparison (default is S&P 500).
    - mt5_timeframe: Timeframe for pulling benchmark data (default is 1H).

    Note: The timeframe for benchmark data must match the timeframe of the strategy's returns for accurate results.
    
    Returns:
    None
    """

    ### Computing Sortino Ratio ###
    
    # Sortino Ratio is being calculated without a Risk-Free Rate
    mean_return = np.mean(returns)
    downside_deviation = np.std(returns[returns<0])
    
    # Number of 15-minute periods in a year
    periods_per_year = anualization_factor * 252
    
    # Annualizing the mean return and downside deviation
    annualized_mean_return = mean_return * periods_per_year
    annualized_downside_deviation = downside_deviation * np.sqrt(periods_per_year)
    
    # Calculating the annualized Sortino ratio
    annualized_sortino = annualized_mean_return / annualized_downside_deviation
    
    print(f"Sortino Ratio: {'%.3f' % annualized_sortino}")
    print("\n- Positive Sortino (> 0): The investment’s returns exceed the target return after accounting for downside risk.")
    print("- Negative Sortino (< 0): The investment’s returns are less than the target return when considering downside risk.\n")


    ### Computing Beta Rario ###

    # Fetching the oldest date from the X_test set to pull data from the same date for the SP500
    date = returns.index.min()
    
    # Extracting the year, month, and day from the date
    year = date.year
    month = date.month
    day = date.day
    hour = date.hour
    min = date.minute
    sec = date.second
    
    # Pulling SP500 data from the specified date and time
    sp500_data = data.get_rates(".US500Cash", mt5.TIMEFRAME_H1, from_date=datetime(year, month, day))
    sp500_data = sp500_data[["close"]]
    
    # Computing the returns on the SP500 to 
    sp500_data["returns"] = sp500_data["close"].pct_change(1)
    sp500_data.drop("close", axis=1, inplace=True)
    sp500_data.dropna(inplace=True)
    
    # Concatenate values between the returns in the predictions and the returns in the SP500
    val = pd.concat((returns, sp500_data["returns"]), axis=1)
    
    # Changing columns names to indentify each one
    val.columns.values[0] = "Returns Pred"
    val.columns.values[1] = "Returns SP500"
    val.dropna(inplace=True)

    # Calculating Beta Ratio
    covariance_matrix = np.cov(val.values, rowvar=False)
    covariance = covariance_matrix[0][1]
    variance = covariance_matrix[1][1]
    beta = covariance / variance
    
    print(f"Beta Ratio: {'%.3f' % beta}")
    print("\n- Beta ≈ 1: The asset moves in line with the market.")
    print("- Beta < 1: The asset is less volatile than the market (considered less risky).")
    print("- Beta > 1: The asset is more volatile than the market (higher potential return but also higher risk).\n")


    ### Computing Alpha Ratio ###

    alpha = (anualization_factor * 252 * mean_return * (1-beta))*100
    
    print(f"Alpha Ratio: {'%.3f' % alpha}")
    print("\n- Positive Alpha (> 0): Indicates the investment outperformed the market.")
    print("- Negative Alpha (< 0): Indicates the investment underperformed the market.")



def compute_model_accuracy(real_positions, predicted_positions):
    """
    Computes and displays the accuracy of predicted positions compared to real positions.

    Parameters:
    real_positions (list or array-like): The actual positions.
    predicted_positions (list or array-like): The positions predicted by the model.

    Returns:
    pd.DataFrame: A DataFrame containing the real positions, predicted positions, and accuracy (1 for correct, 0 for incorrect).
    
    Displays:
    - Counts of correct and incorrect predictions.
    - Histogram showing the distribution of accuracy values.
    - Model accuracy percentage.
    """
    
    # Creating Dataframe with real positons and predicted positions
    df_accuracy = pd.DataFrame(real_positions)
    df_accuracy["pred_position"] = predicted_positions
    
    # Assigning 1 if the position forecasted is equal to the real position and 0 otherwise
    df_accuracy["accuracy"] = np.where(df_accuracy["real_position"] == df_accuracy["pred_position"], 1, 0)

    # Count the occurrences of each unique accuracy value in the 'accuracy' column and store the result in 'accuracy'
    accuracy = df_accuracy["accuracy"].value_counts()

    # Printing explanation for the counts of 0 and 1 in the 'accuracy' column
    print("Counts of 0 indicate instances where the predicted position did not match the real position.")
    print("Counts of 1 indicate instances where the predicted position matched the real position.\n")
    print(accuracy)

    # Total counts of occurrences where model was right (number assigned 1) divided into the total number of predictions
    model_accuracy = accuracy[1] / len(df_accuracy)
    print(f"\nModel has an accuracy of: {model_accuracy * 100:.2f}%")

    plt.hist(df_accuracy["accuracy"], bins=3)
    plt.xticks([0, 1])
    plt.title("Model Accuracy", fontsize=20)
    plt.ylabel("Counts", fontsize=15)
    plt.xlabel("Distribution", fontsize=15);

    return df_accuracy.head()