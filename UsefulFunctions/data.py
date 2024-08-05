import MetaTrader5 as mt5
from datetime import datetime
import pandas as pd


def get_rates(symbol, time_frame=mt5.TIMEFRAME_H1, from_date=None, to_date=None):
    """
    Retrieve historical price data for a given financial symbol and time frame, and return it as a pandas DataFrame.

    Parameters:
    symbol (str): The financial instrument symbol to retrieve data for (e.g., 'EURUSD').
    from_date (datetime, optional): The start date for the data retrieval. Defaults to January 1, 2010.
    to_date (datetime, optional): The end date for the data retrieval. Defaults to the current date and time.
    time_frame (int, optional): The time frame for the data retrieval. Defaults to hourly (mt5.TIMEFRAME_H1).

    Returns:
    pandas.DataFrame: A DataFrame containing the historical price data, with the time column set as the index.
                      The DataFrame includes columns such as open, high, low, close, tick_volume, spread, and real_volume.
                      
    Example:
    >>> from datetime import datetime
    >>> rates = get_rates('EURUSD', datetime(2022, 1, 1), datetime(2022, 2, 1))
    >>> print(rates.head())
    
    Notes:
    - The function uses MetaTrader5 (mt5) to fetch the historical data.
    - If no from_date or to_date is provided, the function will default to retrieving data from January 1, 2010, to the current date.
    - The time column in the returned DataFrame is converted to datetime format and set as the index.
    """
        
    # Compute Date Now   
    if from_date is None:
        from_date = datetime(2010, 1, 1)
    if to_date is None:
        to_date = datetime.now()

    # Extract n Bars before now
    rates = mt5.copy_rates_range(symbol, time_frame, from_date, to_date)

    # Transform Tuple into DataFrame
    df_rates = pd.DataFrame(rates)

    # Convert number format of the date into date format
    df_rates["time"] = pd.to_datetime(df_rates["time"], unit="s")
    
    # Set column time as the index of the DataFrame
    df_rates = df_rates.set_index("time")

    return df_rates