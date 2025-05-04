import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_csv_files(folder_path):
    """
    Get a list of all CSV files in a given folder.
    
    :param folder_path: Path to the folder to search for CSV files.
    :return: List of CSV file names (with full paths).
    """
    try:
        # Ensure the folder exists
        if not os.path.exists(folder_path):
            print(f"Error: The folder '{folder_path}' does not exist.")
            return []

        # Get all CSV files in the folder
        csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".csv")]

        if not csv_files:
            print(f"No CSV files found in '{folder_path}'.")

        return csv_files

    except Exception as e:
        print(f"Error: {e}")
        return []

def plot_and_export_average_with_std(ax, csv_files, time_column='time', value_column='value', label='Average Response'):
    """
    Plots the average response and standard deviation shading from multiple CSV files,
    trims all to the shortest run, and exports the results.

    Returns:
        dict: {
            'time': np.ndarray,
            'mean': np.ndarray,
            'lower': np.ndarray,
            'upper': np.ndarray,
            'ax': matplotlib.axes.Axes
        }
    """

    # Load and truncate
    dataframes = [pd.read_csv(file) for file in csv_files]
    min_length = min(len(df) for df in dataframes)

    # timestep is in nanoseconds
    times = [(df[time_column].values[:min_length] - df[time_column].values[0])*1e-9 for df in dataframes]
    values = [df[value_column].values[:min_length] for df in dataframes]

    time = times[0]  # Assuming all runs have the same time base up to min_length

    value_array = np.vstack(values)
    mean_values = np.mean(value_array, axis=0)
    std_values = np.std(value_array, axis=0)

    lower = mean_values - std_values
    upper = mean_values + std_values

    # Plot
    ax.plot(time, mean_values, label=label, color='gray')
    ax.fill_between(time, lower, upper, linestyle="--",color='gray', alpha=0.3, label='±1 Std Dev')   

    # Return data and axes object
    return {
        'time': time,
        'mean': mean_values,
        'lower': lower,
        'upper': upper,
        'ax': ax
    }

def add_avg_data_to_plot(ax):
    '''averages the data in the side folder and adds it to the passed graph
    
    @returns pipe_angle, and pend_angle
    
    '''
    csvs = get_csv_files("./ball_plant/data/steering_data")
    out_pipe = plot_and_export_average_with_std(ax[0], csvs, time_column='timestamp', value_column="ball_roll.position", label="Average Robot Data")
    out_pend = plot_and_export_average_with_std(ax[1], csvs, time_column='timestamp', value_column="roll_joint.position", label="Average Robot Data")
    
    return ax, out_pipe, out_pend
