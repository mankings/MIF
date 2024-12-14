from src.base_calcs import *
from datetime import datetime

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PLOT_FOLDER = "plots"
DATE_FORMAT = '%Y-%m-%d'

def normalized_graphs(wallet_data, filename):

    # Normalize the 'Close' prices
    normalized_data = wallet_data / wallet_data.iloc[0]

    # Plot normalized prices
    plt.figure(figsize=(10, 6))
    normalized_data.plot(ax=plt.gca())
    plt.title(filename)
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.legend(title="Tickers")
    plt.grid(True)

    # Save the plot
    file_path = os.path.join(PLOT_FOLDER, filename)
    plt.savefig(file_path, format="png")
    
    # Clear the plot for the next ticker
    plt.clf()