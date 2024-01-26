import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df= pd.read_pickle("../../data/interim/01_data_processed.pkl")
df=df[df["label"] !="rest"]

#calculating acc_r and gyr_r
acc_r= df["acc_x"]**2 +df["acc_y"]** 2 +df["acc_z"]** 2
gyr_r=df["gyr_x"] ** 2 +df["gyr_y"]**2 +df["gyr_z"]** 2
df["acc_r"]=np.sqrt(acc_r)
df["gyr_r"]=np.sqrt(gyr_r)


# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------


#splitting data by labels 
bench_df=df[df["label"] == "bench"]
squat_df=df[df["label"] == "sqaut"]
row_df=df[df["label"] == "row"]
ohp_df=df[df["label"] == "ohp"]
dead_df=df[df["label"] == "dead"]





# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------

#looping over data and create plots to spot repetitions 
#and find patterns of plot

plot_df= bench_df
#creating filter to look at each individual set
plot_df[plot_df["set"]==plot_df["set"].unique()[0]]["acc_x"].plot()
plot_df[plot_df["set"]==plot_df["set"].unique()[0]]["acc_y"].plot()
plot_df[plot_df["set"]==plot_df["set"].unique()[0]]["acc_z"].plot()
plot_df[plot_df["set"]==plot_df["set"].unique()[0]]["acc_r"].plot()

plot_df[plot_df["set"]==plot_df["set"].unique()[0]]["gyr_x"].plot()
plot_df[plot_df["set"]==plot_df["set"].unique()[0]]["gyr_y"].plot()
plot_df[plot_df["set"]==plot_df["set"].unique()[0]]["gyr_z"].plot()
plot_df[plot_df["set"]==plot_df["set"].unique()[0]]["gyr_r"].plot()
#we can see in the plot that heavy has 5 sets
#medium has 10 sets. But we are not sure but it doesnt give complete view 
#for excercises like bench press it is more useful to look at accelertion data rather than gyroscope data


# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------


# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------


# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------