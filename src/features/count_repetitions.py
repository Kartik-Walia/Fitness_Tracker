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

#filters the dataframe to exclude rows where the label column is equal to rest
df=df[df["label"] !="rest"]

#calculating acc_r and gyr_r
acc_r= df["acc_x"] **2 +df["acc_y"] ** 2 + df["acc_z"]** 2
gyr_r= df["gyr_x"] **2 +df["gyr_y"]**2 +df["gyr_z"]** 2
df["acc_r"]=np.sqrt(acc_r)
df["gyr_r"]=np.sqrt(gyr_r)


# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------


#splitting data by labels 
bench_df=df[df["label"] == "bench"]
squat_df=df[df["label"] == "squat"]
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

fs= 1000/200
#5 instances per second
LowPass= LowPassFilter()


  


# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------

bench_set=bench_df[bench_df["set"]==bench_df["set"].unique()[0]]
squat_set = squat_df[squat_df["set"] == squat_df["set"].unique()[0]]
row_set =row_df[row_df["set"]==row_df["set"].unique()[0]]
ohp_set = ohp_df[ohp_df["set"]==ohp_df["set"].unique()[0]]
dead_set=dead_df[dead_df["set"]== dead_df["set"].unique()[0]]

bench_set["acc_r"].plot()
column="acc_r"
LowPass.low_pass_filter(
    bench_set,col=column,sampling_frequency=fs,cutoff_frequency=0.3, order=5
    )[column +"_lowpass"].plot()

#trends in plot can be observed by changing the cutoff frequency 
#looking for 5 repetitions in total for heavy  bench press set


column="acc_y"
LowPass.low_pass_filter(
    bench_set,col=column,sampling_frequency=fs,cutoff_frequency=0.4, order=10
    )[column +"_lowpass"].plot()

column="acc_r"
LowPass.low_pass_filter(
    row_set,col=column,sampling_frequency=fs,cutoff_frequency=0.4, order=10
    )[column +"_lowpass"].plot()

column="acc_y"
LowPass.low_pass_filter(
    squat_set,col=column,sampling_frequency=fs,cutoff_frequency=0.4, order=10
    )[column +"_lowpass"].plot()

column="acc_y"
LowPass.low_pass_filter(
    dead_set,col=column,sampling_frequency=fs,cutoff_frequency=0.4, order=10
    )[column +"_lowpass"].plot()
column="acc_y"
LowPass.low_pass_filter(
    ohp_set,col=column,sampling_frequency=fs,cutoff_frequency=0.4, order=10
    )[column +"_lowpass"].plot()



# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------


# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------