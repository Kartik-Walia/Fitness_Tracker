import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")

predictor_columns = list(df.columns[:6])

# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

# We're going to interpolate the data meaning that anywhere where there is a gap, we're just gonna connecting the dots over here & interpolate them linearly, so we're gonna do straight line

for col in predictor_columns:
    df[col] = df[col].interpolate()

df.info()  # You can see there are no msising values (9009 non-null)

# --------------------------------------------------------------
# Calculating average duration of a set
# --------------------------------------------------------------

# This is a preparation we've to make in order to later apply the butterworth low pass filter which is a filter to basically filter subtle noise in the data set

df[df["set"] == 25]["acc_y"].plot()  # Heavy ohp
# We did heavy sets for 5 repetitions & medium sets for 10 repetitions, here we can see there're clearly 5 peaks which translate to the 5 repetitions in the heavy set
df[df["set"] == 50]["acc_y"].plot()  # Medium bench (thatswhy showing 10 peaks)

# By applying the low pass filter we can basically make these lines smoother meaning that we're just going to look at the overall movement pattern & not necessary at like the small tiny differences that are apparent between every repetition and also every participant, so like the small incremental adjustments of the bar and your hands and feet positions for example, we wanna filter out that and look at the big movement patterns
# Thatswhy we've to know how long a repetition takes bcoz in doing so we can later adjust the frequency settings and basically tune to a frequency that is higher meaning faster repetitions itself in order tof ilter out the noise

# Calculate the average duration of a set
duration = (df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0])  # -1 gives us last time stamp & 0 gives us first time stamp
duration.seconds  # Round off to nearest seconds

# This is how we can calculate duration for single set, now let's do that in a loop
# So we're gonna loop over all the unique sets that are present in the data set
for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]

    duration = stop - start
    # Now we're going to add this duration to the data frame in a new column and also taking the set into account
    df.loc[(df["set"] == s), "duration"] = duration.seconds

duration_df = df.groupby(["category"])["duration"].mean()

# Average set duration for sets & divide by amount of repetitions that was present during respective sets
duration_df.iloc[0] / 5    # Average duration for a single repetition for Heavy sets
duration_df.iloc[1] / 10   # Average duration for a single repetition for Medium sets

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()
LowPass = LowPassFilter()   # Creating an instance of the low pass filter class

fs = 1000 / 200   # Defining sampling frequency (btw each record there's a step size of 200ms)
cutoff = 1.3   # Defining cutoff frequency (by setting cutoff freq high, we're allowing higher freq means more rough lines, The higher is number the less we filter smooth data, high no close to raw data & lower no close to smooth data)

df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)

subset = df_lowpass[df_lowpass["set"] == 45]
print(subset["label"][0])

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

# Applying butterworth filter to all fo the columns in a loop 
for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)  # Apply the low-pass filter to the specified column (This adds new column) 
    df_lowpass[col] = df_lowpass[col + "_lowpass"]  # Overwrite the original column with the new filtered column
    del df_lowpass[col + "_lowpass"]    # Delete the temporary column

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

# PCA is a dimensionality reduction method, so we wanna move from alot of columns to less columns 

df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()   # Creating an instance of the low pass filter

# Determining the optimal amount of principal components
pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

# Methods to determine the optimal amount of principal components & we can do this by using the elbow technique 
plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("principal component number")
plt.ylabel("explained variance")
plt.show()
# Elbow occurs at 3, so optimal amount of principal component is 3

df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)
# We've basically summarised the 6 columns (accelaration & gyroscope) into 3 principal components here, while capturing or explaining as much of variance as possible 
# We'll leave the principal components in here next to like all the other columns, not like the low pass filter we're not going to overwrite the initial values 
# Later doing feature selection we're going to check whether the principal components actually peform better than the individual columns

subset = df_pca[df_pca["set"] == 35] 
subset[["pca_1", "pca_2", "pca_3"]].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = df_pca.copy()

acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

subset = df_squared[df_squared["set"] == 14]

subset[["acc_r", "gyr_r"]].plot(subplots=True)  # subplots=True is to create separate subplots for each selected column

# The main goal of implementing a magnitude scaled version of all values over here is bcoz it's impartial to device orientation & this helps us to make model generalize better to different participants 
# Now when we look at dataframe, we've 5 additional features as well as the filtered version of the columns 

# --------------------------------------------------------------
# Temporal abstraction (Calculating rolling averages)
# --------------------------------------------------------------

df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()   # Creating an instance of Numerical Abstraction class

predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

# Window size basically captures how many values we want to look back for each of the values, so if we set ws = 5 this means we can't use first 4 values of the dataframe & later we'll be splitting data based on teh individual sets, so remember that by setting a very large window size we also throw away alot of data that we can use bcoz it'll result in missing values 
# For now, we're going to start off by window size equal to 1s, so in order to compute that remember we've step size of 200ms, so in roder to determine the window size for 1s we take 1000ms and divide it by step size 200ms which will result in 5, meaning that in order to get a window size of 1s we 5 steps
ws = int(1000 / 200)  

# Computing mean & standard deviation by looping over all cols 
for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "std")
# This computes bench data also in dead lift and other exercises which is wrong, so we need to split it up 

df_temporal_list = []
for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)   # Overwriting the original dataframe

df_temporal.info()  # You can see some values are missing 

subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
subset[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()
# We can clearly see in graph too that mean is smoother version of the actual values

# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
