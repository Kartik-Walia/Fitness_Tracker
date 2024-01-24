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
duration = (
    df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]
)  # -1 gives us last time stamp & 0 gives us first time stamp
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


# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------


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
