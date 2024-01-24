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


# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------


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
