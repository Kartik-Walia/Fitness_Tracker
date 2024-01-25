import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans


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
# Frequency features (Appying DFT)
# --------------------------------------------------------------

# We're gonna apply DFT (Discrete Fourier Transformation) which says any sequence of measurements we peform can be represented by a combination of sinusoid functions with different frequencies, by this we're gonna decompose the original signal into the different frequencies 
# We're gonna extract Amplitude (for each relevant frequencies that are part of the time window), max frequency, weighted frequency (average) & Power spectral entropy (All fo these 4 are part of the machine learning for the Quantified self case studies)

# Creating copy & also reset index bcoz the functions we're about to use expect discrete index as input but we've time series index 
df_freq = df_temporal.copy().reset_index() 
FreqAbs = FourierTransformation()   # Creating an instance of Fourier Transformation class

fs = int(1000 / 200)    # Sampling rate
# This time we're gonna set the window size to the average length of a repetition that was about 2.8s divided by 200ms
ws = int(2800 / 200)

# Applying FT to 1 column 
df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], ws, fs)
df_freq.columns

# Visualize results 
subset = df_freq[df_freq["set"] == 15]
subset["acc_y"].plot()
# Now let's have a look at some of the values we've added
subset[["acc_y_max_freq", "acc_y_freq_weighted", "acc_y_pse", "acc_y_freq_1.429_Hz_ws_14", "acc_y_freq_2.5_Hz_ws_14"]].plot()

# Now let's apply to over all dataframe by looping over all the columns & also splitting by set
df_freq_list = []
for s in df_freq["set"].unique():
    print(f"Applying Fourier Transformations to set {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)
    
# Overwriting the original data frame (setting index back to epoch and droping the discrete index)
df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)
# Now we can see more missing values in the frequency domain bcoz we use a larger window size 

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

# You can imagine that since we've added extra columns which are all based on a rolling window that the values in all of the columns between the different rows are highly correlated and this is typically something we wanna avoid when we're building models bcoz this would cause overfitting

# In order to do that we're going to allow for a certain percentage to overlap and remove rest of the data 

# Droping all of the missing values that are currently present in the data frame 
df_freq = df_freq.dropna()

# On loking in literature you'll see that typically an allowance of 50% is recommended meaning that in our case we will get rid of 50% of the data by skipping every other row and this will result in alot of data loss but it has been shown to pay off in the long run by making your models less prone to overfitting 
df_freq = df_freq.iloc[::2]   # This way of writing iloc specifies that you want every other row or every 2nd row (starting from the 1st initial row of the data frame)
# By removing every other row we bsically reduce the correlation between the adjacent records that are in the dataframe 

# --------------------------------------------------------------
# Clustering (using K means clustering library from the Sk learn package)
# --------------------------------------------------------------

df_cluster = df_freq.copy()

# K means clustering is an unsupervised algorithm so you can basically provide the data and the amount of clusters that we want to use to the algorithm and then it'll basically group all of our individual rows to a specific cluster, so we can basically group based on a unsupervised learning algorithm (typically doen for segmenting customers purchase behaviour, anomaly detectiong, image compressions, etc.)
# Just like PCA we can use the elbow method here again but now using different values namely the inertia which is the sum of squared distances of samples to the closest cluster center and we 'll be computing that for each K so for each number of clusters and we'll basically loop over various numbers of K and then create a plot for determinig the optimal amount of K for our dataset 
cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2, 10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)
    
plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias)
plt.xlabel("k")
plt.ylabel("Sum of squared distances")
plt.show()
# We can clearly see that elbow joint occurs at 5 and afterwards the decrease in sum of squared distance diminishes, so optimal values of k is 5

kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

# Plot clusters 
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# Plot accelerometer data to compare
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=l)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# We can now have a look at the individual exercises and see where they eblong withing this 3-dimensional plot, so we can see that the bench press and the overhead press vary close to each other which is to be expected bcoz the movements look alot like each other and also in  green and gray we've deadlift and the row which also look very similar to each other, we can see that squat is very different from all the other exercises & rest is all over the place which is also to be expected bcoz rest is pretty random 

# By looking we clearly see that the cluster 2 corresponds to the squat, cluster 1 probably captures bench press and overhead press data

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/interim/03_data_features.pkl")