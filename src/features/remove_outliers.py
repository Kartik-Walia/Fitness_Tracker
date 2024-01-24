import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor  # pip install scikit-learn

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# All numerical columns should be considered when looking at outliers (first 6)
outlier_columns = list(df.columns[:6])

# --------------------------------------------------------------
# Plotting outliers
# --------------------------------------------------------------

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100

# Creating boxplots [ splitting(or grouping) by label ]
df[["gyr_y", "label"]].boxplot(by="label", figsize=(20, 10))

# Spliting the Accelerometer and Gyroscope data
df[outlier_columns[:3] + ["label"]].boxplot(
    by="label", figsize=(20, 10), layout=(1, 3)
)  # Accelerometer
df[outlier_columns[3:] + ["label"]].boxplot(
    by="label", figsize=(20, 10), layout=(1, 3)
)  # Gyroscope


# Function that can plot outliers in case of a binary outlier score i.e. True or False values
def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()


# custom function to plot outliers in real time


# --------------------------------------------------------------
# Interquartile range (distribution based)
# --------------------------------------------------------------


# IQR function for marking outliers (outlier will be marked as True & non-outlier will be marked as False)
def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset


# Plot a single column
col = "acc_x"
dataset = mark_outliers_iqr(
    df, col
)  # Needed before ploting binary outliers bcoz binary needs outliers marked as True
plot_binary_outliers(
    dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
)

# Loop over all outlier columns
for col in outlier_columns:
    dataset = mark_outliers_iqr(df, col)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )
# Plotting in above way doesn't assisst in differentiating between different excercises

# --------------------------------------------------------------
# Chauvenets criteron (distribution based)
# --------------------------------------------------------------

# Check for normal distribution [ By using histogram (bell shaped) or boxplots(symmetrical) ]
df[outlier_columns[:3] + ["label"]].plot.hist(
    by="label", figsize=(20, 20), layout=(3, 3)
)  # Accelerometer
df[outlier_columns[3:] + ["label"]].plot.hist(
    by="label", figsize=(20, 20), layout=(3, 3)
)  # Gyroscope


# Chauvenet's function
def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.

    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset


# Loop over all columns
for col in outlier_columns:
    dataset = mark_outliers_chauvenet(df, col)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )
# Chauvenet gives alot less outliers (less red) in comparison to IQR method

# --------------------------------------------------------------
# Local outlier factor (distance based) (unsupervised learning model i.e. unlabeled dataset)
# --------------------------------------------------------------

# LOF uses local deviation of the density of a given sample with respect to its neighbors
# Another key difference is we were looking at individual columns, now we'll look at individual rows
# So we're going to consider 6 data points within a row & use that to compare to all of it's neighbours (no. of neighbours that we specify)


# LOF function
def mark_outliers_lof(dataset, columns, n=20):
    """Mark values as outliers using LOF

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        n (int, optional): n_neighbors. Defaults to 20.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """
    
    dataset = dataset.copy()    

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1   # -1 is an outlier & 1 is not an outlier, this'll result in same T/F column
    return dataset, outliers, X_scores    # X_sscores tells us certainity whether it's a outlier or not (the more -ve value is less chance of it to be an outlier)

# We don't have to loop over all the individual columns bcoz we're just gonna import the whole dataset
dataset, outliers, X_scores = mark_outliers_lof(df, outlier_columns)

# Loop over all columns just to see results in similar way as above
for col in outlier_columns:
    plot_binary_outliers(dataset=dataset, col=col, outlier_col="outlier_lof", reset_index=True)
# Now outliers are starting to be identified more within the data itself so before it was usually we could basically draw a straight line and anything underneath that would be marked as an outlier or on top everything above a certain line would be marked as an outlier but now we're starting to see data points within what seems to be a regular movement pattern is marked as an outlier 

# --------------------------------------------------------------
# Check outliers grouped by label
# --------------------------------------------------------------

# We're not putting the whole dataframe we're putting the dataframe with a selection on the label
label = "bench"

# IQR method
for col in outlier_columns:
    dataset = mark_outliers_iqr(df[df["label"] == label], col)
    plot_binary_outliers(dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True)
# Not considering IQR bcoz throwing a lot of data as outlier even clusters are marked as outliers

# Chauvenet method 
for col in outlier_columns:
    dataset = mark_outliers_chauvenet(df[df["label"] == label], col)
    plot_binary_outliers(dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True)
# Can be considered as shows very less outliers and generally here clusters aren't shown as outliers

# LOR method
dataset, outliers, X_scores = mark_outliers_lof(df[df["label"] == label], outlier_columns)
for col in outlier_columns:
    plot_binary_outliers(dataset=dataset, col=col, outlier_col="outlier_lof", reset_index=True)
# Again we see behaviour where we're marking points within the bulk of the data and not necessarily above or below a certain point

# --------------------------------------------------------------
# Choose method and deal with outliers (Choosing Chauvenet method as final one)
# --------------------------------------------------------------

# Test on single column
col = "gyr_z"
dataset = mark_outliers_chauvenet(df, col=col)

# In pandas we can do boolean indexing which will return only those parts where values are true and give values to adjust and replace values with nan
dataset[dataset["gyr_z_outlier"]]

# Making a selection such that when value of gyr_z_outlier is true set value of gyr_z to NaN (this happens in place so we don't need to store result in the variable)
dataset.loc[dataset["gyr_z_outlier"], "gyr_z"] = np.nan

# Peform this transformation in a loop (looping over individual df)
outliers_removed_df = df.copy()     # To create a new final version of dataframe that we'll export later
for col in outlier_columns:
    for label in df["label"].unique():
        dataset = mark_outliers_chauvenet(df[df["label"] == label], col)
        plot_binary_outliers(dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True)

        # Replace values marked as outliers with NaN in the subset
        dataset.loc[dataset[col + "_outlier"], col] = np.nan

        # Update column in the original target output dataframe i.e. outliers_removed_df
        outliers_removed_df.loc[(outliers_removed_df["label"] == label), col] = dataset[col]

        # Counting the number of outliers by subtracting the length of df from the number of NaN values
        n_outliers = len(dataset) - len(dataset[col].dropna())
        print(f"Removed {n_outliers} from {col} for {label}")

outliers_removed_df.info()    # There are some missing values which depicts clearly that some values are marked as NaN, that means our outliers are working correctly 

# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------

outliers_removed_df.to_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")