import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

df = pd.read_pickle("../../data/interim/03_data_features.pkl")

# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------

# Removing the columns from dataframe that we aren't using right now for predictive models
df_train = df.drop(["participant", "category", "set"], axis=1)

# Splitting data into x and y of the training set
X = df_train.drop("label",axis=1)   # Dataframe without the label
y = df_train["label"]   # Just the label 

# Random state makes sure we get same result & test size 25% means 75% data'll be used as training variables 
X_train,x_test,y_train,y_test= train_test_split(X,y,test_size=0.25,random_state=42)

# Adding startify=y 
X_train_train,x_test,y_train,y_test= train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
# Since we're using a labelled dataset we ofcourse wanna make sure that our traind & test are split in such a way that they contain enough labels of all the instances that we can pick from 
# What we don't want is that all of our training set contains for eg only benchpress and squat data and then our test set contains all the rowing data that is not seen in a training set, so we ofcourse want to ensure that that is not happening and typicallty by default the train test split comes with a shuffle parameter that is set to True by default and this kinds of counters this effect, but stratify is specifically made for these kind of problems where you wanna ensure that there is an equal distribution of all your labels & it is specificaly better at equally distributing the data

# Creating a plot to see stratify in action 
fig, ax = plt.subplots(figsize=(10, 5))
df_train["label"].value_counts().plot(kind="bar", ax=ax, color="lightblue", label="Total")
y_train.value_counts().plot(kind="bar",color="dodgerblue",label="Train")
y_test.value_counts().plot(kind="bar" ,color="royalblue", label = "Test")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Split up different features into subsets
# --------------------------------------------------------------

#features into subset to check if additional features are benefitial that
#we added in feature engineering phase
basic_features =["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]
square_features = ["acc_r","gyr_r"]
pca_features = ["pca_1","pca_2","pca_3"]
#use list to loop over the data frame to get the columns in df 
#list comprehension help access them better

time_features= [f for f in df_train.columns if "_temp_" in f]
#access all columns with _freq_ and _pse_
freq_features=  [f for f in df_train.columns if (("_freq" in f)  or ("_pse" in f))]
cluster_features= ["cluster"]

print("basic_features" , len(basic_features))
print("square_features", len(square_features ))
print("pca_features", len(pca_features))
print("time_features", len(time_features))
print("frequency_features", len(freq_features))
print("cluster_features", len(cluster_features))

#help selecting in data selection and test out model 
feature_set_1 = list(set(basic_features))
feature_set_2 = list(set(basic_features+square_features+pca_features))
feature_set_3 = list(set(feature_set_2+time_features))
feature_set_4 = list(set(feature_set_3 + freq_features+cluster_features))

# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------
 
 #feature selection using decision tree which means loop over
 #all individual features and determine using decision tree the accuracy of our features
 # and add all features to the original best performing features
 #adding more features increases acccuraacy over time
 #once enough features are introduces and slope of accuracy ddecrease
 # at this point , more features does not improve accuracy
 #select 10 best performing features
  
learner=ClassificationAlgorithms()
max_features=10
selected_features, ordered_features, ordered_scores = learner.forward_selection(max_features,X_train, y_train)
 
 
 
plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, max_features + 1, 1), ordered_scores)
plt.xlabel("Number of features")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1, max_features + 1, 1))
plt.show()
 
 #total number of feautures selected vs accuracy on training data
 #initially for feature selection on training data 
 #gives a sense of direction to which give best accuracy 
 #frequency domain give good accuracy 
 
 

# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------


# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Try a simpler model with the selected features
# --------------------------------------------------------------