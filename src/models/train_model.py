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


# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/03_data_features.pkl")


#removing the columns that we are not using
df_train = df.drop(["participant", "category", "set"], axis=1)

#splitting data into x and y of the training set
#here axis parameter is used to specify wether operation is
#done along rows or columns
X=df_train.drop("label",axis=1)

#this just gives the label
y=df_train["label"]

#trained test split is taken from sckit learn lib 
#random state  makes sure we get the same result
X_train,x_test,y_train,y_test= train_test_split(X,y,test_size=0.25,random_state=42)

#adding startify=y 
X_train_train,x_test,y_train,y_test= train_test_split(
    X,y,test_size=0.25,random_state=42,stratify=y
    )
#make sure that they are split such that they contain enough 
#instances to train on for eg we dont want that are training data has only row and bench 
#and test data has only squat and ohp data . Shuffle parameter counters this effect,
#but stratify is better at equally distributing the data

y_train.value_counts().plot(kind="bar",color="dodgerblue",label="Train")
y_test.value_counts().plot(kind="bar" ,color="royalblue", label = "Test")
plt.legend()
plt.show()



# --------------------------------------------------------------
# Split feature subsets
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
 
 
a
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