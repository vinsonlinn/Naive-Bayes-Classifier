import numpy as np
import pandas as pd
import sys

train_data_filename = sys.argv[1]
val_data_filename = sys.argv[2]
#print(train_data_filename)

train_data = pd.read_csv(train_data_filename)
val_data = pd.read_csv(val_data_filename)
#print(train_data.head(10))

# MODIFY TRAINING AND VALIDATION DATA
train_data = train_data.drop(["team_abbreviation_home", "team_abbreviation_away", "season_type", "away_wl_pre5", "home_wl_pre5"], axis=1)

val_data = val_data.drop(["team_abbreviation_home", "team_abbreviation_away", "season_type", "away_wl_pre5", "home_wl_pre5"], axis=1)
X_val_data = val_data.iloc[:, 1:].values
Y_val_data = val_data.iloc[:, 0].values


def probability_of_y(df, Y):
    # take a list of the possible classes, in this case it is 1 or 0
    classes = sorted(list(df[Y].unique()))
    prior = []

    # calculate the P(Y = y)
    for i in classes:
        prior.append(len(df[df[Y]==i])/len(df))
    return prior

# calculate the probability of X = x given y using the gaussian formula
def gaussian_x_given_y(df, feat_name, feat_val, Y, label):
    feat = list(df.columns)
    df = df[df[Y]==label]
    # get the mean and standard deviation for the feature for the current label
    mean, std = df[feat_name].mean(), df[feat_name].std()
    
    # gaussian formula
    p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((feat_val - mean)**2 / (2 * std**2)))
    return p_x_given_y

def naive_bayes_calculation(df, X, Y):
    #names of all columns except first
    features = list(df.columns)[1:]

    prior = probability_of_y(df, Y)

    Y_prediction = []

    for x in X:
        #calculate likelihood
        labels = sorted(list(df[Y].unique()))
        likelihood = [1]*len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] += gaussian_x_given_y(df, features[i], x[i], Y, labels[j])

        post_prob = [1]*len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]

        #print(post_prob)
        Y_prediction.append(np.argmax(post_prob))

    return np.array(Y_prediction)




Y_prediction = naive_bayes_calculation(train_data, X=X_val_data, Y="label")

for i in Y_prediction:
    print(i)