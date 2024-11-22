import numpy as np
import pandas as pd
import sys

train_data_filename = sys.argv[1]
val_data_filename = sys.argv[2]
#print(train_data_filename)

train_data = pd.read_csv(train_data_filename)
val_data = pd.read_csv(val_data_filename)

#print(train_data.columns)
#print(val_data.columns)
#print(train_data.head(10))

# MODIFY TRAINING AND VALIDATION DATA


#train_data['season_type_numeric'] = train_data['season_type'].replace({'Pre Season': 0, 'Regular Season': 1, 'Playoffs': 2})
#val_data['season_type_numeric'] = val_data['season_type'].replace({'Pre Season': 0, 'Regular Season': 1, 'Playoffs': 2})
#train_data = pd.get_dummies(train_data, columns=['season_type'])
#val_data = pd.get_dummies(val_data, columns=['season_type'])

#train_data = pd.get_dummies(train_data, columns=['team_abbreviation_home'])
#val_data = pd.get_dummies(val_data, columns=['team_abbreviation_home'])

train_data['w_count_home_avg5'] = train_data['home_wl_pre5'].apply(lambda x: x.lower().count('w'))
val_data['w_count_home_avg5'] = val_data['home_wl_pre5'].apply(lambda x: x.lower().count('w'))
train_data['w_count_away_avg5'] = train_data['away_wl_pre5'].apply(lambda x: x.lower().count('w'))
val_data['w_count_away_avg5'] = val_data['away_wl_pre5'].apply(lambda x: x.lower().count('w'))

#train_data["AST/TOV_home_avg5"] = train_data["ast_home_avg5"] / train_data["tov_home_avg5"]
#val_data["AST/TOV_home_avg5"] = val_data["ast_home_avg5"] / val_data["tov_home_avg5"]
#train_data["AST/TOV_away_avg5"] = train_data["ast_away_avg5"] / train_data["tov_away_avg5"]
#val_data["AST/TOV_away_avg5"] = val_data["ast_away_avg5"] / val_data["tov_away_avg5"]

new_train_data = pd.DataFrame()
new_val_data = pd.DataFrame()

"""
stats_home = ['fg_pct', 'fg3_pct', 'ft_pct', 'reb', 'dreb', 'stl', 'blk', 'tov', 'pf', 'pts', 'w_count']
for stat in stats_home:
    home_col = f'{stat}_home_avg5'
    away_col = f'{stat}_away_avg5'
    train_data[stat + '_diff'] = train_data[home_col] - train_data[away_col]
    val_data[stat + '_diff'] = val_data[home_col] - val_data[away_col]
new_train_data["label"] = train_data["label"]
if (val_data_filename == "validation_data.csv"):
    new_val_data["label"] = val_data["label"]
"""
#train_data = new_train_data
#val_data = new_val_data


#train_data = train_data.drop(["team_abbreviation_home", "team_abbreviation_away", "season_type","away_wl_pre5", "home_wl_pre5", "min_avg5", "oreb_away_avg5", "oreb_home_avg5", "ast_home_avg5", "ast_away_avg5"], axis=1, errors='ignore')
#val_data = val_data.drop(["team_abbreviation_home", "team_abbreviation_away", "season_type", "away_wl_pre5", "home_wl_pre5", "min_avg5", "oreb_away_avg5", "oreb_home_avg5", "ast_home_avg5", "ast_away_avg5"], axis=1, errors='ignore')

#train_data = train_data.drop(["team_abbreviation_home", "team_abbreviation_away", "season_type","away_wl_pre5", "home_wl_pre5", "min_avg5", "fg_pct_home_avg5", "fg3_pct_home_avg5","ft_pct_home_avg5", "fg_pct_away_avg5", "fg3_pct_away_avg5","ft_pct_away_avg5"], axis=1, errors='ignore')
#val_data = val_data.drop(["team_abbreviation_home", "team_abbreviation_away", "season_type", "away_wl_pre5", "home_wl_pre5", "min_avg5", "fg_pct_home_avg5", "fg3_pct_home_avg5","ft_pct_home_avg5", "fg_pct_away_avg5", "fg3_pct_away_avg5","ft_pct_away_avg5"], axis=1, errors='ignore')

#train_data = train_data.drop(["AST/TOV_home_avg5", "AST/TOV_away_avg5"], axis=1, errors='ignore')
#val_data = val_data.drop(["AST/TOV_home_avg5", "AST/TOV_away_avg5"], axis=1, errors='ignore')

train_data = train_data.drop(["team_abbreviation_home", "team_abbreviation_away", "season_type","away_wl_pre5", "home_wl_pre5"], axis=1, errors='ignore')
val_data = val_data.drop(["team_abbreviation_home", "team_abbreviation_away", "season_type", "away_wl_pre5", "home_wl_pre5"], axis=1, errors='ignore')

#train_data["weight"] = train_data["fg_pct_home_avg5"]
#val_data["weight"] = val_data["fg_pct_home_avg5"]
#train_data["weight2"] = train_data["fg_pct_away_avg5"]
#val_data["weight2"] = val_data["fg_pct_away_avg5"]

train_label = train_data["label"]
#val_label = val_data["label"]

train_data = train_data.apply(lambda x: (x - x.mean()) / x.std())
val_data = val_data.apply(lambda x: (x - x.mean()) / x.std())

train_data["label"] = train_label
#val_data["label"] = val_label

#print(train_data.head(10))



X_val_data = val_data.drop("label", axis=1, errors='ignore').values
if (val_data_filename == "validation_data.csv"):
    Y_val_data = val_data["label"].values

#print (train_data.head(10))


def probability_of_y(df, Y):
    # take a list of the possible classes, in this case it is 1 or 0
    classes = sorted(list(df[Y].unique()))
    prior = []

    # calculate the P(Y = y)
    for i in classes:
        prior.append(len(df[df[Y]==i])/len(df))
    return prior

def calculate_mean_std(df, Y, classes, features):
    mean_table = np.zeros((len(classes), len(features)))
    std_table = np.zeros((len(classes), len(features)))
    for c_idx,c in enumerate(classes):
        new_df = df[df[Y] == c]
        for feat_idx, feat in enumerate(features):
            mean_table[c_idx][feat_idx] = new_df[feat].mean()
            std_table[c_idx][feat_idx] = new_df[feat].std()

    return mean_table, std_table


# calculate the probability of X = x given y using the gaussian formula
def gaussian_x_given_y(mean, std, feat_val, feat_name):
    # gaussian formula
    #epsilon = 0.1
    #std = max(std, epsilon)

    p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((feat_val - mean)**2 / (2 * std**2)))

    #var = std**2
    #numerator = np.exp(-((feat_val - mean) ** 2) / (2 * var))
    #denominator = np.sqrt(2 * np.pi * var)
    #p_x_given_y = numerator / denominator

    #print(feat_name + ": " + str(p_x_given_y) + " std: " + str(std))
    #if ("g_pc" in feat_name):
    #    print(p_x_given_y)
    #    p_x_given_y *= 120
    #    print(p_x_given_y)
    return p_x_given_y

def naive_bayes_calculation(df, X, Y):
    #names of all columns except first
    features = list(df.drop(Y, axis=1).columns)
    labels = sorted(list(df[Y].unique()))

    prior = probability_of_y(df, Y)

    Y_prediction = []

    # length of test columns and train columns
    #print(len(X[0]))
    #print(len(df.iloc[0]))

    mean_table, std_table = calculate_mean_std(df, Y, labels, features)


    for x in X:
        #calculate likelihood
        likelihood = np.zeros(len(labels))
        for j in range(len(labels)):
            for i in range(len(features)):
                mean = mean_table[j][i]
                std = std_table[j][i]
                likelihood[j] += np.log(gaussian_x_given_y(mean, std, x[i], features[i]))

        post_prob = [1]*len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] + np.log(prior[j])

        #print(post_prob)
        Y_prediction.append(np.argmax(post_prob))

    return np.array(Y_prediction)




Y_prediction = naive_bayes_calculation(train_data, X=X_val_data, Y="label")


if (val_data_filename == "validation_data.csv"):
    from sklearn.metrics import confusion_matrix, f1_score
    print(confusion_matrix(Y_val_data, Y_prediction))
    print(f1_score(Y_val_data, Y_prediction))

    print(Y_prediction[0:30])
    print(Y_val_data[0:30])
else:
    for i in Y_prediction:
        print(i)
