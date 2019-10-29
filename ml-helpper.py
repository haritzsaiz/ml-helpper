from main import *

import pandas as pd
from sklearn.impute import SimpleImputer

"""
ALERT = IT MODIFIES THE DF argument (it may drop some rows)

action{remove|impute}
"""
def treate_missing_values(df, action="remove", debug=True):
    if debug:
        print("DEBUG - Pre action: Variable Count")
        print(df.count())

    if action == "remove":
        df.dropna(axis=0, how='any', inplace=True)
    elif action == "impute":
        columns = df.columns
        #Saca todas las caregorias de Y
        categories = df[df.columns[-1]].unique()
        #Cambia las categorias por numeros
        for i in range(len(categories)):
            df[df.columns[-1]].replace(categories[i], i, inplace=True)
        imp = SimpleImputer(strategy='mean')
        df = pd.DataFrame(data=imp.fit_transform(imp.fit_transform(df)))
        # Cambia numeros por las categorias
        df.columns = columns
        for i in range(len(categories)):
            df[df.columns[-1]].replace(i, categories[i], inplace=True)

    if debug:
        print("\nDEBUG - Post action: Variable Count")
        print(df.count())

    return df


########################################################################################################################
import pandas as pd
from sklearn.covariance import EllipticEnvelope
import numpy as np

"""
ALERT = IT MODIFIES THE DF argument (it may drop some rows)

action{parallel|colective|individual}
if remove == True then it will drop the outliers
"""
def treate_outliers(df, action="parallel", debug=True, remove=True):
    if action == "colective":
        columns = df.columns
        # Saca todas las caregorias de Y
        categories = df[df.columns[-1]].unique()
        # Cambia las categorias por numeros
        for i in range(len(categories)):
            df[df.columns[-1]].replace(categories[i], i, inplace=True)
        elip_env =  EllipticEnvelope().fit(df)
        detection = elip_env.predict(df)
        #Outilers using Mahalanobis distance.
        outlier_positions_mah = [x for x in range(df.shape[0]) if detection[x] == -1]
        if remove:
            df.drop(df.index[outlier_positions_mah], inplace=True)
        return outlier_positions_mah

    elif action == "individual":
        all_outliers_positions_box = []
        columns = df.columns
        _, bp = pd.DataFrame.boxplot(df, return_type='both')
        outliers = [flier.get_ydata() for flier in bp["fliers"]]
        for i in range(len(outliers)):
            prop_outliers = outliers[i]
            if prop_outliers.size > 0:

                IQR = df.describe()[columns[i]]["75%"] - df.describe()[columns[i]]["25%"]
                whiskers = [df.describe()[columns[i]]["25%"] - (1.5 * IQR),
                            df.describe()[columns[i]]["75%"] + (1.5 * IQR)]
                outlier_positions_box = [x for x in range(df.shape[0]) if
                                         df[columns[i]].values[x] < whiskers[0] or df[columns[i]].values[
                                             x] > whiskers[1]]
                all_outliers_positions_box += outlier_positions_box
                if debug:
                    print("outliers for variable ['" + str(columns[i]) + "'] = "  + str(outlier_positions_box))

        if remove:
            df.drop(df.index[outlier_positions_box], inplace=True)
        return all_outliers_positions_box

    elif action == "parallel":
        outlier_positions_mah = treate_outliers(df, action="colective", remove=False)
        outlier_positions_box = treate_outliers(df, action="individual", remove=False)
        outliers_position = list(np.sort(outlier_positions_mah + outlier_positions_box))
        if remove:
            df.drop(df.index[outliers_position], inplace=True)
        return outliers_position

########################################################################################################################
import pandas as pd

"""
KBinsDiscretizer_Instance examples:
k3width = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
k3frequency = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
"""
def discretizer(df, KBinsDiscretizer_Instance):
    KBinsDiscretizer_Instance.fit(df.values[:, :-1])
    df_discretized_values = KBinsDiscretizer_Instance.transform(df.values[:, :-1])
    df_discretized_data = pd.DataFrame(df_discretized_values, columns=df.columns[:-1])
    df_discretized_data.insert(len(df_discretized_data.columns), df.columns[-1], df.values[:, -1])
    return df_discretized_data

########################################################################################################################
import pandas as pd
from sklearn.decomposition import PCA
"""
PCA_Instance example:
pca = PCA(n_components=0.99)
"""
def get_PCA_from_df(df, PCA_Instance, debug=True):
    X = df.values[:, :-1]
    PCA_Instance.fit(X)
    X_reduced = PCA_Instance.transform(X)
    num_princiapl_components = X_reduced.shape[1]
    if debug:
        print("DEBUG: " + str(num_princiapl_components) + " princiapl components")
    columns = ['PC' + str(i + 1) for i in range(num_princiapl_components)]
    df_reduced = pd.DataFrame(X_reduced, columns=columns)
    df_reduced.insert(len(df_reduced.columns), df.columns[-1], df.values[:, -1])
    return df_reduced

########################################################################################################################
def get_score_2_features_subset(df):
    scores = []
    features = df.columns[:-1]
    y = df.columns[-1]
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            i_feature = features[i]
            j_feature = features[j]

            X_i_feature = df[i_feature]
            X_j_feature = df[j_feature]
            x_i_j_values = {i_feature: X_i_feature, j_feature: X_j_feature}
            i_j_df = pd.DataFrame(data=x_i_j_values)
            i_j_df.insert(len(i_j_df.columns), df.columns[-1], df.values[:, -1])

            score = automatic_scoring(i_j_df)
            scores.append({"feature_1": i_feature, "feature_2": j_feature, "score": score})
    return scores
########################################################################################################################

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

def automatic_scoring(df):
    algorithm = DecisionTreeClassifier()
    score = cross_val_score(estimator=algorithm, X=df.values[:, :-1], y=df.values[:, -1], cv=5, scoring='f1_macro')
    summary_score = score.mean()
    return summary_score


########################################################################################################################
#IGNORE THIS
init_data = pd.read_csv('notebooks/nan_data.csv')
df = treate_missing_values(init_data, action="impute")
outliers_position = treate_outliers(df, action="parallel")
print(outliers_position)
########################################################################################################################
