import pandas as pd
from sklearn.decomposition import PCA

from Preprocessing.utils import Utils


class FeatureSelection():

    @staticmethod
    def get_PCA(df, variance=0.95, debug=True):
        pca = PCA(variance)
        X = df.values[:, :-1]
        pca.fit(X)
        X_reduced = pca.transform(X)
        num_princiapl_components = X_reduced.shape[1]
        if debug:
            print("DEBUG: " + str(num_princiapl_components) + " princiapl components")
        columns = ['PC' + str(i + 1) for i in range(num_princiapl_components)]
        df_reduced = pd.DataFrame(X_reduced, columns=columns)
        df_reduced.insert(len(df_reduced.columns), df.columns[-1], df.values[:, -1])
        return df_reduced

    @staticmethod
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

                score = Utils.automatic_scoring(i_j_df)
                scores.append({"feature_1": i_feature, "feature_2": j_feature, "score": score})
        return scores