from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

class FeatureSelection:
    """
    Link: https://scikit-learn.org/stable/modules/feature_selection.html
    """
    @staticmethod
    def select_features(df, debug=True):
        selected_features = []
        X = df.values[:, :-1]
        y = df.values[:, -1]
        clf = ExtraTreesClassifier(n_estimators=100)
        clf = clf.fit(X, y)
        model = SelectFromModel(clf, prefit=True)
        model.transform(X)
        features = df.columns[:-1]
        feature_importance = clf.feature_importances_
        selected_features_indices = model.get_support(indices=True)
        for i in range(len(selected_features_indices)):
            selected_features.append(features[selected_features_indices[i]])

        if debug:
            print("Importance of each variable: ")
            for i in range(len(feature_importance)):
                print("['"+ str(features[i])+"'] = " + str(feature_importance[i]))

        return selected_features