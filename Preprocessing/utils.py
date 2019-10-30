from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

class Utils():
    @staticmethod
    def automatic_scoring(df):
        algorithm = DecisionTreeClassifier()
        score = cross_val_score(estimator=algorithm, X=df.values[:, :-1], y=df.values[:, -1], cv=5, scoring='f1_macro')
        summary_score = score.mean()
        return summary_score