import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

class Discretization():

    """
    KBinsDiscretizer_Instance examples:
    k3width = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
    k3frequency = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
    """
    @staticmethod
    def discretize(df, KBinsDiscretizer_Instance):
        KBinsDiscretizer_Instance.fit(df.values[:, :-1])
        df_discretized_values = KBinsDiscretizer_Instance.transform(df.values[:, :-1])
        df_discretized_data = pd.DataFrame(df_discretized_values, columns=df.columns[:-1])
        df_discretized_data.insert(len(df_discretized_data.columns), df.columns[-1], df.values[:, -1])
        return df_discretized_data