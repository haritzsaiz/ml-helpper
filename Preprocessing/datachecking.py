import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.covariance import EllipticEnvelope
import numpy as np

class DataChecking():

    """
    ALERT = IT MODIFIES THE DF argument (it may drop some rows). It only find NaN values. It wont Drop Nulls

    action{remove|impute}
    """
    @staticmethod
    def missig_values(df, action="remove", debug=True):
        if debug:
            print("DEBUG - Pre action: Variable Count")
            print(df.count())

        if action == "remove":
            df.dropna(axis=0, how='any', inplace=True)
        elif action == "impute":
            columns = df.columns
            # Saca todas las caregorias de Y
            categories = df[df.columns[-1]].unique()
            # Cambia las categorias por numeros
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


    """
    ALERT = IT MODIFIES THE DF argument (it may drop some rows)

    action{parallel|colective|individual}
    if remove == True then it will drop the outliers
    """
    @staticmethod
    def outliers(df, action="parallel", debug=True, remove=True):
        if action == "colective":
            columns = df.columns
            # Saca todas las caregorias de Y
            categories = df[df.columns[-1]].unique()
            # Cambia las categorias por numeros
            for i in range(len(categories)):
                df[df.columns[-1]].replace(categories[i], i, inplace=True)
            elip_env = EllipticEnvelope().fit(df)
            detection = elip_env.predict(df)
            # Outilers using Mahalanobis distance.
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
                        print("outliers for variable ['" + str(columns[i]) + "'] = " + str(outlier_positions_box))

            if remove:
                df.drop(df.index[outlier_positions_box], inplace=True)
            return all_outliers_positions_box

        elif action == "parallel":
            outlier_positions_mah = DataChecking.outliers(df, action="colective", remove=False)
            outlier_positions_box = DataChecking.outliers(df, action="individual", remove=False)
            outliers_position = list(np.sort(outlier_positions_mah + outlier_positions_box))
            if remove:
                df.drop(df.index[outliers_position], inplace=True)
            return outliers_position