from imblearn.over_sampling import BorderlineSMOTE  # doctest: +NORMALIZE_WHITESPACE
from imblearn.over_sampling import SMOTE  # doctest: +NORMALIZE_WHITESPACE
from imblearn.over_sampling import RandomOverSampler  # doctest: +NORMALIZE_WHITESPACE

from imblearn.under_sampling import EditedNearestNeighbours  # doctest: +NORMALIZE_WHITESPACE
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import TomekLinks  # doctest: +NORMALIZE_WHITESPACE
from imblearn.under_sampling import RandomUnderSampler  # doctest: +NORMALIZE_WHITESPACE

from imblearn.over_sampling import ADASYN
from collections import Counter
import pandas as pd

class Unbalanced():
    class OverSampling():
        """
        Links: https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.BorderlineSMOTE.html
        """
        @staticmethod
        def Borderline_SMOTE(df, variant=1, debug=True):
            X = df.values[:, :-1]
            y = df.values[:, -1].astype(int)
            if debug:
                print('borderline_SMOTE: Original dataset shape %s' % Counter(y))
            if variant == 1:
                sm = BorderlineSMOTE(random_state=0, kind="borderline-1")
            else:
                sm = BorderlineSMOTE(random_state=0, kind="borderline-2")

            X_res, y_res = sm.fit_resample(X, y)
            df_resampled = pd.DataFrame(X_res, columns=df.columns[:-1])
            df_resampled.insert(len(df_resampled.columns), df.columns[-1], y_res)
            if debug:
                print('borderline_SMOTE: Resampled dataset shape %s' % Counter(y_res))
            return df_resampled
        ###########################################################################################################
        """
        Link: https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html
        """
        @staticmethod
        def SMOTE(df, debug=True):
            X = df.values[:, :-1]
            y = df.values[:, -1].astype(int)
            if debug:
                print('SMOTE: Original dataset shape %s' % Counter(y))
            sm = SMOTE(random_state=0)
            X_res, y_res = sm.fit_resample(X, y)
            df_resampled = pd.DataFrame(X_res, columns=df.columns[:-1])
            df_resampled.insert(len(df_resampled.columns), df.columns[-1], y_res)
            if debug:
                print('SMOTE: Resampled dataset shape %s' % Counter(y_res))
            return df_resampled
        ###########################################################################################################
        def ADASYN(df, debug=True):
            X = df.values[:, :-1]
            y = df.values[:, -1].astype(int)
            if debug:
                print('ADASYN: Original dataset shape %s' % Counter(y))
            ada = ADASYN(random_state=0)
            X_res, y_res = ada.fit_resample(X, y)
            df_resampled = pd.DataFrame(X_res, columns=df.columns[:-1])
            df_resampled.insert(len(df_resampled.columns), df.columns[-1], y_res)
            if debug:
                print('ADASYN: Resampled dataset shape %s' % Counter(y_res))
            return df_resampled
        ###########################################################################################################
        """
        Link: https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.RandomOverSampler.html
        """
        def Random(df, debug=True):
            X = df.values[:, :-1]
            y = df.values[:, -1].astype(int)
            if debug:
                print('Original dataset shape %s' % Counter(y))
            ros = RandomOverSampler(random_state=0)
            X_res, y_res = ros.fit_resample(X, y)
            df_resampled = pd.DataFrame(X_res, columns=df.columns[:-1])
            df_resampled.insert(len(df_resampled.columns), df.columns[-1], y_res)
            if debug:
                print('Resampled dataset shape %s' % Counter(y_res))
            return df_resampled



    class UnderSampling():
        """
        Link: https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.NearMiss.html
        """
        @staticmethod
        def NearMiss(df, variant=2, debug=True):
            X = df.values[:, :-1]
            y = df.values[:, -1].astype(int)
            if debug:
                print('Original dataset shape %s' % Counter(y))
            nm = NearMiss(random_state=0, version=variant)
            X_res, y_res = nm.fit_resample(X, y)
            df_resampled = pd.DataFrame(X_res, columns=df.columns[:-1])
            df_resampled.insert(len(df_resampled.columns), df.columns[-1], y_res)
            if debug:
                print('Resampled dataset shape %s' % Counter(y_res))
            return df_resampled
        ###########################################################################################################
        def ENN(df, debug=True):
            X = df.values[:, :-1]
            y = df.values[:, -1].astype(int)
            if debug:
                print('Original dataset shape %s' % Counter(y))
            enn = EditedNearestNeighbours(sampling_strategy="auto")
            X_res, y_res = enn.fit_resample(X, y)
            df_resampled = pd.DataFrame(X_res, columns=df.columns[:-1])
            df_resampled.insert(len(df_resampled.columns), df.columns[-1], y_res)
            if debug:
                print('Resampled dataset shape %s' % Counter(y_res))
            return df_resampled
        ###########################################################################################################
        """
        Link: https://imbalanced-learn.org/en/stable/generated/imblearn.under_sampling.TomekLinks.html
        """
        def Tomeks_Link(df, debug=True):
            X = df.values[:, :-1]
            y = df.values[:, -1].astype(int)
            if debug:
                print('Original dataset shape %s' % Counter(y))
            tl = TomekLinks()
            X_res, y_res = tl.fit_resample(X, y)
            df_resampled = pd.DataFrame(X_res, columns=df.columns[:-1])
            df_resampled.insert(len(df_resampled.columns), df.columns[-1], y_res)
            if debug:
                print('Resampled dataset shape %s' % Counter(y_res))
            return df_resampled
        ###########################################################################################################
        """
        Link: https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.RandomUnderSampler.html
        """
        def Random(df, debug=True):
            X = df.values[:, :-1]
            y = df.values[:, -1].astype(int)
            if debug:
                print('Original dataset shape %s' % Counter(y))
            rus = RandomUnderSampler(random_state=0)
            X_res, y_res = rus.fit_resample(X, y)
            df_resampled = pd.DataFrame(X_res, columns=df.columns[:-1])
            df_resampled.insert(len(df_resampled.columns), df.columns[-1], y_res)
            if debug:
                print('Resampled dataset shape %s' % Counter(y_res))
            return df_resampled