# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import inv, pinv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs

class pcaT2():
    def __init__(self, param_dict={'explained_variance_': 80, 'multiple_threshold': 3, 'warning_threshold': 1.5}):
        try:
            self.explained_variance_ = param_dict['explained_variance_']
            self.multiple_threshold = param_dict['multiple_threshold']
            self.warning_threshold = param_dict['warning_threshold']
        except:
            self.explained_variance_ = 0.8
            self.multiple_threshold = 9
            self.warning_threshold = 3

        for key, value in param_dict.items():
            ### charactor existed checking
            if hasattr(self, key):
                ### charactor for PCA_T_square finction setting
                setattr(self, key, value)
                # print(key, value)
    def fit(self, X, y=None):
        self.std_ = StandardScaler()
        X_std = self.std_.fit_transform(X)
        self.pca = PCA(self.explained_variance_)
        U, S, V = self.pca._fit(X_std)
        self.n_components = self.pca.n_components_
        self.eigenvalues = S[:self.n_components]
        self._caculate_threshold(X_std)
        return self
    def predict(self, X):
        X_std = self.std_.transform(X)
        is_inlier = np.ones(X_std.shape[0], dtype=int)
        is_inlier[(self._decision_function(X_std) >= self.anomaly_upper_threshold)] = -1
        is_inlier[(self._decision_function(X_std) <= self.anomaly_lower_threshold)] = -1
        return is_inlier
    def caculate_threshold(self, X):
        X_std_ = self.std_.transform(X)
        anomaly_upper_threshold, anomaly_lower_threshold, anomaly_warning_threshold = self._caculate_threshold(X_std_)
        return anomaly_upper_threshold, anomaly_lower_threshold, anomaly_warning_threshold
    def _caculate_threshold(self, X):
        T_square = self._decision_function(X)
        th_T2 = np.median(T_square) + self.multiple_threshold * np.std(T_square)
        th_T2_warning = np.median(T_square) + self.warning_threshold * np.std(T_square)
        self.anomaly_upper_threshold = th_T2
        self.anomaly_upper_threshold = th_T2
        self.anomaly_lower_threshold = 0
        self.anomaly_warning_threshold = th_T2_warning
        return self.anomaly_upper_threshold, self.anomaly_lower_threshold, self.anomaly_warning_threshold
    def decision_function(self, X):
        X_std_ = self.std_.transform(X)
        decision_value = self._decision_function(X_std_)
        return decision_value
    def _decision_function(self, X):
        pca_loadings = np.matrix(self.pca.components_)
        new_eigenvalues = self.eigenvalues
        # avoid eigenvalues being zero
        if np.sum(new_eigenvalues) == 0:
            new_eigenvalues = 1.5e-04 * np.ones(len(new_eigenvalues))
        try:
            # Singular Matrix
            T_square_matrix = np.transpose(
                np.transpose(pca_loadings) * np.matrix(np.sqrt(inv(np.diag(new_eigenvalues)))) *
                pca_loadings * np.transpose(X))
        except:
            T_square_matrix = np.transpose(np.transpose(pca_loadings) *
                                           np.matrix(np.sqrt(pinv(np.diag(new_eigenvalues)))) *
                                           pca_loadings * np.transpose(X))
        T_square = np.linalg.norm(T_square_matrix, axis=1) ** 2
        return T_square