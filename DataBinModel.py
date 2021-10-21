import numpy as np


class DataBinModel:
    """
    对数据的每一列进行分箱
    """
    def __init__(self, bin_num):
        self.bin_num = bin_num
        self.feature_quantile_map = None

    def fit(self, data):
        feature_quantile_map = {}
        num_features = data.shape[1]
        for index in range(num_features):
            feature = data[:, index]
            quantiles = []
            for percent in range(1, self.bin_num):
                quantile = np.percentile(feature, 1.0*percent/self.bin_num * 100.0)
                quantiles.append(quantile)
            feature_quantile_map[index] = quantiles
        self.feature_quantile_map = feature_quantile_map

    def transform(self, data):
        num_features = data.shape[1]
        return np.asarray([np.digitize(data[:, index], self.feature_quantile_map[index]) for index in range(num_features)]).T
