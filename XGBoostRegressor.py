from XGBoostBaseTree import XGBoostBaseTree
import numpy as np
from Animator import Animator
from DataBinModel import DataBinModel


def amse(y, y_hat):
    return np.sum((y-y_hat)**2) / len(y)


class XGBoostRegressor:
    def __init__(self, num_iters, num_bins, lr, alpha, gamma, max_depth=None, anim=False):
        self.num_iters = num_iters
        self.num_bins = num_bins
        self.bin_model = DataBinModel(num_bins)
        self.lr = lr
        self.alpha = alpha
        self.gamma = gamma
        self.max_depth = max_depth
        self.trees = []
        self.anim = anim

    def fit(self, x, y):

        # 对 x 分箱
        self.bin_model.fit(x)
        x = self.bin_model.transform(x)

        y_hat = np.zeros(y.shape)
        g = y_hat - y
        h = np.ones(y.shape)
        if self.anim:
            animator = Animator(xlabel='tree number', xlim=[0, self.num_iters], ylim=[0, 1],
                                legend=['amse'])

        # 依次训练每棵树
        for i in range(self.num_iters):
            curr_tree = XGBoostBaseTree(self.alpha, self.gamma, self.max_depth)
            curr_tree.fit(x, g, h)
            self.trees.append(curr_tree)
            curr_predict = curr_tree.predict(x)
            y_hat += self.lr * curr_predict
            if self.anim:
                animator.add(i, amse(y, y_hat))
            g = y_hat - y

    def transform(self, x):
        x = self.bin_model.transform(x)
        result = np.zeros(x.shape[0])
        for i in range(len(self.trees)):
            result += self.trees[i].predict(x) * self.lr
        return result

    def curr_loss(self, x, y):
        y_hat = np.zeros(x.shape[0])
        penalty = 0
        #  遍历当前训练出的树
        for i in range(len(self.trees)):
            y_hat += self.trees[i].predict(x) * self.lr
            curr_penalty = self.trees[i].penalty()
            penalty += curr_penalty
        cost = amse(y, y_hat)
        return cost + penalty


if __name__ == '__main__':
    import sklearn.datasets
    dataset = sklearn.datasets.load_boston()
    data, target = dataset['data'], dataset['target']
    model = XGBoostRegressor(num_iters=3, num_bins=10, lr=1, alpha=0.01, gamma=0.01, max_depth=10)
    model.fit(data, target)
    result = model.transform(data)
    print(amse(result, target))
    print(model.curr_loss(data, target))