from XGBoostBaseTree import XGBoostBaseTree
import numpy as np
from Animator import Animator


def amse(y, y_hat):
    return np.sum((y-y_hat)**2) / len(y)


class XGBoostRegressor:
    def __init__(self, num_iters, num_bins, lr, alpha, gamma, max_depth=None, anim=False):
        self.num_iters = num_iters
        self.num_bins = num_bins
        self.lr = lr
        self.alpha = alpha
        self.gamma = gamma
        self.max_depth = max_depth
        self.trees = []
        self.anim = anim

    def fit(self, x, y):
        y_hat = np.zeros(y.shape)
        residual = y - y_hat
        if self.anim:
            animator = Animator(xlabel='tree number', xlim=[0, self.num_iters], ylim=[0, 1],
                                legend=['amse'])
        # cost_arr = []
        # loss_arr = []
        # TODO：xgboost 不是直接拟合残差的，是传入当前的 g 和 h 给下一棵树去拟合
        for i in range(self.num_iters):
            curr_tree = XGBoostBaseTree(self.num_bins, self.alpha, self.gamma, self.max_depth)
            curr_tree.fit(x, residual)
            self.trees.append(curr_tree)
            curr_predict = curr_tree.predict(x)
            y_hat += self.lr * curr_predict
            residual = y - y_hat
            if self.anim:
                animator.add(i, amse(y, y_hat))
            # cost, loss = self.curr_loss(x, y)
            # cost_arr.append(cost)
            # loss_arr.append(loss)
        # return cost_arr, loss_arr

    def transform(self, x):
        result = np.zeros(x.shape[0])
        for i in range(self.num_iters):
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