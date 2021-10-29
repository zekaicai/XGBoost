from XGBoostBaseTree import XGBoostBaseTree
import numpy as np
from Animator import Animator
from DataBinModel import DataBinModel


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def to_type(x):
    # x = self.data_bin.transform(x)
    rows = x.shape[0]
    result = []
    for row in range(rows):
        sample = x[row]
        result.append(1 if sample > 0.5 else 0)
    return np.asarray(result)


def accuracy(y_hat, y):
    total = len(y_hat)
    correct = 0
    for row in range(total):
        if y_hat[row] == y[row]:
            correct += 1
    print('total: ' + str(total))
    print('correct: ' + str(correct))
    return correct / total


def logit(x):
    return np.log(x / (1 - x))


class XGBoostClassifier:
    def __init__(self, num_iters, num_bins, lr, alpha, gamma, max_depth=None, anim=False):
        self.num_iters = num_iters
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

        # 初始的预测值为 0.5
        p_hat = np.full(y.shape, 0.5)
        g = p_hat - y
        h = p_hat * (1 - p_hat)
        y_hat = logit(p_hat)
        if self.anim:
            animator = Animator(xlabel='tree number', xlim=[0, self.num_iters], ylim=[0, 1],
                                legend=['amse'])

        # 依次训练每棵树
        for i in range(self.num_iters):
            curr_tree = XGBoostBaseTree(self.alpha, self.gamma, self.max_depth)
            curr_tree.fit(x, g, h)
            self.trees.append(curr_tree)
            curr_predict = curr_tree.predict(x)
            if i == 0 or i == self.num_iters - 1:
                y_hat += curr_predict
            else:
                y_hat += self.lr * curr_predict
            if self.anim:
                animator.add(i, amse(y, p_hat))
            p_hat = sigmoid(y_hat)
            g = p_hat - y
            h - p_hat * (1 - p_hat)

    def transform(self, x):
        x = self.bin_model.transform(x)
        logodds = np.zeros(x.shape[0])
        for i in range(len(self.trees)):
            if i == 0 or i == len(self.trees) - 1:  # 学习率对结果的衰减不作用于第一棵树和最后一棵树
                logodds += self.trees[i].predict(x)
            else:
                logodds += self.trees[i].predict(x) * self.lr
        # return to_type(sigmoid(logodds))
        return sigmoid(logodds)
        metrics.roc_auc_score

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
    from sklearn import metrics

    dataset = sklearn.datasets.load_breast_cancer()
    data, target = dataset['data'], dataset['target']
    model = XGBoostClassifier(num_iters=3, num_bins=10, lr=0.01, alpha=0.01, gamma=0.01, max_depth=5)
    model.fit(data, target)
    result = model.transform(data)
    print(accuracy(result, target))
    print('auc: ' + str(metrics.roc_auc_score(result, target)))
