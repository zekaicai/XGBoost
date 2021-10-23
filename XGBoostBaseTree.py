from DataBinModel import DataBinModel
import numpy as np


class Queue:
    def __init__(self):
        self.q = []

    def size(self):
        return len(self.q)

    def add(self, obj):
        self.q.insert(0, obj)

    def pop(self):
        assert self.size() > 0
        return self.q.pop()


class XGBoostBaseTreeNode:
    def __init__(self, left_child=None, right_child=None, split_col_index=None, split_point=None):
        self.left_child = left_child
        self.right_child = right_child
        self.split_col_index = split_col_index
        self.split_point = split_point

    def is_leave_node(self):
        return self.left_child is None and self.right_child is None


class XGBoostBaseTree:
    def __init__(self, alpha, gamma, max_depth=None, bin_model=DataBinModel(10)):
        self.max_depth = max_depth
        self.alpha = alpha
        self.gamma = gamma
        self.root = None
        self.bin_model = bin_model

    def _score(self, g, h):
        G = np.sum(g)
        H = np.sum(h)
        return -0.5*G**2/(H+self.alpha) + self.gamma

    def _build_tree(self, curr_node, x, g, h, curr_depth):
        """
        Build tree from current node.
        :param curr_node:
        :return:
        """
        rows, cols = x.shape
        G = np.sum(g)
        H = np.sum(h)
        curr_node.y_hat = -G / (H + self.alpha)
        curr_node.score = self._score(g, h)
        curr_node.depth = curr_depth
        curr_node.num_samples = rows

        if self.max_depth is not None and curr_depth >= self.max_depth:
            # self.num_leaves += 1
            # self.leaves_weight.append(curr_node.y_hat)
            return

        best_col_index = None
        best_split_point = None
        best_gain = 0

        for col_index in range(cols):  # for each feature
            for split_point in sorted(set(x[:, col_index])): # for each split point
                left_indices = np.where(x[:, col_index] <= split_point)
                right_indices = np.where(x[:, col_index] > split_point)
                left_score = self._score(g[left_indices], h[left_indices])
                right_score = self._score(g[right_indices], h[right_indices])
                gain = curr_node.score - left_score - right_score

                if gain > best_gain:
                    best_gain = gain
                    best_col_index = col_index
                    best_split_point = split_point

        # 如果可以继续分裂
        if best_gain == 0:
            # self.num_leaves += 1
            # self.leaves_weight.append(curr_node.y_hat)
            return

        curr_node.split_col_index = best_col_index
        curr_node.split_point = best_split_point
        left_indices = np.where(x[:, curr_node.split_col_index] <= curr_node.split_point)
        # print("length of left child: %d" % left_indices[0].shape[0])
        # print(left_indices)
        right_indices = np.where(x[:, curr_node.split_col_index] > curr_node.split_point)
        # print("length of right child: %d" % right_indices[0].shape[0])
        # print(right_indices)

        curr_node.left_child = XGBoostBaseTreeNode()
        self._build_tree(curr_node.left_child, x[left_indices], g[left_indices], h[left_indices], curr_depth+1)

        curr_node.right_child = XGBoostBaseTreeNode()
        self._build_tree(curr_node.right_child, x[right_indices], g[right_indices], h[right_indices], curr_depth+1)

    def fit(self, x, y):
        self.root = XGBoostBaseTreeNode()
        init_y_hat = np.zeros(y.shape)
        g = init_y_hat - y
        h = np.ones(y.shape)
        # self.data_bin.fit(x)
        self._build_tree(self.root, x, g, h, 1)

    def _predict_sample(self, x):
        curr_node = self.root
        while not curr_node.is_leave_node():
            split_col_index = curr_node.split_col_index
            split_point = curr_node.split_point
            if x[split_col_index] <= split_point:
                curr_node = curr_node.left_child
            else:
                curr_node = curr_node.right_child
        return curr_node.y_hat

    def predict(self, x):
        # x = self.data_bin.transform(x)
        rows = x.shape[0]
        result = []
        for row in range(rows):
            sample = x[row]
            y_hat = self._predict_sample(sample)
            result.append(y_hat)
        return np.asarray(result)

    def penalty(self):
        q = Queue()
        q.add(self.root)
        num_leaves = 0
        w2_sum = 0
        while q.size() > 0:
            curr_node = q.pop()
            if curr_node.is_leave_node():
                num_leaves += 1
                # print("leaf depth: %d" % curr_node.depth)
                w2_sum += curr_node.y_hat**2
            if curr_node.left_child is not None:
                q.add(curr_node.left_child)
            if curr_node.right_child is not None:
                q.add(curr_node.right_child)
        print("number of leaves: %d" % num_leaves)
        # print("number of leaves2: %d" % self.num_leaves)
        return self.alpha * w2_sum + self.gamma * num_leaves


# import sklearn.datasets
# dataset = sklearn.datasets.load_boston()
# data, target = dataset['data'][:200], dataset['target'][:200]
# model = XGBoostBaseTree(num_bins=10, alpha=0.01, gamma=0.01, max_depth=10)
# model.fit(data, target)
# print(np.sum((model.predict(data) - target)**2)/len(target))
# print(model.penalty())