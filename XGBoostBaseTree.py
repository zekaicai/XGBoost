from DataBinModel import DataBinModel
import numpy as np
from PlotTree import createPlot
import json

class TreeStructureEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, XGBoostBaseTreeNode):
            dict = {}
            dict['split_col_index'] = o.split_col_index
            dict['split_point'] = o.split_point
            dict['y_hat'] = o.y_hat
            dict['score'] = o.score
            dict['depth'] = o.depth
            dict['num_samples'] = o.num_samples
            if o.left_child is not None:
                dict['left_child'] = json.loads(json.dumps(o.left_child, cls=TreeStructureEncoder))
            if o.right_child is not None:
                dict['right_child'] = json.loads(json.dumps(o.right_child, cls=TreeStructureEncoder))
            return dict
        return super().default(o)

class Queue:
    """
    先进先出的队列
    """
    def __init__(self):
        self.q = []

    def size(self):
        return len(self.q)

    def add(self, obj):
        self.q.insert(0, obj)

    def pop(self):
        assert self.size() > 0
        return self.q.pop()

def generate_tree(json):
    root = XGBoostBaseTreeNode(split_col_index=json['split_col_index'], split_point=json['split_point'],
                               y_hat=json['y_hat'], score=json['score'], depth=json['depth'], num_samples=json['num_samples'])
    if "left_child" in json:
        root.left_child = generate_tree(json['left_child'])
    if "right_child" in json:
        root.right_child = generate_tree(json['right_child'])
    return root

def generate_json(node):
    result = {}
    result['split_col_index'] = node.split_col_index
    result['split_point'] = node.split_point
    if node.left_child is not None:
        result['left'] = generate_json(node.left_child)
    if node.right_child is not None:
        result['right'] = generate_json(node.right_child)
    return result

class XGBoostBaseTreeNode:
    """
    XGBoost 使用的基本树的节点
    """
    def __init__(self, left_child=None, right_child=None, split_col_index=None, split_point=None,
                 y_hat=None, score=None, depth=None, num_samples=None):
        self.left_child = left_child
        self.right_child = right_child
        self.split_col_index = split_col_index
        self.split_point = split_point
        self.y_hat = y_hat
        self.score = score
        self.depth = depth
        self.num_samples = num_samples

    def is_leave_node(self):
        """
        一个节点是否是叶子结点
        :return: True if leave node
        """
        return self.left_child is None and self.right_child is None


class XGBoostBaseTree:
    def __init__(self, alpha, gamma, max_depth=None):
        """
        :param alpha: alpha 正则项，控制叶子结点的预测值不能过大
        :param gamma: gamma 正则项，控制树的叶子结点个数
        :param max_depth: 树的最大深度，默认不设限
        """
        self.max_depth = max_depth
        self.alpha = alpha
        self.gamma = gamma
        self.root = None

    def _score(self, g, h):
        """
        计算一个节点的 score
        :param g: 落在当前节点的样本的 loss 在前 k - 1 棵树的预测值处的一阶导
        :param h: 落在当前节点的样本的 loss 在前 k - 1 棵树的预测值处的二阶导
        :return: 节点的 score
        """
        G = np.sum(g)
        H = np.sum(h)
        return -0.5*G**2/(H+self.alpha) + self.gamma

    def _build_tree(self, curr_node, x, g, h, curr_depth):
        """
        以 curr_node 为 root，建立一棵树
        :param curr_node: 当前的节点
        :param x: 样本特征，在建立树的时候要选取最佳分裂特征和特征的最佳分裂点
        :param g: 样本在当前预测（上一次预测）值处损失函数的一阶导数
        :param h: 样本在当前预测（上一次预测）值处损失函数的二阶导数
        :param curr_depth: 当前树的深度，如果设了最大深度，应该比较决定是否停止增长
        """
        rows, cols = x.shape
        G = np.sum(g)
        H = np.sum(h)
        # 当前节点使 obj 最小的预测值
        curr_node.y_hat = -G / (H + self.alpha)
        curr_node.score = self._score(g, h)
        curr_node.depth = curr_depth
        curr_node.num_samples = rows

        # 如果当前树的深度已经达到了最大的限制，则不继续生长了
        if self.max_depth is not None and curr_depth >= self.max_depth:
            return

        best_col_index = None
        best_split_point = None
        best_gain = 0

        for col_index in range(cols):  # for each feature
            for split_point in sorted(set(x[:, col_index])): # for each split point
                # 尝试一个（分裂特征，分裂特征取值）组合，计算 gain
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

    def fit(self, x, g, h):
        """
        使用特征和一阶导数和二阶导数训练一棵树
        :param x: 特征
        :param g: 一阶导数
        :param h: 二阶导数
        """
        self.root = XGBoostBaseTreeNode()
        self._build_tree(self.root, x, g, h, 1)

    def _predict_sample(self, x):
        """
        预测一个样本
        :param x: 样本的特征
        :return: 样本的预测值
        """
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
        """
        预测一批样本
        :param x: 一批样本的特征矩阵
        :return: 一批样本的预测结果
        """
        rows = x.shape[0]
        result = []
        for row in range(rows):
            sample = x[row]
            y_hat = self._predict_sample(sample)
            result.append(y_hat)
        return np.asarray(result)

    def penalty(self):
        """
        计算这棵树的正则项，为叶子结点预测值的 l2 正则 + 叶子结点个数的 l1 正则
        :return: 树的正则值
        """
        # 采用 BFS 遍历所有叶子结点，记录叶子结点的个数以及叶子结点预测值的平方和
        q = Queue()
        q.add(self.root)
        num_leaves = 0
        w2_sum = 0
        while q.size() > 0:
            curr_node = q.pop()
            if curr_node.is_leave_node():
                num_leaves += 1
                w2_sum += curr_node.y_hat**2
            if curr_node.left_child is not None:
                q.add(curr_node.left_child)
            if curr_node.right_child is not None:
                q.add(curr_node.right_child)
        return self.alpha * w2_sum + self.gamma * num_leaves

    def json(self, labels):
        """
        将当前的树转化成 json 的格式，用来画图
        :param labels: 特征的名字，用来作为节点的 key
        :return: 树的 json 格式
        """
        assert not self.root.is_leave_node()
        return self._build_json(self.root, labels)

    def _build_json(self, curr_node, labels):
        """
        返回以当前节点为根节点的树的 json 格式
        :param curr_node: 当前节点
        :param labels: 特征的名字，用来作为节点的 key
        :return: 树的 json 格式
        """
        if curr_node.is_leave_node():
            return str(curr_node.y_hat)

        split_feature_name = labels[curr_node.split_col_index]
        split_point = curr_node.split_point
        key = str(split_feature_name) + ' > ' + str(split_point)
        return {key: {'no': self._build_json(curr_node.left_child, labels),
                      'yes': self._build_json(curr_node.right_child, labels)}}

    def plot(self, labels):
        """
        画出当前的树
        :param labels: 特征的名字，用来作为节点的 key
        """
        createPlot(self.json(labels))

root = XGBoostBaseTreeNode(split_col_index=2, split_point=5, y_hat=18, score=1, depth=1, num_samples=80)
root.left_child = XGBoostBaseTreeNode(split_col_index=3, split_point=8, y_hat=12, score=1, depth=2, num_samples=30)
root.right_child = XGBoostBaseTreeNode(split_col_index=4, split_point=6, y_hat=13, score=2, depth=2, num_samples=50)
root.right_child.left_child = XGBoostBaseTreeNode(split_col_index=3, split_point=2, y_hat=10, score=3, depth=3, num_samples=20)
root.right_child.right_child = XGBoostBaseTreeNode(split_col_index=5, split_point=1, y_hat=15, score=4, depth=3, num_samples=30)
json2 = json.loads(json.dumps(root, cls=TreeStructureEncoder))
root2 = generate_tree(json2)
json3 = json.dumps(root2, cls=TreeStructureEncoder)
