from XGBoostBaseTree import XGBoostBaseTreeNode
from XGBoostBaseTree import XGBoostBaseTree
import itertools
import numpy as np

class Shap:

    def __init__(self, tree):
        self.tree = tree

    @staticmethod
    def EXPVALUE(x, S, tree):

        def G(node: XGBoostBaseTreeNode, w):
            # 如果当前节点是叶子结点，直接返回 w * node.y_hat
            if node.is_leave_node():
                return w * node.y_hat
            else:
                if node.split_col_index in S:
                    if x[node.split_col_index] <= node.split_point:
                        return G(node.left_child, w)
                    else:
                        return G(node.right_child, w)
                else:
                    left_num_leaves = node.left_child.num_samples
                    right_num_leaves = node.right_child.num_samples
                    left_weight = left_num_leaves / (left_num_leaves + right_num_leaves)
                    right_weight = right_num_leaves / (left_num_leaves + right_num_leaves)
                    left_score = G(node.left_child, w * left_weight)
                    right_score = G(node.right_child, w * right_weight)
                    return left_score + right_score

        return G(tree.root, 1)

    def explain(self, x):

        # 找到所有可能的特征子集
        feature_subsets = []
        for feature_num in range(0, len(x)+1):
            feature_subsets += itertools.combinations(list(range(len(x))), r=feature_num)

        result = [0 for i in range(len(x))]

        # 分别算出每个特征的 shap 值
        m_mode = len(x)
        for feature_idx in range(len(x)):
            # 找出所有包含该特征的子集
            for feature_subset in feature_subsets:
                if feature_idx in feature_subset:
                    # 去掉这个特征本身
                    feature_subset_exclude_itself = feature_subset[:feature_subset.index(feature_idx)] +feature_subset[feature_subset.index(feature_idx)+1:]
                    s_mode = len(feature_subset_exclude_itself)
                    weight = np.math.factorial(s_mode)*np.math.factorial(m_mode-s_mode-1) / np.math.factorial(m_mode)
                    diff = self.EXPVALUE(x, feature_subset, self.tree) - self.EXPVALUE(x, feature_subset_exclude_itself, self.tree)
                    result[feature_idx] += diff * weight
        return result



# root = XGBoostBaseTreeNode(split_col_index=2, split_point=5, y_hat=18, score=1, depth=1, num_samples=80)
# root.left_child = XGBoostBaseTreeNode(split_col_index=3, split_point=8, y_hat=12, score=1, depth=2, num_samples=30)
# root.right_child = XGBoostBaseTreeNode(split_col_index=4, split_point=6, y_hat=13, score=2, depth=2, num_samples=50)
# root.right_child.left_child = XGBoostBaseTreeNode(split_col_index=3, split_point=2, y_hat=10, score=3, depth=3, num_samples=20)
# root.right_child.right_child = XGBoostBaseTreeNode(split_col_index=5, split_point=1, y_hat=15, score=4, depth=3, num_samples=30)
# tree = XGBoostBaseTree(alpha=1,gamma=1)
# tree.root = root
# # v = Shap.EXPVALUE([1,1,4,1,1,1], [], tree)
# # print(v)
# print(Shap(tree).explain([1,2,2,4,5,6]))