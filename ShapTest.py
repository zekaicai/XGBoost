from XGBoostBaseTree import XGBoostBaseTreeNode
from XGBoostBaseTree import XGBoostBaseTree, TreeStructureEncoder
from Shap import Shap
import json


root = XGBoostBaseTreeNode(split_col_index=0, split_point=0, y_hat=-1, score=-1, depth=1, num_samples=100)
root.left_child = XGBoostBaseTreeNode(split_col_index=1, split_point=0, y_hat=-1, score=-1, depth=2, num_samples=30)
root.right_child = XGBoostBaseTreeNode(split_col_index=2, split_point=0, y_hat=-1, score=-1, depth=2, num_samples=70)
root.left_child.left_child = XGBoostBaseTreeNode(split_col_index=-1, split_point=-1, y_hat=1, score=-1, depth=3, num_samples=10)
root.left_child.right_child = XGBoostBaseTreeNode(split_col_index=-1, split_point=-1, y_hat=2, score=-1, depth=2, num_samples=20)
root.right_child.left_child = XGBoostBaseTreeNode(split_col_index=-1, split_point=2, y_hat=3, score=-1, depth=3, num_samples=30)
root.right_child.right_child = XGBoostBaseTreeNode(split_col_index=-1, split_point=-1, y_hat=4, score=-1, depth=3, num_samples=40)

print(json.dumps(root, cls=TreeStructureEncoder))

tree = XGBoostBaseTree(alpha=1,gamma=1)
tree.root = root

# res = Shap.EXPVALUE([1,1,1], (0,1,2), tree)
shap = Shap(tree)
res = shap.explain([1,1,1])
print(res)
total = 0
for i in res:
    total += i
print(total)


def test_EXPVALUE(tree):
    pass