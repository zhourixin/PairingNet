import numpy as np

def dcg_at_k(scores, k):
    """计算 DCG@k 的值"""
    return np.sum(scores[:k] / np.log2(np.arange(2, k + 2)))

def ndcg_at_k(scores, k):
    """计算 NDCG@k 的值"""
    best = dcg_at_k(sorted(scores, reverse=True), k)
    if best == 0:
        return 0
    return dcg_at_k(scores, k) / best

# 假设我们有一个形状为 (20, 10) 的搜索结果矩阵
results = np.random.randint(0, 5, size=(20, 10))

# 计算每个检索项的 NDCG@10 值
ndcg_scores = [ndcg_at_k(row, 10) for row in results]

# 计算所有检索项的平均 NDCG@10 值
mean_ndcg = np.mean(ndcg_scores)

print(f"Mean NDCG@10: {mean_ndcg:.3f}")
