import numpy as np

m1 = np.array([[1, 2],
               [3, 4]]) 
m2 = np.array([[2, 3], 
               [5, 6]])

def dot_product(a: np.array, b: np.array):
    return np.dot(a, b)

print("m1.m2: \n", dot_product(m1, m2))

def vector_norm(n: np.array):
    return np.linalg.norm(n) # np.sqrt(np.sum(n ** 2))

print("Vector Norm |m1|: ", vector_norm(m1))
print("Vector Norm |m2|: ", vector_norm(m2))

def matrix_multiply(a: np.array, b: np.array):
    return np.matmul(a, b) # a @ b

print("m1 x m2: \n", matrix_multiply(m1, m2))

def cosine_similarity(a: np.array, b: np.array):
    return dot_product(a, b) / (vector_norm(a) + vector_norm(b))

print("Cosine Similarity cos_sim(m1, m2): \n", cosine_similarity(m1, m2))


### Day 3

def mean_by_row(a: np.array):
    return np.mean(a, axis=-1)

print("\n Mean by row (m1): \n", mean_by_row(m1))

def normalize_columns(a: np.array):
    return (a - np.mean(a, axis=0)) / np.std(a, axis=0)

print("\n Normaliza Columns (m1): \n", normalize_columns(m1))

def pairwise_l2_distances(a: np.array):
    
    diff = a[:, None, :] - a[None, :, :]
    # print("Debug: \n", a[:, None, :], a[None, :, :], diff, end="\n")
    return np.sqrt((diff ** 2).sum(2))

print("\n Pairwise l2 distance (m1): \n", pairwise_l2_distances(m1))

print("Sum axis=0: ", np.array([[2, 3], [5, 6]]).sum(axis=-2))