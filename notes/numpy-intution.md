# Numpy

Numpy is a library that simplifies numerical/mathematical operations on n-dimensional arrays.

## Broadcasting
Broadcasting lets numpy perform element-wise operations on arrays of different shapes by virtually "stretching" the smaller array to match the larger one, without copying data.

Rule: Two dimensions are compatible when they are equal, or one of them is 1. Comparison starts from the trailing (rightmost) dimension.

Gotcha: If a pair of dimensions differ and neither is 1, broadcasting fails (shape mismatch error).

## Definitions

### Dot product
Take two vectors of the same length, multiply corresponding elements, sum the results.

```
a = [1, 2, 3]
b = [4, 5, 6]

dot = (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32
```

Geometrically, the dot product measures how much two vectors point in the same direction, scaled by their lengths. If they point exactly the same way, the dot product is large and positive. If they're perpendicular, it's zero. If they point opposite ways, it's negative.

**Where you'll use it again:** it's the single most-repeated operation in this entire plan. Neural network layers are dot products (weights · inputs). Attention (Week 5) is dot products between "query" and "key" vectors to measure how relevant one token is to another. It's everywhere.

### Vector norm
The length (magnitude) of a vector. The standard one (L2 norm) is:
```
v = [3, 4]
norm = sqrt(3² + 4²) = sqrt(9+16) = sqrt(25) = 5
```

It's just the Pythagorean theorem generalized to more dimensions — "how far is this point from the origin."

**Where you'll use it again:** normalizing vectors (dividing by their norm so they all have length 1), measuring how big a gradient or weight update is, and gradient clipping later on when you train your GPT.

### Matrix multiply
Matrices are grids of numbers. Multiplying two matrices A @ B means: for each row of A and each column of B, take their dot product, and that becomes one entry in the result.
```
A = [[1,2],       B = [[5,6],
     [3,4]]            [7,8]]

A @ B = [[1×5+2×7, 1×6+2×8],   = [[19, 22],
         [3×5+4×7, 3×6+4×8]]     [43, 50]]
```
Each output entry is literally a dot product of a row and a column. So this is just dot products, batched.

**Where you'll use it again:** every neural network layer is a matrix multiply (a batch of inputs times a weight matrix). This is what your GPT is doing millions of times per forward pass.

### Cosine similarity

Measures how similar the direction of two vectors is, ignoring their length. It's the dot product, normalized by both vectors' norms:
```
cosine_similarity(a, b) = dot(a, b) / (norm(a) × norm(b))
```
Result ranges from -1 (opposite direction) to 1 (same direction), regardless of how long the vectors are. This is why it's built from the two pieces above — it's not a separate concept, it's dot product and norm composed together.

**Where you'll use it again:** semantic search and RAG (Week 10) — comparing embedding vectors to find "similar" text. Your gateway's caching system later also uses this to detect similar requests.

### what does axis=0 mean for np.sum? 
If we have a 2-d array [[2, 3], [5, 6]], then sum across axis 0 will be across it columns so it will result in [7, 9].


### Draw a 3×4 matrix and annotate what summing along each axis produces.
Let's take this example:
[
     [2, 4, 1, 1],
     [1, 9, 2, 4],
     [3, 4, 2, 6]
]

sum when using axis=0, [6, 17, 5, 11]
sum when using axis=1, [8, 16, 15]

axis=0 removes the row dimension → left with one value per column
axis=1 removes the column dimension → left with one value per row