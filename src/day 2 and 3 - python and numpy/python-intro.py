matrix = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
]

print(list(zip(*matrix)))

basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
print(basket) 


import numpy as np
import matplotlib.pyplot as plt

# 1. Define matrices and compute outputs
m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([[2, 3], [5, 6]])

dot_product = np.dot(m1.flatten(), m2.flatten())
norm_m1 = np.linalg.norm(m1)
norm_m2 = np.linalg.norm(m2)
matmul_res = np.matmul(m1, m2)
cos_sim = dot_product / (norm_m1 * norm_m2)

# 2. Setup Figure
fig, ax = plt.subplots(figsize=(10, 10))

# --- Plot 1 & 2: Matrices Columns & Matmul Outputs ---
# m1 Column Vectors (Red)
ax.quiver(0, 0, m1[0, 0], m1[1, 0], angles='xy', scale_units='xy', scale=1, color='red', label='m1 Col 1 (1,3)')
ax.quiver(0, 0, m1[0, 1], m1[1, 1], angles='xy', scale_units='xy', scale=1, color='darkred', label='m1 Col 2 (2,4)')

# m2 Column Vectors (Blue)
ax.quiver(0, 0, m2[0, 0], m2[1, 0], angles='xy', scale_units='xy', scale=1, color='blue', label='m2 Col 1 (2,5)')
ax.quiver(0, 0, m2[0, 1], m2[1, 1], angles='xy', scale_units='xy', scale=1, color='darkblue', label='m2 Col 2 (3,6)')

# Matmul Column Vectors (Purple - scaled down for visualization layout if needed, or exact)
ax.quiver(0, 0, matmul_res[0, 0], matmul_res[1, 0], angles='xy', scale_units='xy', scale=1, color='purple', linestyle='--', label='Matmul Col 1 (12,26)')
ax.quiver(0, 0, matmul_res[0, 1], matmul_res[1, 1], angles='xy', scale_units='xy', scale=1, color='magenta', linestyle='--', label='Matmul Col 2 (15,33)')

# --- Plot 3: Vector Norms as Concentric Circles ---
circle_m1 = plt.Circle((0, 0), norm_m1, color='red', fill=False, linestyle=':', linewidth=1.5, label=f'Norm m1 ({norm_m1:.2f})')
circle_m2 = plt.Circle((0, 0), norm_m2, color='blue', fill=False, linestyle=':', linewidth=1.5, label=f'Norm m2 ({norm_m2:.2f})')
ax.add_patch(circle_m1)
ax.add_patch(circle_m2)

# --- Metadata Text Boxes (Dot Product & Cosine Similarity) ---
stats_text = (
    f"Flattened Dot Product: {dot_product}\n"
    f"Cosine Similarity: {cos_sim:.4f} (Highly Aligned)"
)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)

# --- Graph Limits & Formatting ---
ax.set_xlim(-1, 20)
ax.set_ylim(-1, 38)
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)
ax.grid(color='gray', linestyle='--', linewidth=0.5)
ax.set_title("Unified Linear Algebra Visual Representation", fontsize=14, fontweight='bold')
ax.set_xlabel("X Component")
ax.set_ylabel("Y Component")
ax.legend(loc='lower right')

plt.gca().set_aspect('equal', adjustable='box')
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# 1. Create data grids matching the array shapes
# 2D Grid structure: 3 rows, 5 columns (flat layer)
grid_2d = np.ones((1, 3, 5), dtype=bool)

# 3D Cube structure: 2 layers, 3 rows, 5 columns
grid_3d = np.ones((2, 3, 5), dtype=bool)

# 2. Setup a 3D plotting canvas with two subplots
fig = plt.figure(figsize=(14, 6))

# --- Left Subplot: 2D Representation ---
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.voxels(grid_2d, facecolors='skyblue', edgecolor='navy', alpha=0.8)
ax1.set_title("2D Array Structure\nShape: (3, 5) | ndim = 2", fontsize=12, fontweight='bold')
ax1.set_xlabel("Depth (1 Layer)")
ax1.set_ylabel("Rows (3)")
ax1.set_zlabel("Columns (5)")
# Force a perspective that looks flat
ax1.view_init(elev=20, azim=-10)

# --- Right Subplot: 3D Representation ---
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.voxels(grid_3d, facecolors='salmon', edgecolor='darkred', alpha=0.8)
ax2.set_title("3D Array Structure\nShape: (2, 3, 5) | ndim = 3", fontsize=12, fontweight='bold')
ax2.set_xlabel("Depth/Layers (2)")
ax2.set_ylabel("Rows (3)")
ax2.set_zlabel("Columns (5)")
ax2.view_init(elev=20, azim=-45)

plt.tight_layout()
# plt.show()

a = np.ones((2 ,2), dtype=bool)
print(a)
print(a.ndim)