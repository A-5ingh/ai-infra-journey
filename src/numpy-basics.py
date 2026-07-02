# install numpy with pip install numpy

import numpy as np
import matplotlib.pyplot as plt

a = np.array([[1, 2, 3],
              [4, 5, 6]])
print(a.shape, a.ndim)


a = np.array([1, 2, 7, 2 , 4, 5, 6])

plt.plot(a)
plt.show()

print(a, a.shape)

a2 = a[np.newaxis, :]
print(a2, a2.shape)

a3 = a[:, np.newaxis]
print(a3, a3.shape)

rng = np.random.default_rng()
r = rng.integers(5, size=(2, 4))
print(r)