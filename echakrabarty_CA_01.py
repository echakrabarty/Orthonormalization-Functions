# Esha Chakrabarty
# 920884665

from scipy import io
from echakrabarty_MODIFIED_GRAM_SCHMIDT import mgs
import matplotlib.pyplot as plt
import numpy as np

data_dict = io.loadmat('/Users/eshachakrabarty/Downloads/CA_01.mat')

# From Piazza used to quickly transfer matrices into Latex
def mat_texer(A, r): #matrix to print A, digits to round to r
    m, n = np.shape(A)
    for x in range(m):
        for y in range(n - 1):
            print(("%." + str(r) + "f")%(A[x,y]), end=" & ")
        print(("%." + str(r) + "f")%(A[x,n - 1]), end=" \\\\")
        print("")


# PART A: Loading and Understanding the Data
# (a)
x = data_dict['x'].reshape(-1)
U = data_dict['U']

spacing = np.linspace(1, 8, 8)
# Figure 1
plt.stem(spacing, x, linefmt = "m")
plt.title("Vector x")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()


# PART B: Finding orthonormal basis vectors
# (b)
# Figure 2

def dctFunctions(x):
    colors = ["b", "g", "r", "c", "m", "y", "k", "b"]
    spacing = np.linspace(1, 8, 100)
    for k in range(len(x)):
        y = np.cos((k*np.pi*(spacing-1))/7)
        plt.plot(spacing, y, color=colors[k]) 
        
    plt.title("Functions")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    return(plt.show())

dctFunctions(x)


# (c)

cols = np.array([1,2,3,4,5,6,7,8])
def dctArray(x):
    result = np.empty([8,8])
    for k in range(len(x)):
        y = np.cos((k*np.pi*(x-1))/7)
        result[:,k] = y
    return(result)

A = dctArray(cols)

# (d)
A_T = np.transpose(A)
A_TA = A_T @ A

#matrix is not orthonormal because ATA would result in the identity matrix
#creating matrix using my mgs function from Homework 1
U = mgs(A)

# (e)
# Figure 3

# Create stem plots for each vector in the matrix
num_vectors = U.shape[1]
num_rows = 4
num_columns = num_vectors // num_rows

fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, 6))

for k in range(num_vectors):
    ax = axes[k // num_columns, k % num_columns]  # Indexing the axes
    ax.stem(U[:, k])
    ax.plot(U[:, k])
    ax.set_title(f"Vector {k + 1}")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.grid(True)

plt.tight_layout()  # Adjust spacing
plt.show()


# PART C: An Audio 'Signal'
# (f)
a = np.transpose(U) @ x

# (g)
aPos = abs(a) #check for largest absolute values
a2 = np.zeros([1,8])
#manually inputting those points into vector of zeros
a2[:,4] = a[4]
a2[:,6] = a[6]

x2 = U @ np.transpose(a2)

# (h)
# Figure 4
plt.stem(spacing, x, linefmt="m")
plt.stem(spacing, x2, linefmt = "g", markerfmt="*")
plt.plot(spacing, x2, color = "g")
plt.title("Vector x2 and x")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()


# (i)
#creating a4 from a2 and adding necessary values
a4 = a2
a4[:,1] = a[1]
a4[:,7] = a[7]

# (j)
x4 = U @ np.transpose(a4)

# Figure 5
plt.stem(spacing, x, linefmt="m")
plt.stem(spacing, x4, linefmt = "k--", markerfmt="x")
plt.plot(spacing, x4, color = "k")
plt.title("Vector x4 and x")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()


# (k)
x8 = U @ a

relativeErrorx2 = np.sqrt(np.sum((x - np.transpose(x2)) ** 2) / np.sum(x ** 2))
relativeErrorx4 = np.sqrt(np.sum((x - np.transpose(x4)) ** 2) / np.sum(x ** 2))
relativeErrorx8 = np.sqrt(np.sum((x - x8) ** 2) / np.sum(x ** 2))

print(relativeErrorx2)
print(relativeErrorx4)
print(relativeErrorx8)
    






