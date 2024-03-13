# Esha Chakrabarty
# 920884665

import numpy as np

matrix1= np.array([[1, 0, 1, 1],
                   [0, 1, 0, 1],
                   [1, 0, 0, 1],
                   [0, -1, 1, 1]])

dim = matrix1.shape
a1 = matrix1[0]
r11 = np.linalg.norm(a1)
q1 = a1/r11


a2 = matrix1[1]

r12 = np.inner(q1, a2) #number
r22 = np.linalg.norm(a2-r12*q1) #number
q2 = np.subtract(a2,r12*q1)/r22


#classic
def cgs(matrix):
  dim = matrix.shape
  ortho = np.empty(dim)
  for j in range(dim[1]):
    aj = matrix[j]
    allj = aj
    for i in range(j):
      qi = ortho[i]
      rij = np.inner(qi,aj)
      allj = allj-rij*qi

    rjj=np.linalg.norm(allj)
    qj=allj/rjj
    ortho[j] = qj
  return(ortho)

print(cgs(matrix1))

