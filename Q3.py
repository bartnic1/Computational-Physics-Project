#This code shows how to use inverse iteration to find the smallest
#eigenvalue/eigenvector pair for a matrix A

from numpy.linalg import inv, eig
from numpy import array,zeros,dot
import numpy as np

#Define the matrix and the initial guess

A = array([[2,0,1],[0,2,0],[1,0,2]])
x_o = array([0.0,0.0,1.0])

#Now invert the matrix and initialize arrays

A_inv = inv(A)

k_max = 9
x_array = zeros([3,k_max+1],float)
y_array = zeros([3,k_max+1],float)
x_array[:,0] = x_o

#Perform power iteration using the inverse; the smallest eigenvalue
#now becomes the largest

print('The convergent vector values:\n')

for k in range(k_max):
    y_array[:,k+1] = dot(A_inv,x_array[:,k])
    x_array[:,k+1] = y_array[:,k+1]/np.max(y_array[:,k+1])

    print(x_array[:,k+1])

#Have to remember that what you end up with is the inverse of the
#smallest eigenvalue

print('\nThe successive ratio values (1/eigenvalue):\n')

for k in range(k_max):
    print(np.max(y_array[:,k+1]))

#Now compare with the actual results!

eigenvalues, eigenvectors = eig(A)

print('\nThe actual eigenvalues:\n\n', eigenvalues)
print('\nThe corresponding actual (normalized) eigenvectors:\n\n', eigenvectors)

print('\n The fractional error is:', (abs((1.0/np.max(y_array[:,-1]))-1.0)/1.0)*100.0, '%')

