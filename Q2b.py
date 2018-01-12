#This code shows how to use normalized power iteration to obtain
#the maximum eigenvalues/eigenvectors of a matrix A

from numpy.linalg import eigvals, eig
from numpy import array,zeros,dot
import numpy as np

#Define the initial array and the arbitrary starting vector

A = array([[2,0,1],[0,2,0],[1,0,2]])
x_o = array([0.0,0.0,1.0])

#Initialize an array to hold the iterated values,
#and define the maximum number of iterations

k_max = 9
x_array = zeros([3,k_max+1],float)
y_array = zeros([3,k_max+1],float)
x_array[:,0] = x_o

#Now apply the normalized power iteration process

print('The convergent vector values:\n')

for k in range(k_max):
    y_array[:,k+1] = dot(A,x_array[:,k])
    x_array[:,k+1] = y_array[:,k+1]/np.max(y_array[:,k+1])
    
    #Print the results at each step
    print(x_array[:,k+1])

#Print the component ratios as before

print('\nThe vector component ratios (1st and 3rd index; 2nd is zero):\n')

for k in range(k_max):
    if k >= 1:
        print(x_array[:,k][2]/x_array[:,k][0])

#Note that this time the successive ratio values (from above) are simply equal
#to the max norm.

print('\nThe successive ratio values:\n')

for k in range(k_max):
    print(np.max(y_array[:,k+1]))

#Now compare to the actual eigenvalues and (corresponding) eigenvectors
#Search for the largest values and see whether there is agreement

eigenvalues, eigenvectors = eig(A)

print('\nThe actual eigenvalues:\n\n', eigenvalues)
print('\nThe corresponding actual (normalized) eigenvectors (columns):\n\n', eigenvectors)

