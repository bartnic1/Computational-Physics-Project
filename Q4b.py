#This code is used to compare with Q4a.py, showing how long it would take
#to compute the maximal eigenvalue/eigenvector pair without Rayleigh quotients

from numpy.linalg import eigvals, eig
from numpy import array,zeros,dot
import numpy as np

#Define the initial array and the arbitrary starting vector

A = array([[1,2,3],[1,2,1],[3,2,1]])
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

#Note that this time the successive ratio values (from above) are simply equal
#to the max norm.

print('\nThe successive ratio values:\n')

for k in range(k_max):
    print(np.max(y_array[:,k+1]))

#Now compare to the actual eigenvalues and (corresponding) eigenvectors
#Search for the largest values and see whether there is agreement

eigenvalues, eigenvectors = eig(A)

print('\nThe actual eigenvalues:\n\n', eigenvalues)
print('\nThe corresponding actual (normalized) eigenvectors:\n\n', eigenvectors)

print('\n The fractional error is:', (abs(np.max(y_array[:,-1]) - 5.23606798)/5.23606798)*100.0, '%')
