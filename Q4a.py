#This code demonstrates how Rayleigh quotients can be used to speed
#up the convergence process of inverse iteration

from numpy.linalg import inv, eig, solve
from numpy import array,zeros,dot
import numpy as np

#Define all the arrays
A = array([[1,2,3],[1,2,1],[3,2,1]])
I = array([[1,0,0],[0,1,0],[0,0,1]])
x_o = array([1,1,1])

#Initialize the containers, and the max number of iterations
k_max = 3
sigma_init = 10
x_array = zeros([3,k_max+1],float)
y_array = zeros([3,k_max+1],float)
sigma_array = zeros(k_max+1)

#Initial guess for the vector and the eigenvalue
x_array[:,0] = x_o
sigma_array[0] = sigma_init

#Now iterate - at each stage refine the guesses for the eigenvectors and eigenvalues

print('The convergent vector values:\n')

for k in range(k_max):
    y_array[:,k+1] = dot(inv(A - sigma_array[k]*I),x_array[:,k])
    x_array[:,k+1] = y_array[:,k+1]/np.max(y_array[:,k+1])    
    sigma_array[k+1] = dot(x_array[:,k],dot(A,x_array[:,k]))/dot(x_array[:,k],x_array[:,k])
    
    print(x_array[:,k])

print('\nThe convergent eigenvalues:\n')

for k in range(k_max+1):
    print(sigma_array[k])

eigenvalues, eigenvectors = eig(A)

print('\nThe actual eigenvalues:\n\n', eigenvalues)
print('\nThe corresponding actual (normalized) eigenvectors:\n\n', eigenvectors)

print('\n The fractional error is:', (abs(sigma_array[-1] - 5.23606798)/5.23606798)*100.0, '%')
