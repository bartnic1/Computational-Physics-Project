#This code uses power iteration to calculate a maximum eigenvalue/eigenvector pair

from numpy.linalg import eig
from numpy import array,zeros,dot

#Define the initial array and the arbitrary starting vector

A = array([[2,0,1],[0,2,0],[1,0,2]])
x_o = array([0.0,0.0,1.0])

#Initialize an array to hold the iterated values,
#and define the maximum number of iterations

k_max = 9
x_array = zeros([3,k_max+1],float)
x_array[:,0] = x_o

#Using the np.dot function, apply the power iteration process

print('The convergent vector values:\n')

for k in range(k_max):
    x_array[:,k+1] = dot(A,x_array[:,k])

    #Print the results at each step
    print(x_array[:,k+1])

#Starting at k = 1, print the ratios at each step
#We do this as the vector components have a 'zero' element for k = 0 that you can't
#divide over.

#We can use any component for the successive ratio values;
#based on the printed x_array, choose the last one

print('\nThe vector component ratios (1st and 3rd index; second is zero):\n')

for k in range(k_max):
    if k >= 1:
        print(x_array[:,k][2]/x_array[:,k][0])

print('\nThe successive ratio values:\n')

for k in range(k_max):
    if k >= 1:
        print(x_array[:,k+1][-1]/x_array[:,k][-1])

#Now compare to the actual eigenvalues and (corresponding) eigenvectors
#Search for the largest values and see whether there is agreement

eigenvalues, eigenvectors = eig(A)

print('\nThe actual eigenvalues:\n\n', eigenvalues)
print('\nThe corresponding actual normalized eigenvectors (columns):\n\n', eigenvectors)
