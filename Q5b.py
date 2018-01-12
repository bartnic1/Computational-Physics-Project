#This code integrates the QR decomposition program used in Q5a to perform
#Simultaneous Orthogonal Iteration to find the entire range of eigenvalues
#and eigenvectors for some square matrix A.

#Unresolved issue: Some eigenvectors are multiplied by -1.

from numpy import array,zeros,dot,sqrt
from numpy.linalg import eig
import numpy as np
import time

#Start the clock!

t1 = time.clock()

#Define the initial matrix, and then import the necessary functions

A = array([[1,4,8,4],[4,2,3,7],[8,3,6,9],[4,7,9,2]])

def norm(x):
    val = 0
    for i in range(len(x)):
        val += x[i]**2
    return sqrt(val)

def QR_decompose(A):
    
    u_array = zeros([len(A),len(A)],float)
    q_array = zeros([len(A),len(A)],float)
    r_array = zeros([len(A),len(A)],float)

    for i in range(len(A)):
        u_array[:,i] = A[:,i]
        if i >= 1:
            for j in range(i):
                subtractions = dot(q_array[:,j],A[:,i])
                u_array[:,i] -= subtractions*q_array[:,j]
                r_array[j,i] = subtractions
        q_array[:,i] = u_array[:,i]/float(norm(u_array[:,i]))

    for i in range(len(A)):
        for j in range(len(A)):
            if i > j:
                r_array[i,j] = 0
            elif i == j:
                r_array[i,j] = norm(u_array[:,i])

    return q_array,r_array

#Initialize arrays, and maximum number of iterations

X_0 = array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
X_0_normed = zeros([len(X_0),len(X_0)],float)
eigvals = zeros(len(X_0),float)

k_max = 8

#Main power iteration loop
#Only near the end do you need the diagonal matrix to find the eigenvalues

for k in range(k_max):
    X_1 = dot(A,QR_decompose(X_0)[0])
    if k == k_max - 1:
        Q_final = QR_decompose(X_0)[0]
        A_diag = dot(Q_final.transpose(),dot(A,Q_final))
    X_0 = X_1

#Once the eigenvectors have been found, normalize them for a comparison.
#Also find the eigenvalues using the upper triangular matrix R

for i in range(len(X_0)):
    X_0_normed[:,i] = X_0[:,i]/norm(X_0[:,i])
    eigvals[i] = A_diag[i,i]
    
t2 = time.clock()

#Now compare to the actual eigenvectors

print('Calculated eigenvectors:\n', X_0_normed)

t3 = time.clock()
eigenvalues, eigenvectors = eig(A)
t4 = time.clock()

print('\nActual Eigenvectors using Built-in Function:\n', eigenvectors)

print('\nCalculated Eigenvalues:\n', eigvals)
print('\nActual Eigenvalues using Built-in Function:\n', eigenvalues)

#Compare the time taken for either process

print('\nSimultaneous Orthogonal Iteration Time:', t2 - t1)
print('numpy.linalg Eigenvector Solver Time:', t4 - t3)

#Finding the average fractional error

fractional_error_array = (np.abs(eigvals - eigenvalues)/eigenvalues)*100.0

print('\nThe average fractional error is:', np.average(fractional_error_array))









