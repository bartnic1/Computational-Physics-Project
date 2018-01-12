#This code demonstrates how to find all of the eigenvectors and eigenvalues
#of a given matrix A using Jacobi rotations.

from numpy.linalg import eig
from numpy import array,sqrt,dot,zeros
import numpy as np
import time

#Define the array to find eigenvalues/eigenvectors of

A = array([[1,4,8,4],[4,2,3,7],[8,3,6,9],[4,7,9,2]])

#Find the time taken to calculate the eigenvectors/eigenvalues using the built-in function

time1 = time.clock()
theory_values, theory_vectors = eig(A)
time2 = time.clock()

time3 = time.clock()

#Initialize the eigenvector and eigenvalue arrays

eigenvector_matrix = zeros([len(A),len(A)],float)
for k in range(len(A)):
    eigenvector_matrix[k,k] += 1

eigenvalue_array = zeros(len(A),float)
                    
k_max = 4

#The main program

for k in range(k_max):
    for i in range(len(A)):
        for j in range(len(A)):
            if j > i:
                a = A[i,j-(j-i)]
                b = A[i,j]
                c = A[i+(j-i),j-(j-i)]
                d = A[i+(j-i),j]

                #Solve for t (should be positive)
                t1 = ((d-a)/float(b) + sqrt(((d-a)/float(b))**2+4))/-2.0
                t2 = ((d-a)/float(b) - sqrt(((d-a)/float(b))**2+4))/-2.0
                t = max(t1,t2)

                #Solve for sine and cosine (s,c)
                c = 1.0/sqrt(1+t**2)
                s = c*t

                #Define the Jacobi rotation matrix

                J = zeros([len(A),len(A)],float)
                for k in range(len(A)):
                    J[k,k] += 1

                J[i,j] = s
                J[i,j-(j-i)] = c
                J[i+(j-i),j-(j-i)] = -s
                J[i+(j-i),j] = c

                #Perform the rotation

                A = dot(J.transpose(),dot(A,J))

                #Keep track of the product of all the rotations (this is the eigenvector matrix)

                eigenvector_matrix = dot(eigenvector_matrix,J)

#Calculate the eigenvalues by reading off of the diagonal of the resultant matrix A

for i in range(len(A)):
    eigenvalue_array[i] = A[i,i]

time4 = time.clock()


#And now its time to print the results

print('The Jacobi calculated eigenvectors are:\n', eigenvector_matrix)
print('\nThe eigenvectors calculated by the built-in \' eig \' function are:\n', theory_vectors)

print('\nThe Jacobi calculated eigenvalues are:\n', eigenvalue_array)
print('\nThe eigenvalues calculated by the built-in \' eig \' function are:\n', theory_values)

#Note that it turns out that these times are also unreliable as in Q5 (they vary quite a bit)
print('\nThe time taken to calculate these values using the Jacobi method was:\n', time4 - time3)
print('\nThe time taken to calculate these values using the \' eig \' function was:\n', time2 - time1)

#Finding the average fractional error

#The eigenvalues and eigenvectors don't come out sorted, so have to do it manually
#Then can compare them with the theoretical eigenvalues directly

ordered_eigenvalue_array = sorted(np.abs(eigenvalue_array),reverse=True)
fractional_error_array = (np.abs(np.abs(ordered_eigenvalue_array) - np.abs(theory_values))/np.abs(theory_values))*100.0

print('\nThe average fractional error is:', np.average(fractional_error_array), '%')

