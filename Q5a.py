#This code shows how to perform a QR decomposition of a square matrix A
#QR Decomposition decomposes a matrix A into an orthogonal matrix Q and an upper triangular matrix R.

from numpy import array,zeros,dot,sqrt

#Create an arbitrary (but square) test array

A = array([[1,4,8,4],[4,2,3,7],[8,3,6,9],[4,7,9,2]])

#Initialize storage matrices

u_array = zeros([len(A),len(A)],float)
q_array = zeros([len(A),len(A)],float)
r_array = zeros([len(A),len(A)],float)

#The QR Algorith requires a norm; I couldn't find it in numpy,
#but its easily defined:

def norm(x):
    val = 0
    for i in range(len(x)):
        val += x[i]**2
    return sqrt(val)

#The main part of the code. Generates Q and R according to the relations
#found in Newman's text (see section 6.2, page 247)

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

#Test the results

print('Do the matrices match?\n')
print(A)
print(dot(q_array,r_array))

print('\nAre the columns of the Q array orthogonal?\n')

a = q_array
b = q_array.transpose()
print(dot(a,b))

#Thus we see that the results are equivalent, and that
#the columns are orthogonal
