#This code limits eigenvalues to their Gershgorin discs

from numpy.linalg import eigvals
from numpy import array, zeros, linspace, pi, sin, cos
import matplotlib.pyplot as plt

#Input the square (n x n) matrix A, initialize some arrays to define the disc radii and centers

A = array([[4.0,-0.5,0.0],[0.6,5.0,-0.6],[0.0,0.5,3.0]])

radius_array = zeros(len(A),float)
disc_centers = zeros(len(A),float)

##For Part B: Calculating the eigenvalues
eigenvalues = eigvals(A)
print(eigenvalues)
##

#Compute the disc radii and centers according to Gershgorin's theorem

for i in range(len(A)):
    for j in range(len(A)):
        if j == i:
            disc_centers[i] = A[i,j]
        if j != i:
            radius_array[i] += abs(A[i,j])

#Define some functions (and a theta array) to translate these values onto a plot

def x_vals(r,theta,disc_center):
    return r*cos(theta) + disc_center.real

def y_vals(r,theta,disc_center):
    return r*sin(theta) + disc_center.imag

theta_array = linspace(0,2*pi,1000)

#Now plot the results! Note that this plot is being done on the complex plane.

for i in range(len(A)):
    plt.plot(x_vals(radius_array[i],theta_array,disc_centers[i]),y_vals(radius_array[i],theta_array,disc_centers[i]))


## For part B: Also plot the eigenvalues on the complex plane noting that the y-component should be imaginary  
plt.plot(eigenvalues.real,eigenvalues.imag,'o')

plt.title('Gershgorin Discs Localizing the Eigenvalues of the 3 x 3 Matrix A')
plt.xlabel('Real Values')
plt.ylabel('Imaginary Values')
plt.grid()
plt.show()
