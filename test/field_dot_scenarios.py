from hcipy import *
import numpy as np

'''
	A small script to test all possible scenarios for the field dot.
'''

grid = make_pupil_grid(2)

n = np.array([1,0,0])
print("n")
print(n)

n = np.tile(n,(grid.size,1))
n = Field(n.T, grid)
	
alpha = np.array([2,0,0])
print("alpha")
print(alpha)

A = np.diag(np.arange(3)+1)
A[0,2] = 1

print("A")
print(A)

B=field_dot(A,A)
print("A dot A")
print(np.all(np.isclose(B,A.dot(A))))

print("A dot n")
C=field_dot(A,n)
print(np.all(np.isclose(C[:,0],A.dot(n[:,0]))))

print("n dot A")
D=field_dot(n,A)
print(np.all(np.isclose(D[:,0],n[:,0].dot(A))))

print("n dot n")
E=field_dot(n,n)
print(np.isclose(E[0], n[:,0].dot(n[:,0])))

print("alpha dot n")
F=field_dot(alpha,n)
print(np.isclose(F[0],alpha.dot(n[:,0])))

print("n dot alpha")
G=field_dot(n,alpha)
print(np.isclose(G[0],n[:,0].dot(alpha)))