#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
======================================
Clustering by rotation of eigenvectors
======================================

cluster by rotating eigenvectors to align with the canonical coordinate system

usage:
nc = cluster_rotate(evecs, evals, group_num, method, verbose)
 
Input:
 - evecs = array of eigenvectors
 - evals = eigenvalues associated with eigenvectors
 - group_num - an array of group numbers to test it is assumed to be a continuous set
 - method - 1   gradient descent 
	    2   approximate gradient descent
 - verbose
        
Output:
 - the highest number with best alignment quality

"""

# Code source by Marie Tahon (2018) adapted for Python from Lihi Zelnik-Manor (2005) (Matlab)
# License: ISC

import numpy as np
import os, sys


def evqual(X, dim, ndata, verbose):
	#compute alignement quality
	
	max_values = np.amax(abs(X), axis=1)
	if verbose: print('Found max of each row')
	
	#compute cost
	if verbose: print('dim:', dim, 'ndata:', ndata)
	J=0
	for i in range(ndata):
		R = X[i,:] / max_values[i]
		J += np.sum(R**2)
	
	J = 1 - (J/ ndata - 1) / dim
	if np.isnan(J):
		print('J is nan number')
		sys.exit()
	if verbose: print('computed quality:', J)
	return J



def evqualitygrad(X, theta, ik, jk, angle_num, angle_index, dim, ndata, verbose):
	#compute gradient quality

	V = gradU(theta, angle_index, ik, jk, dim)
	U1 = build_Uab(theta, 0, angle_index-1, ik, jk, dim)
	U2 = build_Uab(theta, angle_index+1, angle_num-1, ik, jk, dim)
	A = buildA(X, U1, V, U2)
	
	#get rid of no longer needed arrays
	del V, U1, U2
	
	#rotate vecs according to current angles
	Y = rotate_givens(X, theta, ik, jk, angle_num, dim)
	
	#find max of each row
	max_values = np.amax(abs(Y), axis=1)
	max_index = np.argmax(abs(Y), axis=1)
	
	if verbose: print('Found max of each row')
	
	#compute cost # Yij = Zij et Aij = Aij with mi = max_index[i]
	dJ = 0
	for j in range(dim): # loop over all columns
		for i in range(ndata): #loop over all rows
			tmp1 = A[i,j]*Y[i,j] / (max_values[i] * max_values[i])
			tmp2 = A[i, max_index[i]]*Y[i,j]*Y[i,j]  / (max_values[i] * max_values[i] * max_values[i])
			dJ += 2*(tmp1 - tmp2)
			
	dJ = - dJ / (ndata *dim) # on normalize
	if verbose: print('Computed gradient:', dJ)
	
	del Y, A, max_values, max_index
	
	return dJ



def cluster_assign(X, ik, jk, dim, ndata):
	#take the square of all entries and find max of each row

	max_values = np.zeros(shape=[ndata], dtype=float)
	max_index = np.zeros(shape=[ndata], dtype=int)
	cluster_count = np.zeros(shape=[dim], dtype=int)
	
	for j in range(dim): #loop over columns
		for i in range(ndata): #loop over rows
			if j == 0:
				max_index[i] = -1
			if max_values[i] <= X[i,j]*X[i,j]:
				if max_index[i] >= 0:
					cluster_count[max_index[i]] -= 1
					cluster_count[j] += 1
					max_values[i] = X[i,j] * X[i,j]
					max_index[i] = j
					
	#allocate memory for cluster assignements
	cluster_cell_array = np.empty(shape=[dim], dtype=object)
	for j in range(dim): #loop over all columns
		cluster = np.empty(shape=[cluster_count[j]], dtype=float)
		cind = 0
		for i in range(ndata): # loop over all rows
			if max_index[i] == j:
				cluster[cind] = i+1
				cind += 1
		cluster_cell_array[j] = cluster

	del max_values, max_index, cluster_count
	
	return cluster_cell_array


def gradU(theta, k, ik, jk, dim):
	#compute V as the Gradient of a single Givens rotation
	
	V = np.zeros(shape=[dim, dim], dtype=float)
	V[ik[k], ik[k]] = -np.sin(theta[k])
	V[ik[k], jk[k]] = -np.cos(theta[k])
	V[jk[k], ik[k]] = np.cos(theta[k])
	V[jk[k], jk[k]] = -np.sin(theta[k])
	
	return V

# Givens rotation for angles a to b
def build_Uab(theta, a, b, ik, jk, dim):
	if not (type(a) is int) & (type(b) is int):
		print('Angles are not integers')
		sys.exit()
		
	#set Uab to be an identity matrix
	Uab = np.identity(dim, dtype=float) 
	if b < a:
		return Uab
	else:
		for k in range(a,b+1):
			#tt = theta[k]
			c = np.cos(theta[k])
			s = np.sin(theta[k])
			for i in range(dim):
				u_jk = Uab[ik[k],i] * s + Uab[jk[k],i] * c
				Uab[jk[k],i] = u_jk
				Uab[ik[k],i] = Uab[ik[k],i] * c - Uab[jk[k],i] * s
				
		return Uab
	
	
	
	
	
def  buildA(X, U1, Vk, U2): # A(k) = X U(1,k-1) V(k) U(k+1,K) indexes correspond to angles.
	A1 = np.dot(Vk, U2)
	A2 = np.dot(U1, A1)
	A = np.dot(X, A2)
	del A1, A2
	return A


def rotate_givens(X, theta, ik, jk, angle_num, dim):
	#Rotate vectors in X with Givens rotation according to angles in theta
	G = build_Uab(theta, 0, angle_num -1, ik, jk, dim)
	#print(G)
	Y = np.dot(X,G)
	del G
	return Y



def evrot(evecs, method, verbose):
	
	#get the number and length of eigenvectors dimensions
	ndata, dim = evecs.shape 
	if verbose: print('Got {0} vectors of length {1}'.format(dim, ndata))

	#get the number of angles
	angle_num = int(dim* (dim -1) /2) #K
	angle_step = np.pi/angle_num
	if verbose: print('Angle number is:', angle_num)
	#print(angle_step)
	#build index mapping
	ik = np.empty(shape=[angle_num], dtype=int)
	jk = np.empty(shape=[angle_num], dtype=int)
	#print('shapes:', theta.shape, ik.shape) 
	k=0
	for i in range(dim):
		for j in range(i+1,dim):
			ik[k] = i
			jk[k] = j
			k += 1
	theta = np.random.uniform(-np.pi/2, np.pi/2-0.001, size=angle_num)
	if verbose: print('Built index mapping for {0} angles'.format(k))

	#definitions
	max_it = 20

	#evaluate intial quality
	Q = evqual(evecs,ik,jk,dim,ndata, verbose)
	
	Q_old1 = Q
	Q_old2 = Q

	it = 0

	while it < max_it: #iterate to refine quality
		it += 1
		for d in range(angle_num):
			td = theta[d]
			if verbose: print('----------------------d=', d, it)
			if method == 1: # descend through true derivative
				alpha = 1
				
				dQ = evqualitygrad(evecs, theta, ik, jk, angle_num, d, dim, ndata, verbose)
				
				theta_new = np.array([td - alpha * dQ if k == d else t for k,t in enumerate(theta) ])
				if theta_new[d]-theta[d] == 0:
					print('(it, d)', it, d, theta_new[d]-theta[d])
					sys.exit()
				evecsRot = rotate_givens(evecs, theta_new, ik, jk, angle_num, dim)
				Q_new = evqual(evecsRot, ik, jk, dim, ndata, verbose)
				#print(Q_new)
				if Q_new > Q: 
				#we need to maximize quality (minimize cost function). Then if running k  improves quality we keep the changes else, we do not change anything.
					theta = np.array([td - alpha * dQ if k == d else t for k,t in enumerate(theta) ])
					Q = Q_new
				else:
					theta_new = np.array([td if k ==d else t for k,t in enumerate(theta_new)])
				del evecsRot
				
			elif method == 2:
				alpha = 0.1
				#move up
				theta_new = np.array([(td + alpha) if k == d else t for k,t in enumerate(theta) ])
				evecsRot = rotate_givens(evecs, theta_new, ik, jk, angle_num, dim)
				Q_up = evqual(evecsRot, ik, jk, dim, ndata, verbose)
				del evecsRot
				
				#move down
				theta_new = np.array([(td - alpha) if k == d else t for k,t in enumerate(theta) ])
				evecsRot = rotate_givens(evecs, theta_new, ik, jk, angle_num, dim)
				Q_down = evqual(evecsRot, ik, jk, dim, ndata, verbose)
				
				
				#update only if at least one of them is better 
				if (Q_up > Q) | (Q_down > Q):
					if Q_up > Q_down:
						theta = np.array([(td + alpha) if k == d else t for k,t in enumerate(theta) ])
						Q = Q_up
					else:
						theta = np.array([(td - alpha) if k == d else t for k,t in enumerate(theta) ])
						
						Q = Q_down
					theta_new = np.array([td if k == d else t for k,t in enumerate(theta_new) ])
				else:
					theta_new = np.array([t for t in theta])
				del evecsRot
				
		#stopping criteria
		if it > 2:
			if Q - Q_old2 < 0.01: #if we loose too much quality, stop iterations and output C running value
				break
		Q_old2 = Q_old1
		Q_old1 = Q
		
	if verbose: print('Done after {0} iterations, quality Q={1}'.format(it,Q))

	evecsRot = rotate_givens(evecs, theta_new, ik, jk, angle_num, dim)
	clusts = cluster_assign(evecsRot, ik, jk, dim, ndata)
	
	return Q, clusts, evecsRot


def cluster_rotate(evecs, evals, group_num, method, verbose):

	group_num = sorted(group_num)
	#find 
	group_num2 = [k for k in group_num if k!= 1]
	if verbose: print('group_num:', group_num)
	ndata, dim = evecs.shape #dim will correspond to the number of clusters
	
	mag = np.zeros(shape=[dim], dtype=float)
	for i in range(dim):
		mag[i] = np.linalg.norm(evecs[:,i])

	ind_mag = np.argsort(mag)

	Xevecs = np.empty(shape=evecs.shape, dtype=float)
	for k in range(dim):
		Xevecs[:, k] = evecs[:,ind_mag[dim - k -1]]
		mag[k] = np.linalg.norm(Xevecs[:,k])
	
	#rotate eigenvectors
	Vcurr = Xevecs[:, 0:group_num2[0]]
	Vr = {}
	Quality = np.zeros(shape=[len(group_num2)], dtype=float)
	for g in range(len(group_num2)):
		if verbose: print('Vucrr shape:', Vcurr.shape)

		#make it incremental (used already aligned vectors)
		if g > 0:
			Vcurr = np.concatenate((Vr[g-1] , Xevecs[:, group_num2[g]].reshape(ndata,-1)), axis=1)
		Quality[g], clusts, Vr[g] = evrot(Vcurr, method, verbose)
	i = np.where(np.max(Quality)-Quality <= 0.003)[0]
	print('Quality vector:', Quality)
	return i[-1]

