# -*- coding:utf-8 -*-
from __future__ import division
import numpy as np
import os
import math
import pickle as Pickle
import pynlpir
from math import log
from numpy import linalg as LA
from QLM_utils import *


class GQLM(object):

	def __init__(self,lr = 1e-4, alpha = (0.5,0.5), tmax = 0.85, threshold_values = (1e-7,1e-7,1e-7)):
		self.lr = lr
		self.alpha = alpha
		self.tmax = tmax
		self.threshold_values = threshold_values

	def prepare(self,dictionary, init_rho = None):
		self.dictionary = dictionary

		for onevalue in dictionary:
			self.dim = dictionary[onevalue][1].shape[0]
			break
		if init_rho == None:
			self.init_rho = np.eye(self.dim,self.dim)/ self.dim
		else:
			self.init_rho = init_rho

		self.rhoM = self.init_rho

	def train(self):
		# print('initial density matrix:{}'.format(np.trace(self.init_rho)))
		# print('initial_score: {}'.format(F(self.rhoM, self.dictionary)))
		iter_num = 1
		f_old = 0
		t = self.tmax
		while judgement(self.rhoM, self.dictionary,f_old,self.dim, self.threshold_values):
			f_old = F(self.rhoM, self.dictionary)
			t = set_t(t)
			flag = False

			iter_D = D(t, self.rhoM, self.dictionary,self.dim)
			# print(iter_D)
			# print np.trace(np.dot(Grad_F(rhoM, proDict), iter_D))
			while judge_t(t, iter_D, self.rhoM, self.dictionary,self.dim, self.lr):
				# print(t)
				ttt = np.trace(np.dot(Grad_F(self.rhoM, self.dictionary,self.dim), iter_D))
				# print np.trace(np.dot(Grad_F(rhoM, proDict), iter_D))
				if ttt <= 0:
					flag = True
					break

				t *= np.random.uniform(self.alpha[0], self.alpha[1])
				# Should be t belongs to [ao*t, a1*t]

				# print t
				# if t < bound_t(t):
				# 	flag = True
				# 	break
				iter_D = D(t, self.rhoM, self.dictionary,self.dim)
			if flag:
				break
			self.rhoM += t * iter_D

			# print('iter {}: target function value {}'.format(iter_num,F(self.rhoM, self.dictionary)))
			iter_num = iter_num+1
			if(iter_num > 15):
				break
		# print(self.rhoM)
		self.max_value = F(self.rhoM, self.dictionary)



if __name__ == '__main__':
	vector_dim = 3
	projector_num = 200
	test_dict = test_set_generator(vector_dim, projector_num)

	gqlm = GQLM()
	gqlm.prepare(test_dict)

	gqlm.train()
	print('The optimal density matrix is: {}\n The maximum value of the loglikelihood function is: {}\n'.format(gqlm.rhoM, gqlm.max_value))
