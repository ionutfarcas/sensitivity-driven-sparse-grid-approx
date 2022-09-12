import numpy as np
from collections import OrderedDict
from itertools import product

class Multiindex(object):
	
	def __init__(self, dim):
		self._dim = dim

		self.__poly_mindex_dict 	= OrderedDict()
		self.__poly_mindex_dict_key = 0
		self.__poly_prev_pos 		= 0

		self.__poly_mindex_dict[self.__poly_mindex_dict_key] = [0 for d in xrange(dim)]

	def __get_l1_norm(self, vec):
		
		return np.sum(vec)

	def get_successors(self, multiindex):
		
		successors = np.zeros((self._dim, self._dim), dtype=int)

		for i in xrange(self._dim):
			temp 				= np.zeros(self._dim, dtype=int)
			temp[i] 			= 1
			successors[i, :]	= multiindex + temp

		return successors

	def get_successors_boundary_set(self, multiindex_set_boundary):
		
		successors_boundary_set = []

		for multiindex in multiindex_set_boundary:
			successors = self.get_successors(multiindex)

			for successor in successors:
				successor = successor.tolist()

				if successor not in successors_boundary_set:
					successors_boundary_set.append(successor)

		successors_boundary_set = np.array(successors_boundary_set, dtype=int)

		return successors_boundary_set

	def get_std_total_degree_mindex_level(self, level):
		
		multiindex_set = []

		init_multiindex = np.ones(self._dim).tolist()
		multiindex_set.append(init_multiindex)

		prev_pos = 0
		curr_pos = len(multiindex_set)

		not_finished = True

		if level == 1:
			not_finished = False

		while not_finished:
			for i in xrange(prev_pos, curr_pos):

				successors = self.get_successors(multiindex_set[i]).tolist()
				for successor in successors:
					if np.sum(successor) <= self._dim + level - 1 and successor not in multiindex_set:
						multiindex_set.append(successor)

					if np.sum(successor) > self._dim + level - 1:
						not_finished = False

			prev_pos = curr_pos
			curr_pos = len(multiindex_set)

		multiindex_set = [multiindex for multiindex in multiindex_set if self.__get_l1_norm(multiindex) == self._dim + level - 1]

		multiindex_set = np.array(multiindex_set, dtype=int)

		return multiindex_set

	def get_std_total_degree_mindex(self, level):
		
		multiindex_set = []

		init_multiindex = np.ones(self._dim).tolist()
		multiindex_set.append(init_multiindex)

		prev_pos = 0
		curr_pos = len(multiindex_set)

		not_finished = True

		if level == 1:
			not_finished = False

		while not_finished:
			for i in xrange(prev_pos, curr_pos):

				successors = self.get_successors(multiindex_set[i]).tolist()
				for successor in successors:
					if np.sum(successor) <= self._dim + level - 1 and successor not in multiindex_set:
						multiindex_set.append(successor)

					if np.sum(successor) > self._dim + level - 1:
						not_finished = False

			prev_pos = curr_pos
			curr_pos = len(multiindex_set)

		multiindex_set = np.array(multiindex_set, dtype=int)

		return multiindex_set

	def get_poly_mindex(self, level):

		multiindex_set = []

		init_multiindex = np.zeros(self._dim).tolist()
		multiindex_set.append(init_multiindex)

		prev_pos = 0
		curr_pos = len(multiindex_set)

		not_finished = True
		
		while not_finished:
			for i in xrange(prev_pos, curr_pos):

				successors = self.get_successors(multiindex_set[i]).tolist()
				for successor in successors:
					if np.sum(successor) <= level and successor not in multiindex_set:
						multiindex_set.append(successor)

					if np.sum(successor) > level:
						not_finished = False

			prev_pos = curr_pos
			curr_pos = len(multiindex_set)

		multiindex_set = np.array(multiindex_set, dtype=int)

		return multiindex_set

	def get_poly_mindex_binary(self, level):

		multiindex_set = []

		init_multiindex = np.zeros(self._dim).tolist()
		multiindex_set.append(init_multiindex)

		prev_pos = 0
		curr_pos = len(multiindex_set)

		not_finished = True
		
		while not_finished:
			for i in xrange(prev_pos, curr_pos):

				successors = self.get_successors(multiindex_set[i]).tolist()
				for successor in successors:
					if np.amax(successor) == 1 and  np.sum(successor) <= level and successor not in multiindex_set:
						multiindex_set.append(successor)

					if np.sum(successor) > level:
						not_finished = False

			prev_pos = curr_pos
			curr_pos = len(multiindex_set)

		multiindex_set = np.array(multiindex_set, dtype=int)

		return multiindex_set[1:]

	def get_poly_degs(self, max_degs):

		multiindex_set = []

		init_multiindex = np.zeros(self._dim).tolist()
		multiindex_set.append(init_multiindex)

		prev_pos = 0
		curr_pos = len(multiindex_set)

		not_finished = True
		
		while not_finished:
			for i in xrange(prev_pos, curr_pos):

				successors = self.get_successors(multiindex_set[i]).tolist()
				for successor in successors:
					if successor not in multiindex_set:
						truth = 1

						for d in xrange(self._dim):
							if successor[d] > max_degs[d]:
								truth = 0
								not_finished = False
						if truth:
							multiindex_set.append(successor)						

			prev_pos = curr_pos
			curr_pos = len(multiindex_set)

		multiindex_set = np.array(multiindex_set, dtype=int)

		return multiindex_set