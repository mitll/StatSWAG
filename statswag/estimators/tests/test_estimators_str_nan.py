from __future__ import division

import numpy as np
import unittest

from statswag.datasets import make_classification_labels
from statswag.metrics import nan_accuracy

from statswag.estimators import MajorityVote,IWMV,Spectral,Agreement,MLEOneParameterPerLabeler


MED_MATRIX = np.array([['C','D','C','C','A'],
						['B','D','C','C','C'],
						['C','C','D','C','C'],
						['B','B','D','D','B'],
						['A','B','B','B','B'],
						['C','B','D','A','A'],
						['A','A','A','A','A'],
						['A','D','B','C','C'],
						[np.nan,'B','A','A',np.nan],
						['A','D','A','B','B']],dtype=object)

def create_sim_dataset(num_instances, num_labelers, num_class, first_acc, other_acc, intshift, missing_vals, string):

	string_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p']

	y = np.random.randint(0, num_class, num_instances)

	# Create a confusion matrix for the first labeler, here we use 3 classes
	diagonal_first = first_acc
	off_diagonal_first = (1.0-diagonal_first)/(num_class-1.0)
	confusion_mat_first = np.zeros((num_class,num_class))
	for i in range(num_class):
		for j in range(num_class):
			if i == j:
				confusion_mat_first[i,j] = diagonal_first
			else:
				confusion_mat_first[i,j] = off_diagonal_first


	# Create a confusion matrix for the other labelers
	diagonal_other = other_acc
	off_diagonal_other = (1.0-diagonal_other)/(num_class-1.0)
	confusion_mat_other = np.zeros((num_class,num_class))
	for i in range(num_class):
		for j in range(num_class):
			if i == j:
				confusion_mat_other[i,j] = diagonal_other
			else:
				confusion_mat_other[i,j] = off_diagonal_other

	# From the ground truth labels, produce the predicted labels
	print(confusion_mat_first)
	print(confusion_mat_other)
	labels = make_classification_labels(y=y,
										n_labelers=num_labelers,
										confusion=[confusion_mat_first]+
										[confusion_mat_other for i in range(num_labelers-1)])
	if intshift:
		labels = labels + 1
		y = y + 1

	if missing_vals == True:
		labels = np.asarray(labels, dtype=object)
		for i in range(int(num_instances*num_class/2)):
			ind1 = np.random.randint(0, num_instances)
			ind2 = np.random.randint(0, num_labelers)
			labels[ind1, ind2] = np.nan
	if string == True:
		labels = np.asarray(labels,dtype=object)
		for i in range(num_class):
			labels[labels==i] = string_list[i]
		y = np.asarray(y,dtype=object)
		for i in range(num_class):
			y[y==i] = string_list[i]

	true_labeler_accuracy = np.asarray([nan_accuracy(y, labels[:, col]) for col in range(np.size(labels, axis=1))])

	return labels, true_labeler_accuracy


class TestMLE(unittest.TestCase):

	def test_one(self):
		labels, truth = create_sim_dataset(1000, 6, 2, 0.8, 0.8, False, True, True)
		MLE = MLEOneParameterPerLabeler()
		ll_list = []
		results_list = []
		for i in range(5):
			results_list.append(MLE.fit(labels))
			ll_list.append(MLE.expert_models.log_likelihood(labels,results_list[i]['class_names']))
		index = np.argmax(np.asarray(ll_list))
		accuracy = results_list[index]['accuracies']
		self.assertTrue(np.all(np.abs(accuracy - truth) < 0.075))

	def test_two(self):
		labels, truth = create_sim_dataset(1000, 6, 3, 0.8, 0.8, False, True, True)
		MLE = MLEOneParameterPerLabeler()
		ll_list = []
		results_list = []
		for i in range(5):
			results_list.append(MLE.fit(labels))
			ll_list.append(MLE.expert_models.log_likelihood(labels,results_list[i]['class_names']))
		index = np.argmax(np.asarray(ll_list))
		accuracy = results_list[index]['accuracies']
		self.assertTrue(np.all(np.abs(accuracy - truth) < 0.075))


class TestIWMV(unittest.TestCase):

	def test_one(self):
		labels, truth = create_sim_dataset(1000, 6, 2, 0.8, 0.8, False, True, True)
		iwmv = IWMV()
		result = iwmv.fit(labels)
		accuracy = result['accuracies']
		self.assertTrue(np.all(np.abs(accuracy - truth) < 0.075))

	def test_two(self):
		labels, truth = create_sim_dataset(1000, 6, 3, 0.8, 0.8, False, True, True)
		iwmv = IWMV()
		result = iwmv.fit(labels)
		accuracy = result['accuracies']
		self.assertTrue(np.all(np.abs(accuracy - truth) < 0.075))

class TestMajorityVote(unittest.TestCase):

	def test_fit(self):
		# Since some labels are tied for most popular, pass a fixed random seed
		mv_obj = MajorityVote(hold_out=True)
		result = mv_obj.fit(MED_MATRIX)
		print(result['accuracies'])
		self.assertTrue(result['labels'][0] == 'C')
		self.assertTrue(result['labels'][1] == 'C')
		self.assertTrue(result['labels'][2] == 'C')
		self.assertTrue(result['labels'][3] == 'B')
		self.assertTrue(result['labels'][4] == 'B')
		self.assertTrue(result['labels'][5] == 'A')
		self.assertTrue(result['labels'][6] == 'A')
		self.assertTrue(result['labels'][7] == 'C')
		self.assertTrue(result['labels'][8] == 'A')
		self.assertTrue(result['labels'][9] == 'A' or result['labels'][9] == 'B')
		self.assertTrue(result['accuracies'][0] in [3.0/9.0, 4.0/9.0])
		self.assertTrue(result['accuracies'][1] in [0.3,0.4])
		self.assertTrue(result['accuracies'][2] in [0.4, 0.5])
		self.assertTrue(result['accuracies'][3] in [0.5,0.6,0.7,0.8])
		self.assertTrue(result['accuracies'][4] in [4.0/9.0,5.0/9.0,6.0/9.0,7.0/9.0])
		self.assertListEqual(result['class_names'].tolist(),['A','B','C','D'])

	def test_fit_no_hold(self):
		mv_obj = MajorityVote(hold_out=False)
		result = mv_obj.fit(MED_MATRIX)
		self.assertTrue(result['labels'][0] == 'C')
		self.assertTrue(result['labels'][1] == 'C')
		self.assertTrue(result['labels'][2] == 'C')
		self.assertTrue(result['labels'][3] == 'B')
		self.assertTrue(result['labels'][4] == 'B')
		self.assertTrue(result['labels'][5] == 'A')
		self.assertTrue(result['labels'][6] == 'A')
		self.assertTrue(result['labels'][7] == 'C')
		self.assertTrue(result['labels'][8] == 'A')
		self.assertTrue(result['labels'][9] == 'A' or result['labels'][9] == 'B')
		self.assertTrue(result['accuracies'][0] in [4.0/9.0,5.0/9.0])
		self.assertTrue(result['accuracies'][1] in [0.4])
		self.assertTrue(result['accuracies'][2] in [0.5, 0.6])
		self.assertTrue(result['accuracies'][3] in [0.8, 0.9])
		self.assertTrue(result['accuracies'][4] in [7.0/9.0, 8.0/9.0])
		self.assertListEqual(result['class_names'].tolist(),['A','B','C','D'])

if __name__ == '__main__':
	unittest.main()
