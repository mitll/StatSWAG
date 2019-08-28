from __future__ import division

import numpy as np
import unittest
from statswag.datasets import make_classification_labels
from statswag.metrics import nan_accuracy

from statswag.estimators import MajorityVote,IWMV,Spectral,Agreement,MLEOneParameterPerLabeler

# All fit functions should return a dictionary containing these elements
FIT_RETURN_SET = set(['labels','probs','class_names','accuracies'])

# The matrix from the Lehner paper to be used as a medium complexity matrix
MED_MATRIX = np.array([[3, 4, 3, 3, 1],
					[2, 4, 3, 3, 3],
					[3, 3, 4, 3, 3],
					[2, 2, 4, 4, 2],
					[1, 2, 2, 2, 2],
					[3, 2, 4, 1, 1],
					[1, 1, 1, 1, 1],
					[1, 4, 2, 3, 3],
					[4, 2, 1, 1, 4],
					[1, 4, 1, 2, 2]])

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


class TestMajorityVote(unittest.TestCase):

	def test_return_fields(self):
		mv_obj = MajorityVote()
		result = mv_obj.fit(MED_MATRIX)
		self.assertEqual(set(result.keys()),FIT_RETURN_SET)

	def test_fit(self):
		# Since some labels are tied for most popular, pass a fixed random seed
		mv_obj = MajorityVote(hold_out=True)
		result = mv_obj.fit(MED_MATRIX)
		self.assertTrue(result['labels'][0] == 3)
		self.assertTrue(result['labels'][1] == 3)
		self.assertTrue(result['labels'][2] == 3)
		self.assertTrue(result['labels'][3] == 2)
		self.assertTrue(result['labels'][4] == 2)
		self.assertTrue(result['labels'][5] == 1)
		self.assertTrue(result['labels'][6] == 1)
		self.assertTrue(result['labels'][7] == 3)
		self.assertTrue(result['labels'][8] == 1 or result['labels'][8] == 4)
		self.assertTrue(result['labels'][9] == 1 or result['labels'][9] == 2)
		self.assertTrue(result['accuracies'][0] in [0.3,0.4])
		self.assertTrue(result['accuracies'][1] in [0.3,0.4])
		self.assertTrue(result['accuracies'][2] in [0.4])
		self.assertTrue(result['accuracies'][3] in [0.5,0.6,0.7])
		self.assertTrue(result['accuracies'][4] in [0.4,0.5,0.6,0.7])
		self.assertListEqual(result['class_names'].tolist(),[1,2,3,4])

	def test_fit_no_hold(self):
		mv_obj = MajorityVote(hold_out=False)
		result = mv_obj.fit(MED_MATRIX)
		self.assertTrue(result['labels'][0] == 3)
		self.assertTrue(result['labels'][1] == 3)
		self.assertTrue(result['labels'][2] == 3)
		self.assertTrue(result['labels'][3] == 2)
		self.assertTrue(result['labels'][4] == 2)
		self.assertTrue(result['labels'][5] == 1)
		self.assertTrue(result['labels'][6] == 1)
		self.assertTrue(result['labels'][7] == 3)
		self.assertTrue(result['labels'][8] == 1 or result['labels'][8] == 4)
		self.assertTrue(result['labels'][9] == 1 or result['labels'][9] == 2)
		self.assertTrue(result['accuracies'][0] in [0.4, 0.5, 0.6])
		self.assertTrue(result['accuracies'][1] in [0.4])
		self.assertTrue(result['accuracies'][2] in [0.4, 0.5, 0.6])
		self.assertTrue(result['accuracies'][3] in [0.7, 0.8, 0.9])
		self.assertTrue(result['accuracies'][4] in [0.7, 0.8, 0.9])
		self.assertListEqual(result['class_names'].tolist(),[1,2,3,4])


class TestIWMV(unittest.TestCase):

	def test_return_fields(self):
		iwmv_obj = IWMV()
		result = iwmv_obj.fit(MED_MATRIX)
		self.assertEqual(set(result.keys()),FIT_RETURN_SET)

	def test_one(self):
		labels, truth = create_sim_dataset(1000, 6, 2, 0.8, 0.8, False, False, False)
		iwmv_obj = IWMV()
		result = iwmv_obj.fit(labels)
		accuracy = result['accuracies']
		self.assertTrue(np.all(np.abs(accuracy - truth) < 0.075))

	def test_two(self):
		labels, truth = create_sim_dataset(1000, 6, 2, 0.8, 0.8, True, False, False)
		iwmv_obj = IWMV()
		result = iwmv_obj.fit(labels)
		accuracy = result['accuracies']
		self.assertTrue(np.all(np.abs(accuracy - truth) < 0.075))

	def test_three(self):
		labels, truth = create_sim_dataset(1000, 6, 3, 0.8, 0.8, False, False, False)
		iwmv_obj = IWMV()
		result = iwmv_obj.fit(labels)
		accuracy = result['accuracies']
		self.assertTrue(np.all(np.abs(accuracy - truth) < 0.075))

	def test_four(self):
		labels, truth = create_sim_dataset(1000, 6, 3, 0.8, 0.8, True, False, False)
		iwmv_obj = IWMV()
		result = iwmv_obj.fit(labels)
		accuracy = result['accuracies']
		self.assertTrue(np.all(np.abs(accuracy - truth) < 0.075))


class TestAgreement(unittest.TestCase):

	def test_return_fields(self):
		agree_obj = Agreement()
		result = agree_obj.fit(MED_MATRIX)
		self.assertEqual(set(result.keys()),FIT_RETURN_SET)

	@classmethod
	def setUp(self):
		# Some global variables need to be set to test methods called as part of "fit"
		self.model = Agreement()
		self.model.num_classes = 4
		self.model.class_names = [1,2,3,4]

	def test_agreement(self):
		agreement = self.model._percent_agreement(MED_MATRIX[:,:4])
		self.assertEqual(agreement,1/3)

	def test_accuracy(self):
		accuracy = self.model._prob_correct(MED_MATRIX[:,:4])
		self.assertEqual(accuracy,0.5)

	def test_base_rates(self):
		base_rates = self.model._base_rates(MED_MATRIX[:,:4])
		base_rates_GT = {1:0.325,2:0.25,3:0.25,4:0.175}
		for i in range(self.model.num_classes):
			rate_expected = base_rates_GT[self.model.class_names[i]]
			rate_actual = base_rates[i]
			self.assertAlmostEqual(rate_expected,rate_actual,places=3)

	def test_ground_truths(self):
		ground_truth_probs = self.model._ground_truth_probs(MED_MATRIX[:,:4])
		ground_truth_probs_GT = {1:0.041,2:0.032,3:0.860,4:0.067}
		for i in range(self.model.num_classes):
			prob_expected = ground_truth_probs_GT[self.model.class_names[i]]
			prob_actual = ground_truth_probs[0][i]
			self.assertAlmostEqual(prob_expected,prob_actual,places=3)

	def test_sys_accuracy(self):
		sys_accuracy = self.model._sys_accuracy(MED_MATRIX[:,:4],MED_MATRIX[:,4])
		self.assertAlmostEqual(sys_accuracy,0.731,places=3)

	def test_fit_med(self):
		new_model = Agreement()
		result = new_model.fit(MED_MATRIX)
		self.assertAlmostEqual(result['accuracies'][4], 0.731, places=3)

	def test_generate(self):
		labels = self.model._generate_labels(MED_MATRIX)
		labels_GT = [3,3,3,2,2,1,1,3,1,1]
		self.assertListEqual(labels,labels_GT)


class TestMLE(unittest.TestCase):

	def test_return_fields(self):
		mle_obj = MLEOneParameterPerLabeler()
		result = mle_obj.fit(MED_MATRIX)
		self.assertEqual(set(result.keys()),FIT_RETURN_SET)

	def test_one(self):
		labels, truth = create_sim_dataset(1000, 6, 2, 0.8, 0.8, False, False, False)
		MLE = MLEOneParameterPerLabeler()
		ll_list = []
		results_list = []
		for i in range(5):
			results_list.append(MLE.fit(labels))
			ll_list.append(MLE.expert_models.log_likelihood(labels,results_list[i]['class_names']))
		index = np.argmax(np.asarray(ll_list))
		accuracy = results_list[index]['accuracies']
		self.assertTrue(np.all(np.abs(accuracy-truth)<0.075))

	def test_two(self):
		labels, truth = create_sim_dataset(1000, 6, 2, 0.8, 0.8, True, False, False)
		MLE = MLEOneParameterPerLabeler()
		ll_list = []
		results_list = []
		for i in range(5):
			results_list.append(MLE.fit(labels))
			ll_list.append(MLE.expert_models.log_likelihood(labels,results_list[i]['class_names']))
		index = np.argmax(np.asarray(ll_list))
		accuracy = results_list[index]['accuracies']
		self.assertTrue(np.all(np.abs(accuracy-truth)<0.075))

	def test_three(self):
		labels, truth = create_sim_dataset(1000, 6, 3, 0.8, 0.8, False, False, False)
		MLE = MLEOneParameterPerLabeler()
		ll_list = []
		results_list = []
		for i in range(5):
			results_list.append(MLE.fit(labels))
			ll_list.append(MLE.expert_models.log_likelihood(labels,results_list[i]['class_names']))
		index = np.argmax(np.asarray(ll_list))
		accuracy = results_list[index]['accuracies']
		self.assertTrue(np.all(np.abs(accuracy-truth)<0.075))

	def test_four(self):
		labels, truth = create_sim_dataset(1000, 6, 3, 0.8, 0.8, True, False, False)
		MLE = MLEOneParameterPerLabeler()
		ll_list = []
		results_list = []
		for i in range(5):
			results_list.append(MLE.fit(labels))
			ll_list.append(MLE.expert_models.log_likelihood(labels,results_list[i]['class_names']))
		index = np.argmax(np.asarray(ll_list))
		accuracy = results_list[index]['accuracies']
		self.assertTrue(np.all(np.abs(accuracy-truth)<0.075))

	# def test_fit(self):
	# 	mle_obj = MLEOneParameterPerLabeler()
	# 	result = mle_obj.fit(SMALL_MATRIX)
	# 	self.assertListEqual(result['accuracies'].tolist(), [1, 1, 1])
	# 	self.assertListEqual(result['labels'].tolist(),list(SMALL_LABELS))
	#
	# def test_fit_med(self):
	# 	mle_obj = MLEOneParameterPerLabeler()
	# 	result = mle_obj.fit(MED_MATRIX)
	# 	print(result)
	# 	self.assertListEqual(result['labels'].tolist(),list([2,3,33,3]))


class TestSpectral(unittest.TestCase):

	def test_return_fields(self):
		spectral_obj = Spectral()
		result = spectral_obj.fit(MED_MATRIX)
		self.assertEqual(set(result.keys()),FIT_RETURN_SET)

	def test_one(self):
		labels, truth = create_sim_dataset(1000, 6, 2, 0.8, 0.8, False, False, False)
		spectral_obj = Spectral()
		result = spectral_obj.fit(labels)
		accuracy = result['accuracies']
		self.assertTrue(np.all(np.abs(accuracy - truth) < 0.075))

	def test_two(self):
		labels, truth = create_sim_dataset(1000, 6, 2, 0.8, 0.8, True, False, False)
		spectral_obj = Spectral()
		result = spectral_obj.fit(labels)
		accuracy = result['accuracies']
		self.assertTrue(np.all(np.abs(accuracy - truth) < 0.075))

	def test_three(self):
		labels, truth = create_sim_dataset(1000, 6, 3, 0.8, 0.8, False, False, False)
		spectral_obj = Spectral()
		result = spectral_obj.fit(labels)
		accuracy = result['accuracies']
		self.assertTrue(np.all(np.abs(accuracy - truth) < 0.075))

	def test_four(self):
		labels, truth = create_sim_dataset(1000, 6, 3, 0.8, 0.8, True, False, False)
		spectral_obj = Spectral()
		result = spectral_obj.fit(labels)
		accuracy = result['accuracies']
		self.assertTrue(np.all(np.abs(accuracy - truth) < 0.075))


if __name__ == '__main__':
	unittest.main()
