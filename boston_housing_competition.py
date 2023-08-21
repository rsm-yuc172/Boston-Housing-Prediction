"""
File: boston_housing_competition.py
Name: Jack Chen
--------------------------------
This file demonstrates how to analyze boston
housing dataset. Students will upload their 
results to kaggle.com and compete with people
in class!

You are allowed to use pandas, sklearn, or build the
model from scratch! Go data scientists!
"""

import pandas as pd
from sklearn import preprocessing, linear_model, metrics, model_selection, tree, ensemble, svm
from scipy import stats
import numpy as np



TRAIN_FILE = 'boston_housing/train.csv'
TEST_FILE = 'boston_housing/test.csv'

col2remove = []
test_ID = []
ALPHA = 0.001


def main():
	total_x = 0
	total_x_val = 0
	for i in range(30):
		train_data, Y, val_data, val_Y = data_preprocess(TRAIN_FILE, mode='Train')
		test_data = data_preprocess(TEST_FILE, mode='Test')

		### Normalization ###
		normalizer = preprocessing.MinMaxScaler()
		X = normalizer.fit_transform(train_data)
		X_val = normalizer.transform(val_data)
		X_test = normalizer.transform(test_data)

		### Polynomial ###
		poly_phi = preprocessing.PolynomialFeatures(degree=2)
		X = poly_phi.fit_transform(X)
		X_val = poly_phi.transform(X_val)
		X_test = poly_phi.transform(X_test)

		bagging = ensemble.BaggingRegressor(base_estimator=ensemble.
											GradientBoostingRegressor(alpha=0.9, learning_rate=0.1), bootstrap_features=True, bootstrap=False)
		bag_classifier = bagging.fit(X, Y)
		kaggle_prediction = bag_classifier.predict(X_test)
		prediction_x = bag_classifier.predict(X)
		prediction_xval = bag_classifier.predict(X_val)

		total_x += metrics.mean_squared_error(prediction_x, Y) ** 0.5
		total_x_val += metrics.mean_squared_error(prediction_xval, val_Y) ** 0.5

	print(total_x/30)
	print(total_x_val/30)
	### Output Prediction ###
	print(kaggle_prediction)
	outfile(kaggle_prediction, 'D2_bag(GBR)_chi**2_2.csv')

	### Training ###
	# h = linear_model.LinearRegression()
	# classifier = h.fit(X, Y)
	#
	# # Make predictions on the test data
	# prediction = classifier.predict(X)
	# print(metrics.mean_squared_error(prediction, Y)**0.5)

	# Construct Forest
	# forest = ensemble.RandomForestRegressor(max_depth=4, min_samples_split=4, min_samples_leaf=10)
	# forest_classifier = forest.fit(X, Y)
	# prediction = forest_classifier.predict(X)
	# print(metrics.mean_squared_error(prediction, Y) ** 0.5)
	# prediction = forest_classifier.predict(X_val)
	# print(metrics.mean_squared_error(prediction, val_Y) ** 0.5)

	# SVM Model
	# SVC = svm.LinearSVR()
	# SVC_classifier = SVC.fit(X, Y)
	# prediction = SVC_classifier.predict(X)
	# print(metrics.mean_squared_error(prediction, Y) ** 0.5)
	# prediction = SVC_classifier.predict(X_val)
	# print(metrics.mean_squared_error(prediction, val_Y) ** 0.5)

	# Ada boost

	# ada = ensemble.AdaBoostRegressor()
	# ada_classifier = ada.fit(X, Y)
	# prediction = ada_classifier.predict(X)
	# print(metrics.mean_squared_error(prediction, Y) ** 0.5)
	# prediction = ada_classifier.predict(X_val)
	# print(metrics.mean_squared_error(prediction, val_Y) ** 0.5)

	# bagging = ensemble.BaggingRegressor(base_estimator=ensemble.GradientBoostingRegressor(min_samples_leaf=15, min_samples_split=12))
	# bag_classifier = bagging.fit(X, Y)
	# prediction = bag_classifier.predict(X)
	# print(metrics.mean_squared_error(prediction, Y) ** 0.5)
	# prediction = bag_classifier.predict(X_val)
	# print(metrics.mean_squared_error(prediction, val_Y) ** 0.5)


def data_preprocess(filename: str, mode='Train'):
	global col2remove
	data = pd.read_csv(filename)
	# data.pop('chas')  # classification problem
	pd.get_dummies(data, columns=['chas'])
	if mode == 'Train':
		data.pop('ID')
		train_data, val_data = model_selection.train_test_split(data, test_size=0.4)
		Y = train_data.pop('medv')
		val_Y = val_data.pop('medv')
		### Chi-Square Test ### -> Removing the cols that are independent of medv
		for col in train_data.columns:
			cross_tab = pd.crosstab(train_data[col], Y)
			if stats.chi2_contingency(cross_tab)[1] <= ALPHA:
				col2remove.append(col)
				train_data.pop(col)
				val_data.pop(col)
		# k = ensemble.GradientBoostingRegressor()
		# k.fit(train_data, Y)
		# f_select = pd.Series(k.feature_importances_, index=train_data.columns)
		# final_features = f_select.nlargest(8).index
		### Alter outliers ###
		# for col in train_data.columns:
		# 	train_std = train_data[col].std()
		# 	train_mean = train_data[col].mean()
		# 	# for i in range(train_data[col].count()):
		# 	for d in train_data[col]:
		# 		# z = (train_data[col][i] - train_mean) / train_std
		# 		z = (d - train_mean) / train_std
		# 		if z > 3 or z < -3:
		# 			train_data[col].replace(d, train_data[col].median(), inplace=True)
		return train_data, Y, val_data, val_Y
	else:
		for col in col2remove:  # dropping the cols that was removed from chi-square test
			data.pop(col)
		col2remove = []
		[test_ID.append(num) for num in data['ID']]  # getting the IDs
		data.pop('ID')
		return data


def outfile(prediction, filename):
	with open(filename, 'w') as out:
		out.write('ID,medv\n')
		for i in range(len(prediction)):
			out.write(f'{test_ID[i]},{prediction[i]}\n')


if __name__ == '__main__':
	main()
