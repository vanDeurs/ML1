from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

iris = load_iris()
testing_idx = [0, 50, 100]
# print(iris.feature_names)
# print(iris.target_names)
# print(iris.data[0])
#
# for i in range(len(iris.data)):
#     print("Example %d: labels: %s, features: %s" % (i, iris.target[i], iris.data[i]))

# training data
training_target = np.delete(iris.target, testing_idx)
training_data = np.delete(iris.data, testing_idx, axis=0)

# testing data
testing_target = iris.target[testing_idx]
testing_data = iris.data[testing_idx]

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(training_data, training_target)

print(testing_target)
print(classifier.predict(testing_data))
