from sklearn import tree

features = [[90, 1], [85, 1], [88, 1], [120, 2], [130, 2], [150, 2]]
labels = [1, 2, 1, 2, 2, 2]

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(features, labels)

print(classifier.predict([[80, 2]]))
