import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


#Input Data and corresponding labels
# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

test_data = [[190, 70, 43],[154, 75, 42],[181,65,40]]
test_labels = ['male','male','male']


#Decision Tree Classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

prediction = clf.predict(test_data)
print(prediction)

#Random Forest Classifier
r_clf = RandomForestClassifier(n_estimators=10)
r_clf.fit(X,Y)
r_prediction = clf.predict(test_data)
print(r_prediction)

#Logistic Regression
l_clf = LogisticRegression()
l_clf.fit(X,Y)
l_prediction = l_clf.predict(test_data)
print(l_prediction)

#Support Vector Classifier
s_clf = SVC(gamma='scale')
s_clf.fit(X,Y)
s_prediction = s_clf.predict(test_data)
print(s_prediction)

#Comparing accuracy of different classifiers

classifiers = ['Decision Tree', 'Random Forest', 'Logistic Regression' , 'SVC']
#Decision Tree
d_tree_acc = accuracy_score(prediction,test_labels)
#Random Forest
random_acc = accuracy_score(r_prediction,test_labels)
#Logistic Regression
logistic_acc = accuracy_score(l_prediction,test_labels)
#SVC
support_acc = accuracy_score(s_prediction,test_labels)

accuracy = np.array([d_tree_acc, random_acc, logistic_acc, support_acc])
max_acc = np.argmax(accuracy)
print(accuracy)
print(classifiers[max_acc] + ' is the best classifier for this problem')