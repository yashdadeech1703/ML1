import sys
print('Python: {}'.format(sys.version))
import scipy
print('Scipy: {}'.format(scipy._version_))
import numpy
print('Numpy: {}'.format(numpy._version_))
import matplotlib
print('Matplotlib: {}'.format(matplotlib._version_))
import pandas
print('Pandas: {}'.format(pandas._version_))
import sklearn
print('Sklearn: {}'.format(sklearn._version_))

import pandas
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantanalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier

# loading the data
url="https://raw.githubusercontest.com/jbrownlee/Datasets/master.iris.csv"
names=['sepal-length','sepal-width','petal-length','petal-width','class']
dataset=read_csv(url,names=names)

# dimensions of the dataset
print(dataset.shape)

# take a peek at the data
print(dataset.head(20))

# statistical summary
print(dataset.describe())

# class distribution
print(dataset.groupby('class').size())

# univariate plots-box and whisker plots
dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
pyplot.show()

# histogram of the variable
datasets.hist()
pyplot.show()

# multivariate plots
scatter_matrix(dataset)
pyplot.show()

# creating a validation dataset
# splitting dataset
array=dataset.values
X=array[:,0:4]
Y=array[:,4]
X_train,X_validation,Y_train,Y_validation=train_test_split(X,Y,test_size=0.2,random_state=1)

# Logistic Regression
# Linear Discriminant Analysis
# K-Nearest neighbors
# Classification and Regression Trees
# Gaussian Naive Bayes
# Support Vector Machines

# building models
models=[]
models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('NB,GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))

# evaluate the created models
results=[]
names=[]
for name,model in models:
    kfold=StratifiedFold(n_splits=10,random_state=1)
    cv_results=cross_val_score(model,X_train,Y_train,cv=kfold,scoring='accuracy')
    results.append(cv-results)
    names.append(name)
    print('%s:%f(%f)' % (name,cv_results.mean(),cv_results.std()))
    
# compare our models
pyplot.boxplot(results,labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# make predictions on svm
model=SVC(gamma='auto')
model.fit(X_train,Y_train)
predictions=model.predict(X_validation)

# evaluate our predictions
print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))

