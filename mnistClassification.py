from sklearn.linear_model import *
from sklearn.utils import shuffle
from sklearn.datasets import fetch_mldata  
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.svm import LinearSVC, SVC
from  sklearn.ensemble import GradientBoostingClassifier
import numpy as np

mnist = fetch_mldata('MNIST original')

totalSize = 4000
mnist.data = preprocessing.scale(mnist.data.astype(np.float))
mnist.data, mnist.target = shuffle(mnist.data, mnist.target, random_state = 0)
mnist.data = mnist.data[:totalSize,:]
mnist.target = mnist.target[:totalSize]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(mnist.data, mnist.target, test_size=0.2, random_state=0)

lr = LogisticRegression()
lr.fit(X_train, y_train)
predicted = lr.predict(X_test)
print "logistic regression correct:%d total:%d accuracy:%.4f" %(y_test[predicted == y_test].shape[0], y_test.shape[0], y_test[predicted == y_test].shape[0] * 1.0 / y_test.shape[0])

svm = SVC(kernel = 'linear')
svm.fit(X_train, y_train)
predicted = svm.predict(X_test)
print "linear svm correct:%d total:%d accuracy:%.4f" %(y_test[predicted == y_test].shape[0], y_test.shape[0], y_test[predicted == y_test].shape[0] * 1.0 / y_test.shape[0])

svm = SVC(kernel = 'poly', degree= 2)
svm.fit(X_train, y_train)
predicted = svm.predict(X_test)
print "svm: poly kernel correct:%d total:%d accuracy:%.4f" %(y_test[predicted == y_test].shape[0], y_test.shape[0], y_test[predicted == y_test].shape[0] * 1.0 / y_test.shape[0])

gbdt = GradientBoostingClassifier()
gbdt.fit(X_train, y_train)
predicted = gbdt.predict(X_test)
print "svm: poly kernel correct:%d total:%d accuracy:%.4f" %(y_test[predicted == y_test].shape[0], y_test.shape[0], y_test[predicted == y_test].shape[0] * 1.0 / y_test.shape[0])
