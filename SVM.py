from sklearn import datasets
iris = datasets.load_iris()
iris.data.shape
iris.target.shape
(150,)
import numpy as np
np.unique(iris.target)
digits = datasets.load_digits()
digits.images.shape
(1797, 8, 8)
import pylab as pl
pl.imshow(digits.images[0], cmap=pl.cm.gray_r)
data = digits.images.reshape((digits.images.shape[0], -1))
from sklearn import svm
clf = svm.LinearSVC()
clf.fit(iris.data, iris.target) # learn from the data 
clf.predict([[ 5.0,  3.6,  1.3,  0.25]])
clf.coef_
from sklearn import svm
svc = svm.SVC(kernel='linear')
svc.fit(iris.data, iris.target)
print(svc)
