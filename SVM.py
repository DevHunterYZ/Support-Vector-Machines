from sklearn import datasets		
from sklearn import svm    		
import numpy as np
import matplotlib.pyplot as plt
iris_dataset = datasets.load_iris() # Iris veriseti

iris = datasets.load_iris()
X = iris.data[:, :2]  # Sadece Sepal iki özelliğini alıyoruz.
y = iris.target
C = 1.0  # SVM normalleştirme parametresi

# Lineer çekirdekli Destek Vektör Sınıflandırıcısı(SVC)
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
# LinearSVC (linear kernel)
lin_svc = svm.LinearSVC(C=C).fit(X, y)
# RBF çekirdeği ile SVC
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
# Polinom (derece 3) çekirdeği olan SVC
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)

h = .02  # Kafesteki adım büyüklüğü.

# Çizmek için bir kafes oluşturalım.
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
	                     np.arange(y_min, y_max, h))
# Parseller için başlık belirleyelim.
titles = ['SVC with linear kernel',
	   'LinearSVC (linear kernel)',
	    'SVC with RBF kernel',
	    'SVC with polynomial (degree 3) kernel']


for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
	 # Karar sınırını çizin. Bunun için her birine bir renk atayacağız.
	 # Kafesin içindeki nokta [x_min, x_max] x [y_min, y_max].
	 plt.subplot(2, 2, i + 1)
	 plt.subplots_adjust(wspace=0.4, hspace=0.4)

	 Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

	 # Sonucu renk alanı haline getirin.
	 Z = Z.reshape(xx.shape)
	 plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

	 # Ayrıca eğitim noktalarını da çiziniz.
	 plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
	 plt.xlabel('Sepal length')
	 plt.ylabel('Sepal width')
	 plt.xlim(xx.min(), xx.max())
	 plt.ylim(yy.min(), yy.max())
	 plt.xticks(())
	 plt.yticks(())
	 plt.title(titles[i])

plt.show()
