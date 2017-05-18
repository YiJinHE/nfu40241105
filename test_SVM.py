from sklearn import datasets
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)
print clf
mydata=[0,0,0,0,0,0,0,0,
        0,0,0,12,12,12,0,0,
        0,4,4,12,12,12,0,0,
        0,4,4,4,12,21,12,0,0,
        0,4,4,4,5,5,0,0,
        0,4,4,4,5,5,0,0,
        0,0,0,0,5,5,0,0,
        0,0,0,0,0,0,0,0]

mytestdata=[0,0,0,0,0,0,0,0,
            0,0,0,12,12,12,0,0,
            0,4,4,12,12,12,0,0,
            0,4,4,4,12,21,12,0,0,
            0,4,4,4,5,5,0,0,
            0,4,4,4,5,5,0,0,
            0,0,0,0,5,5,0,0,
            0,0,0,0,0,0,0,0]

digits = datasets.load_digits()

#print(len(my.data))
import matplotlib.pyplot as plot
plot.figure(1, figsize=(3, 3))
plot.imshow(digits.images[-10], cmap=plot.cm.gray_r, interpolation='nearest')
plot.show()