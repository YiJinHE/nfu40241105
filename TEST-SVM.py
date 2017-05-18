from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plot
clf = svm.SVC(gamma=0.001, C=100.)
digits = datasets.load_digits()
clf.fit(digits.data[:-1], digits.target[:-1])


my_data1 = [0,0,0,8,8,2,0,0,
           0,0,0,4,12,4,0,0,
           0,0,1,15,15,2,0,0,
           0,3,15,15,15,1,0,0,
           0,0,15,12,15,15,0,0,
           0,0,20,12,12,20,0,0,
           0,0,1,12,12,1,0,0,
           0,0,0,8,15,4,0,0]

my_data1_img = [[0,0,0,8,8,0,0,0],
               [ 0,0,0,4,12,4,0,0],
               [0,0,1,15,15,2,0,0],
               [0,3,15,15,15,1,0,0],
               [0,0,15,12,15,15,0,0],
               [0,0,20,12,12,20,0,0],
               [0,0,1,12,12,1,0,0],
               [0,0,0,8,15,4,0,0]]


result=clf.predict(my_data1)

print "predict: " ,result
print "actual: " ,my_data1," my_data ans is 1 "

plot.figure(1, figsize=(3, 3))
plot.imshow(my_data1_img, cmap=plot.cm.gray_r, interpolation='nearest')
plot.show()

my_data2 = [00,10,10,10,10,10,10,0,
10,15,15,8,8,15,15,10,
10,15,15,8,8,15,15,10,
10,8,8,8,8,8,8,10,
10,8,8,15,15,8,8,10,
10,15,8,8,8,8,15,10,
10,8,15,8,8,15,8,10,
0,10,10,15,15,10,10,0]

my_data2_img = [[0,10,10,10,10,10,10,0],
                 [10,15,15,8,8,15,15,10],
                 [10,15,15,8,8,15,15,10],
                 [10,8,8,8,8,8,8,10],
                 [10,8,8,15,15,8,8,10],
                 [10,15,8,8,8,8,15,10],
                 [10,8,15,8,8,15,8,10],
                 [0,10,10,15,15,10,10,0]]

result2=clf.predict(my_data2)

print "predict: " ,result2
print "actual: " ,my_data2," my_data ans is 2 "

plot.figure(1, figsize=(3, 3))
plot.imshow(my_data2_img, cmap=plot.cm.gray_r, interpolation='nearest')
plot.show()

