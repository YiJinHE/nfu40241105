宜錦筆記

##Part 1

#coding=utf-8
import numpy as np
from numpy import *

#第一個範例
#矩陣相加錯誤示範
x = [10,20,30]
y = [40,50,60]
print x+y

#正確示範
a = np.array([10, 20, 30])
b = np.array([20, 40, 60])
print a+b

#第二個範例
a=np.array([2,4,5])
b=np.array([-3,2,-1])

la=np.sqrt(a.dot(a))
lb=np.sqrt(b.dot(b))
print("----計算向量長度---")
print (la,lb)

cos_angle=a.dot(b)/(la*lb)

print("----計算cos ----")
print (cos_angle)

angle=np.arccos(cos_angle)

print("----計算夾角(單位為π)----")
print (angle)

angle2=angle*360/2/np.pi
print("----轉換單位為角度----")
print (angle2)

a=np.array([[2, 5], [3, 2]])
b=np.array([[2, 3], [2, 5]])
c=np.mat([[2, 4], [2, 3]])
d=np.mat([[1, 2], [3, 4]])
e=np.dot(a,b)
f=np.dot(c,d)
print("----乘法運算----")
print (a*b)
print (c*d)
print("----矩陣相乘----")
print (e)
print (f)

a=np.random.randint(1, 10, (3, 5))
#a=np.random.randint(1, 10, 8)
print (a)

a = mat([[1, 3, -1], [2, 0, 1], [3, 2, 1]])

print linalg.det(a)

from matplotlib import pyplot

x = np.arange(0,10,0.1)
y = np.sin(x)
pyplot.plot(x,y)
pyplot.show()

##Part 2

#coding=utf-8
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import pandas as pd


num_friends=pd.Series([100,49,41,40,25,21,21,19,19,18,18,16,
15,15,15,15,14,14,13,13,13,13,12,12,11,
10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,
9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,
7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

num_newFriends =  pd.Series(np.sort(np.random.binomial(203,0.06,204))[::-1]) #用A series去建立B series
df_friendsGroup = pd.DataFrame({"A":num_friends,"B":num_newFriends}) #將兩張series合成為一個DataFrame

print("印出Col A")
print(df_friendsGroup["A"])
print("印出Col A及Col B的前10row")
select = df_friendsGroup[["A", "B"]]
print(select.head(10))
print("印出row5")
print(df_friendsGroup.ix[5])
print("印出row5~row9")
print(df_friendsGroup[5:10])

print(df_friendsGroup.describe()) #統計量
print(df_friendsGroup.corr()) #相關係數
print("cov = {}".format(num_friends.cov(num_newFriends))) #共變異數

############圖表###########
plt.hist(df_friendsGroup["A"],bins=25)
plt.hist(df_friendsGroup["B"],bins=25, color="r")
plt.xlabel("# of Friends")
plt.ylabel("# of People")
plt.show()
###Part 2-1 Pandas介紹

呼叫指定的一個ROW
DataFrame[“Column_Name"]
呼叫多個指定ROW
select = DataFrame[[" Column_Name ", " Column_Name 2",……]]
select.head(Col_Number)
呼叫指定的一個COL
DataFrame.ix[index]
呼叫多個指定COL
DataFrame.ix[START:END]
###Part 2-1 Pandas介紹

CSV檔
data = pd.read_csv('file.csv')
EXCEL
data = pd.read_excel('file.xls', 'sheet')
Html
data = pd.read_html('url')
###Part 2-2 資料讀取

############資料讀取###########
dataExcel = pd.read_excel('C:/Users/40341127/Downloads/test.xls')
print(dataExcel)

############統計量###########

print("最大值= {}".format(df_friendsGroup["A"].max()))
print("最小值= {}".format(df_friendsGroup["A"].min()))
print("平均值= {}".format(df_friendsGroup["A"].mean()))
print("變異數= {}".format(df_friendsGroup["A"].var()))
print("標準差= {}".format(df_friendsGroup["A"].std()))
print("中位數= {}".format(df_friendsGroup["A"].median()))
###Part 2-3 Pandas統計量之運用

求最大最小值
DataFrame.max(), DataFrame.min()
求平均值
DataFrame.mean()
求變異數
DataFrame.var()
求標準差
DataFrame.std()
求中位數
DataFrame.median()


#利用describe() 可以更方便查詢
DataFrame.describe()
Data Science from Scratch

Here's all the code and examples from my book Data Science from Scratch. The code directory contains Python 2.7 versions, and the code-python3 direction contains the Python 3 equivalents. (I tested them in 3.5, but they should work in any 3.x.)

Each can be imported as a module, for example (after you cd into the /code directory):

from linear_algebra import distance, vector_mean
v = [1, 2, 3]
w = [4, 5, 6]
print distance(v, w)
print vector_mean([v, w])
Or can be run from the command line to get a demo of what it does (and to execute the examples from the book):

python recommender_systems.py
Additionally, I've collected all the links from the book.

And, by popular demand, I made an index of functions defined in the book, by chapter and page number. The data is in a spreadsheet, or I also made a toy (experimental) searchable webapp.

Table of Contents

Introduction
A Crash Course in Python
Visualizing Data
Linear Algebra
Statistics
Probability
Hypothesis and Inference
Gradient Descent
Getting Data
Working With Data
Machine Learning
k-Nearest Neighbors
Naive Bayes
Simple Linear Regression
Multiple Regression
Logistic Regression
Decision Trees
Neural Networks
Clustering
Natural Language Processing
Network Analysis
Recommender Systems
Databases and SQL
MapReduce
Go Forth And Do Data Science