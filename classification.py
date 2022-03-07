import pandas as pd
import glob
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
Y=[]
# for i in range(13):
#     Y= np.append(Y,[i]*20)
# Y = np.append(Y,[13]*18)
# Y = np.append(Y,[14]*19)

# for j in range(15,18):
#     Y= np.append(Y,[j]*20)
# print(Y)
y=[]
for i in range(3):
    y= np.append(y,[i]*20)
############################
path = "/home/nest/Downloads/p0"
all_files = glob.glob(path + "/*.csv")
# filename= "/home/nest/project/alldata.csv"
X = []
# df = pd.read_csv(filename, index_col=0, header=0)
# X.append(df)

for filename in all_files:
    df = pd.read_csv(filename, index_col=0, header=0)
    X.append(df)
path1 = "/home/nest/Downloads/p1"
all_files1 = glob.glob(path1 + "/*.csv")


for filename1 in all_files1:
    df = pd.read_csv(filename1, index_col=0, header=0)
    X.append(df)
path2 = "/home/nest/Downloads/p2"
all_files2 = glob.glob(path2 + "/*.csv")
for filename2 in all_files2:
    df = pd.read_csv(filename2, index_col=0, header=0)
    X.append(df)
print(np.shape(X))
X_train, X_test, y_train, y_test = train_test_split( X, y)

################# model#######################
model = OneVsRestClassifier(SVC(random_state=0))
# model= MLPClassifier()
# model = DecisionTreeClassifier(max_depth = 2)
# model= SVC(kernel = 'linear', C = 1)
# model= KNeighborsClassifier(n_neighbors = 7)
# model=GaussianNB()
#print(np.shape(X_train[0]))
# SVC(kernel="linear", C=0.025)
# SVC(gamma=3.5, C=1)
nsamples, nx, ny = np.shape(X_train)
X_train_2d = np.reshape(X_train,(nsamples, nx * ny))

model.fit(X_train_2d, y_train)
#print(np.shape(X_test))
nsamples1, nx1, ny1 = np.shape(X_test)
X_test_2d = np.reshape(X_test,(nsamples1, nx1 * ny1))

prediction = model.predict(X_test_2d)

score = model.score(X_test_2d, y_test)
print(score)
y_pred = model.predict(X_test_2d)
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True)
plt.show()
# print(f"Test Set Accuracy : {accuracy_score( y_test, prediction) * 100} %\n\n")
# print(f"Classification Report : \n\n{classification_report(y_test, prediction)}")
# Evaluating the model
