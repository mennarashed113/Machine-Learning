import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from KNN import KNN

file_path = "diabetes.csv"
data = pd.read_csv(file_path)

X = data.drop(['Outcome'], axis=1)
y = data[['Outcome']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=34)

X_test=np.array(X_test).reshape((-1,8))
X_train=np.array(X_train).reshape((-1,8))
y_test=np.array(y_test).reshape((-1,1))
y_train=np.array(y_train).reshape((-1,1))

def normlization(X):
    return np.array(X - X.min()/ (X.max() - X.min()))


x_train_final=normlization(X_train)
x_test_final=normlization(X_test)
y_train_final=normlization(y_train)
y_test_final=normlization(y_test)



def accur(y_actual, y_predict):
    count= np.sum(y_actual==y_predict)
    print("the number of correct things is ",count)
    print("the total number of instance ",y_actual.shape[0])
    return count/len(y_actual)

K=[2,3,5,6,9,10,13]
acc=[]
count=0

clf = KNN(2)
clf.fit(x_train_final, y_train)
predictions = np.array(clf.predict(x_test_final)).reshape((-1,1))
acc.append(accur(y_test_final,predictions))
print("The Accuracy of KNN at k= ",2)
print(round(acc[count]*100,2)," % \n")
count+=1

# for k in K:
#     clf = KNN(k)
#     clf.fit(x_train_final, y_train)
#     predictions = np.array(clf.predict(x_test_final)).reshape((-1,1))
#     acc.append(accur(y_test_final,predictions))
#     print("The Accuracy of KNN at k= ",k)
#     print(round(acc[count]*100,2)," % \n")
#     count+=1

avg_acc=sum(acc)/len(acc)
print("The average accuracy of the KNN is ",round(avg_acc*100,2),"%")
