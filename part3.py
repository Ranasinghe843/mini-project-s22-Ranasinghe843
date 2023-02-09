import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

dataset_1 = pandas.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
dataset_1['Total']  = pandas.to_numeric(dataset_1['Total'].replace(',','', regex=True))

X = dataset_1[["Total"]].values
Y = dataset_1[["Precipitation"]].values

y_transformed = []

for data in Y:
    if data > 0:
        y_transformed.append(1)
    else:
        y_transformed.append(0)
        
y_transformed = np.array(y_transformed)
        
X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.2, random_state=1)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model score: ", accuracy_score(y_test, y_pred))
print(y_test)
print(y_pred)
con_mat = confusion_matrix(y_test, y_pred)
print(con_mat)