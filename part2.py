import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

dataset_1 = pandas.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
dataset_1['Total']  = pandas.to_numeric(dataset_1['Total'].replace(',','', regex=True))

X = dataset_1[["High Temp", "Low Temp", "Precipitation"]].values
Y = dataset_1[["Total"]].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# print(y_test.transpose())
# print(y_pred.transpose())
print("r-squared score: ", model.score(X_test, y_test))
print("mean squared error:", mean_squared_error(y_test, y_pred))
print(f"Total = {model.coef_[0][0]}*(High Temp) + {model.coef_[0][1]}*(Low Temp) + {model.coef_[0][2]}*(Precipitation) + {model.intercept_[0]}")