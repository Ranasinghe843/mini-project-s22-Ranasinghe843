import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset_1 = pandas.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
dataset_1['Brooklyn Bridge']      = pandas.to_numeric(dataset_1['Brooklyn Bridge'].replace(',','', regex=True))
dataset_1['Manhattan Bridge']     = pandas.to_numeric(dataset_1['Manhattan Bridge'].replace(',','', regex=True))
dataset_1['Queensboro Bridge']    = pandas.to_numeric(dataset_1['Queensboro Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
dataset_1['Total']  = pandas.to_numeric(dataset_1['Total'].replace(',','', regex=True))

def score_three_bridges(bridge1, bridge2, bridge3):

    X = dataset_1[[f"{bridge1} Bridge", f"{bridge2} Bridge", f"{bridge3} Bridge"]].values
    Y = dataset_1[["Total"]].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    PredictScore = model.score(X_test, y_test)

    return model, PredictScore

r_scores = [0, 0, 0, 0]
# Brooklyn, Manhattan, Williamsburg
brook_man_will_Model, r_scores[0] = score_three_bridges("Brooklyn", "Manhattan", "Williamsburg")

# Brooklyn, Manhattan, Queensboro
brook_man_queens_Model, r_scores[1] = score_three_bridges("Brooklyn", "Manhattan", "Queensboro")

# Brooklyn, Williamsburg, Queensboro
brook_will_queens_Model, r_scores[2] = score_three_bridges("Brooklyn", "Williamsburg", "Queensboro")

# Manhattan, Williamsburg, Queensboro
man_will_queens_Model, r_scores[3] = score_three_bridges("Manhattan", "Williamsburg", "Queensboro")

print("Brooklyn, Manhattan, Williamsburg bridges value:", r_scores[0])
print("Brooklyn, Manhattan, Queensboro bridges value:", r_scores[1])
print("Brooklyn, Williamsburg, Queensboro bridges value:", r_scores[2])
print("Manhattan, Williamsburg, Queensboro bridges value:", r_scores[3])

print(f"Total = {brook_man_will_Model.coef_[0][0]}*(Brooklyn) + {brook_man_will_Model.coef_[0][1]}*(Manhattan) + {brook_man_will_Model.coef_[0][2]}*(Williamsburg) + {brook_man_will_Model.intercept_[0]}")