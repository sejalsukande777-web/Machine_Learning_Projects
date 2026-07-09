import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error,r2_score

#-------------------------------------------------
# Step 1 : load the dataset 
#-------------------------------------------------

df = pd.read_csv("california_housing.csv")
print("shape of dataset :",df.shape)
print("first 5 records:",df.head())

#-------------------------------------------------
# Step 2 : separate features and labels 
#-------------------------------------------------

X = df.drop("target",axis=1)
Y = df["target"]

#-------------------------------------------------
# Step 3 : split dataset for training and testing  
#-------------------------------------------------

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

#-------------------------------------------------
# step 4 : Create base model 
#-------------------------------------------------

base_model = DecisionTreeRegressor(random_state=42)

#-------------------------------------------------
# step 5 : Create bagging model 
#-------------------------------------------------

bagging_model = BaggingRegressor(
    estimator=base_model,
    n_estimators=10,
    random_state=42
)
#-------------------------------------------------
# step 6 : Train bagging model
#-------------------------------------------------

bagging_model.fit(X_train,Y_train)

#-------------------------------------------------
# step 7 : test bagging model 
#-------------------------------------------------

Y_pred = bagging_model.predict(X_test)

#-------------------------------------------------
# step 8 : evaluate bagging model 
#-------------------------------------------------

print("Mean_squared_error : ",mean_squared_error(Y_test,Y_pred))
print("r square : ",r2_score(Y_test,Y_pred))


