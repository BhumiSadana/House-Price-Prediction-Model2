import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("House Price Prediction Dataset.csv")
print(df.head())

#preprocessing
print(df.shape)
print(df.info())

print(df.isnull().sum())


le=LabelEncoder()
df["Location"]=le.fit_transform(df["Location"])
df["Condition"]=le.fit_transform(df["Condition"])
df["Garage"]=le.fit_transform(df["Garage"])

X=df.drop("Price",axis=1)
y=df["Price"]

features=["Area","Bathrooms","Floors","Location","Condition","Bedrooms","Garage"]
standard_scaler=StandardScaler()
df_scaled=df.copy()
df_scaled[features]=standard_scaler.fit_transform(df[features])

X=df_scaled[features]
y=df["Price"]

#split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#model
model=model = RandomForestRegressor(n_estimators=50,max_depth=15, n_jobs=-1,random_state=42)
model.fit(X_train,y_train)
#pred
y_pred=model.predict(X_test)

#evaluate
print("MAE is:",mean_absolute_error(y_test,y_pred))
print("MSE is:",mean_squared_error(y_test,y_pred))
print("RMSE is:",np.sqrt(mean_squared_error(y_test,y_pred)))




#visualize

sns.scatterplot(x=y_test,y=y_pred,alpha=0.7)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(" Actual Vs Predicted House Price")
plt.tight_layout()
plt.show()


print("--Price Prediction--")
try:
  area=int(input("Enter area:"))
  bedroom=int(input("Enter bedrooms"))
  floor=int(input("Enter floors:"))
  bathroom=int(input("Enter bathrooms:"))
  location = int(input("Enter location code (same encoding used): "))
  condition = int(input("Enter condition code: "))
  garage = int(input("Enter garage (1 for Yes, 0 for No): "))


  user_input=pd.DataFrame([{
      "Area":area,
      "Bathrooms":bathroom,
      "Floors":floor,
      "Location": location,
      "Condition": condition,
      "Bedrooms": bedroom,
      "Garage": garage
  }])

  user_input=user_input[features]
  user_input_scaled = standard_scaler.transform(user_input)
  pred=model.predict(user_input_scaled)[0]
  print(f"Predicted House Price: {pred:.2f}")



except Exception as e:
  print("An error occured",e)

