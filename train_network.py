from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow

import tensorflow as tf
import pandas as pd
from sklearn import preprocessing

data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
data = data.drop(["customerID"], axis=1)
#Maybe case as boolean?
data["gender"].replace(inplace = True, to_replace = "Male", value=0)
data["gender"].replace(inplace = True, to_replace = "Female", value=0)
data["Partner"].replace(inplace = True, to_replace = "Yes", value=1)
data["Partner"].replace(inplace = True, to_replace = "No", value=0)
data["Dependents"].replace(inplace = True, to_replace = "Yes", value=1)
data["Dependents"].replace(inplace = True, to_replace = "No", value=0)
data["PhoneService"].replace(inplace = True, to_replace = "Yes", value=1)
data["PhoneService"].replace(inplace = True, to_replace = "No", value=0)
data["PaperlessBilling"].replace(inplace = True, to_replace = "Yes", value=1)
data["PaperlessBilling"].replace(inplace = True, to_replace = "No", value=0)
#data["OnlineSecurity"].replace(inplace = True, to_replace = "Yes", value=1)
#data["OnlineSecurity"].replace(inplace = True, to_replace = "No", value=0)
#data["OnlineBackup"].replace(inplace = True, to_replace = "Yes", value=1)
#data["OnlineBackup"].replace(inplace = True, to_replace = "No", value=0)
#data["DeviceProtection"].replace(inplace = True, to_replace = "Yes", value=1)
#data["DeviceProtection"].replace(inplace = True, to_replace = "No", value=0)
#data["TechSupport"].replace(inplace = True, to_replace = "Yes", value=1)
#data["TechSupport"].replace(inplace = True, to_replace = "No", value=0)
#data["StreamingTV"].replace(inplace = True, to_replace = "Yes", value=1)
#data["StreamingTV"].replace(inplace = True, to_replace = "No", value=0)
#data["StreamingMovies"].replace(inplace = True, to_replace = "Yes", value=1)
#data["StreamingMovies"].replace(inplace = True, to_replace = "No", value=0)
data["Churn"].replace(inplace = True, to_replace = "Yes", value=1)
data["Churn"].replace(inplace = True, to_replace = "No", value=0)

data["TotalCharges"].replace(inplace = True, to_replace = " ", value = "")
data.drop(index = data[data["TotalCharges"] == ""].index, inplace = True)
data["TotalCharges"] = data["TotalCharges"].astype("float64")

v_min = data.TotalCharges.min()
v_max = data.TotalCharges.max()
print("Total Charges " + str(v_max) + " " + str(v_min));
data.TotalCharges = (data.TotalCharges - v_min)/(v_max - v_min)

v_min = data.MonthlyCharges.min()
v_max = data.MonthlyCharges.max()
data.MonthlyCharges = (data.MonthlyCharges - v_min)/(v_max - v_min)
print("Monthly Charges" + str(v_max) + " " + str(v_min));


v_min = data.tenure.min()
v_max = data.tenure.max()
data.tenure = (data.tenure - v_min)/(v_max - v_min)
print("Tenure" + str(v_max) + " " + str(v_min));

data.InternetService = data.InternetService.astype("category")
data.MultipleLines = data.MultipleLines.astype("category")
data.Contract = data.Contract.astype("category")

#data["OnlineSecurity"] = data["OnlineSecurity"].astype("category")
#data["OnlineBackup"] = data["OnlineBackup"].astype("category")
#data["DeviceProtection"] = data["DeviceProtection"].astype("category")
#data["TechSupport"] = data["TechSupport"].astype("category")
#data["StreamingTV"] = data["StreamingTV"].astype("category")
#data["StreamingMovies"] = data["StreamingMovies"].astype("category")


data = pd.concat([data, pd.get_dummies(data["InternetService"], prefix="InternetService")], sort=True, axis=1)
data = pd.concat([data, pd.get_dummies(data["OnlineSecurity"], prefix="OnlineSecurity")], sort=True, axis=1)
data = pd.concat([data, pd.get_dummies(data["OnlineBackup"], prefix="OnlineBackup")], sort=True, axis=1)
data = pd.concat([data, pd.get_dummies(data["DeviceProtection"], prefix="DeviceProtection")], sort=True, axis=1)
data = pd.concat([data, pd.get_dummies(data["TechSupport"], prefix="TechSupport")], sort=True, axis=1)
data = pd.concat([data, pd.get_dummies(data["StreamingTV"], prefix="StreamingTV")], sort=True, axis=1)
data = pd.concat([data, pd.get_dummies(data["StreamingMovies"], prefix="StreamingMovies")], sort=True, axis=1)
data = pd.concat([data, pd.get_dummies(data["Contract"], prefix="Contract")], sort=True, axis=1)
data = pd.concat([data, pd.get_dummies(data["PaymentMethod"], prefix="PaymentMethod")], sort=True, axis=1)
data = pd.concat([data, pd.get_dummies(data["MultipleLines"], prefix="MultipleLines")], sort=True, axis=1)
data = data.drop(["InternetService", "OnlineSecurity", "OnlineBackup", "Contract"], axis=1)
data = data.drop(["DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"], axis=1)
data = data.drop(["PaymentMethod", "MultipleLines"], axis=1)


x_train = data[0:6000]
y_train = x_train["Churn"]
x_train = x_train.drop(["Churn"], axis=1)

x_test = data[6001:-1]
y_test = x_test["Churn"]
x_test = x_test.drop(["Churn"], axis=1)

model = tf.keras.Sequential([
	tf.keras.layers.Dense(len(x_train.columns), activation="relu", input_shape =(len(x_train.columns),)),
	tf.keras.layers.Dense(int(len(x_train.columns) / 2), activation="relu"),
	tf.keras.layers.Dropout(0.5),
	tf.keras.layers.Dense(int(len(x_train.columns) / 4), activation="relu"),
	tf.keras.layers.Dropout(0.5),
	tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=2000)
model.evaluate(x_test, y_test, verbose=2)
model.save("weights.h5")


