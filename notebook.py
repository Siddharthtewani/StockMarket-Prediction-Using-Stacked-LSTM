#%%
import pandas as pd
import numpy as np
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense , Dropout
print("Done")
#%%
# df=pdr.get_data_tiingo("AAPL",api_key="api")
# # %%
# df.head()
# # %%
# df.describe()
# # %%
# df.isnull().sum()
# # %%
# df.to_csv("data.csv")
# # %%
df=pd.read_csv("data.csv")
# %%
df.head()
# %%
df.tail()
# %%
df1=df.reset_index()["close"]
# %%
print(df1)
print(df1.shape)
# %%
ax=plt.figure(figsize=(10,10))
plt.plot(df["date"],df["close"])
plt.ylabel("Date")
plt.xlabel("Close")
plt.show()
# %%
ax=plt.figure(figsize=(10,10))
plt.plot(df["date"],df["volume"])
plt.ylabel("Date")
plt.xlabel("volume")
plt.show()
# %%
ax=plt.figure(figsize=(10,10))
plt.plot(df["date"],df["high"])
plt.ylabel("Date")
plt.xlabel("high")
plt.show()
#%%
df["divCash"].value_counts()
#%%
df["splitFactor"].value_counts()
# %%
scalar=MinMaxScaler(feature_range=(0,1))
df1=scalar.fit_transform(np.array(df1).reshape(-1,1))
df1
# %%
df1.shape
# %%
train_size=int(len(df1)*0.7)
test_size=len(df1)-train_size
train_data,test_data=df1[0:train_size,:],df1[train_size:len(df1),:1]
# %%
len(test_data)
# %%
len(train_data)
# %%
test_data
# %%
def create_dataset(dataset,time_step=1):
    dataX,dataY=[],[]
    for i in range(len(dataset)-time_step-1):
        a=dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0]) 

    return np.array(dataX),np.array(dataY)
#%%
train_step=100
x_train,y_train=create_dataset(train_data,train_step)
x_test,y_test=create_dataset(test_data,train_step)

# %%
x_train.shape
# %%
x_train
# %%
x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)
# %%
def build_model():
    model=Sequential()
    model.add(Dense(50,activation="relu",input_shape=(100,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    return model
# %%
model=build_model()
# %%
model.summary()
# %%
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
# %%
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=15,batch_size=2)
# %%
train_pred=model.predict(x_train)
test_pred=model.predict(y_test)
# %%
train_pred=train_pred.reshape(train_pred.shape[0],train_pred.shape[1])
test_pred=test_pred.reshape(test_pred.shape[0],test_pred.shape[1])
#%%
train_pred=scalar.inverse_transform(train_pred)
test_pred=scalar.inverse_transform(test_pred)
# %%
import math  
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_pred))
# %%
    
# %%

# %%
