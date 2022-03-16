import imp
from keras import Sequential
from keras.layers import Activation,Dropout,Dense,Flatten	
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils import np_utils
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data=pd.read_csv("train.csv")
print(data.shape)
print(data.columns)
label=data["label"]
data=data.drop(labels="label",axis=1)
x_train,x_test,y_train,y_test=train_test_split(data,label,test_size=0.2)
print(x_train)
