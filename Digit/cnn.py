import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import numpy as np

#loading data
data=pd.read_csv("train.csv")
print(data.shape)
print(data.columns)
#print(data.describe().to_string())

#Normalize the data
label=data["label"]
data=data.drop(labels="label",axis=1)
df=data.to_numpy()
min_max_scaler = preprocessing.MinMaxScaler()
df_scaled = min_max_scaler.fit_transform(df)
data = pd.DataFrame(df_scaled)
#data_augmentation


#visualize the image
plt.imshow(df[2].reshape(28,28))
plt.show()
#train and dev
x_train,x_test,y_train,y_test=train_test_split(data,label,test_size=0.2)
#Knn Classifier
#clf=KNeighborsClassifier(n_neighbors=10)
#clf.fit(x_train,y_train)
#y_pred=clf.predict(x_test)
#print(accuracy_score(y_test,y_pred))
#CNN 
# 1)convertir en array
x_train=x_train.to_numpy()
x_test=x_test.to_numpy()
# 2)création du model
model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu',padding='same',input_shape=(28, 28, 1)))
model.add(layers.AveragePooling2D())
model.add(layers.Conv2D(32, (5, 5), activation='relu'))
model.add(layers.AveragePooling2D())
model.add(layers.Conv2D(120, (5, 5), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#reshape pour definir height width channels
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)
#training the model
history = model.fit(x_train, y_train, epochs=10,
                    validation_data=(x_test, y_test))
#ploting the test and train accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
test_loss, test_acc = model.evaluate(x_test,y_test, verbose=2)
print(test_acc)

######testing the model########

test=pd.read_csv("test.csv")
target=pd.read_csv("sample_submission.csv")
lb=target["Label"]
lb=lb.to_numpy()
test=test.to_numpy()
test=test.reshape(test.shape[0],28,28,1)
# history = model.fit(x_train, y_train, epochs=10,
#                     validation_data=(test,lb))
test_loss1, test_acc1 = model.evaluate(test,lb, verbose=2)
# print(test_acc)
result = model.predict(test)
x=np.argmax(result,axis=1)
print(x)
sub=pd.DataFrame({"ImageId":target["ImageId"],"Label":x})
sub.to_csv(path_or_buf="submission.csv",index=False)