from pandas import read_csv
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

trainingsdata = read_csv("datasets/dataset_train.csv", header=1)
testdata = []
testdata.append(read_csv("datasets/dataset_test_0.csv", header=1))
testdata.append(read_csv("datasets/dataset_test_1.csv", header=1))

del testdata[0]["0"]
del testdata[1]["1"]

X, Y = trainingsdata.values[:, 1:], trainingsdata.values[:, 0]
X = X.astype('float32')
Y = LabelEncoder().fit_transform(Y)

n_features = X.shape[1]
opt = tf.keras.optimizers.SGD(momentum=0.0, nesterov=False)

model = Sequential()
model.add(Dense(10, input_shape =(n_features,)))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, Y, epochs=200, batch_size=32, verbose=0)

#model.save('trained') # SAVE THE MODEL

print("Results")
i = 0
for data in testdata:
    count = [0,0]

    for row in np.array(data):
        row = np.array([row])
        yhat = model.predict(row)
        if yhat[0][0] < 0.5:
            count[0] += 1
        else:
            count[1] += 1
    print(count[0], count[1])
    print("Accuracy: " + str(round((count[i]/sum(count)), 5)*100) + "%")
    i+=1
