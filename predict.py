from pandas import read_csv
import numpy as np
import tensorflow as tf

testdata = []
testdata.append(read_csv("datasets/dataset_test_0.csv", header=1))
testdata.append(read_csv("datasets/dataset_test_1.csv", header=1))
del testdata[0]["0"]
del testdata[1]["1"]

model = tf.keras.models.load_model('trained')

print()
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
