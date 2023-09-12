# -*- coding: utf-8 -*-


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

mnist = tf.keras.datasets.mnist # Object of the MNIST dataset
num_classes = 10
(x_train, y_train),(x_test, y_test) = mnist.load_data() # Load data
# print(x_train[300][17])
y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes)
x_train_oh = x_train.reshape(x_train.shape[0], 784)
x_test_oh = x_test.reshape(x_test.shape[0],784)
x_train = x_train/255
x_test = x_test/255
# x_train[300].reshape(784)

counter = 0
index = 100
for i in range(index):
  if y_test[i]==1:
    counter+=1
print(counter)

j=20
best_acc = 0
best_neurons = 0
neurons = np.empty(10)
accs = np.empty(10)
train_accs = np.empty(10)

for i in range(10):
  model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(j,activation="relu"),#relu drops accuracy
            tf.keras.layers.Dense(10,activation="relu")
        ])
  model.compile(optimizer="Adam", loss="mse", metrics=["accuracy"])
  model.fit(x=x_train, y=y_train_oh) # Start training process
  metr = model.evaluate(x_test,y_test_oh)
  train_metr = model.evaluate(x_train,y_train_oh)
  if metr[1]>best_acc:
    best_neurons = j
    best_acc = metr[1]
    model.save("neurons_"+str(j))
  accs[i] = metr[1]
  neurons[i] = j
  train_accs[i] = train_metr[1]
  j=j+10
print("THE Accuracy : " + str(best_acc))
print("THE neurons  : " + str(best_neurons))
print(accs)
print(best_neurons)

plt.xlabel("Number of Neurons in Hidden Layer")
plt.ylabel("Accuracy in %")
plt.title("Accuracy against number of hidden layer neurons")
plt.plot(neurons,accs*100, label="Testing accuracy")
plt.plot(neurons,train_accs*100, label= "Training accuracy")
plt.legend()
plt.show()

optimizers = ["nadam","SGD","RMSprop","adam","adadelta","Adagrad","adamax"]
best_acc_opt=0
optimizer = ""
train_accs_opt = np.empty(7)
accs_opt = np.empty(7)
for i in range(7):
  model = tf.keras.models.load_model("neurons_80")
  model.compile(optimizer=optimizers[i], loss="mse", metrics=["accuracy"])
  model.fit(x=x_train, y=y_train_oh) # Start training process
  metr = model.evaluate(x_test,y_test_oh)
  train_metr = model.evaluate(x_train,y_train_oh)

  if metr[1]>best_acc_opt:
    optimizer = optimizers[i]
    best_acc_opt = metr[1]
    model.save("opt_"+optimizer)
  accs_opt[i] =  metr[1]
  train_accs_opt[i] = train_metr[1]

print("The optimizer : "+optimizer)
print("The accuracy : "+str(best_acc_opt))

print("BEST OPTIMIZER  :  " +optimizer)
print(best_acc_opt)

plt.xlabel("Optimizer")
plt.ylabel("Accuracy in %")
plt.title("Accuracy against different optimizers")
plt.scatter(optimizers,accs_opt*100, label="Testing accuracy")
plt.scatter(optimizers,train_accs_opt*100, label= "Training accuracy")
plt.legend()
plt.show()

batch_size = 10
best_acc =0
train_accs = np.empty(15)
accs = np.empty(15)
best_batch =0

for i in range(15):
  model = tf.keras.models.load_model("opt_nadam")
  model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
  model.fit(x=x_train, y=y_train_oh, batch_size=batch_size) # Start training process
  metr = model.evaluate(x_test,y_test_oh)
  train_metr = model.evaluate(x_train,y_train_oh)

  if metr[1]>best_acc:
    best_batch = batch_size
    best_acc = metr[1]
    model.save("batch_"+str(best_batch)+".h5")

  batch_size+=10
  accs[i] =  metr[1]
  train_accs[i] = train_metr[1]

print(best_batch)
print(best_acc)

batchs = np.array([10,20,30,40,50,60,70,80,90,100,110,120,130,140,150])
plt.xlabel("Batch Size")
plt.ylabel("Accuracy in %")
plt.title("Accuracy against different batch sizes")
plt.plot(batchs,accs*100, label="Testing accuracy")
plt.plot(batchs,train_accs*100, label= "Training accuracy")
plt.legend()
plt.show()

model = tf.keras.models.load_model("opt_nadam")
model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
model.fit(x=x_train, y=y_train_oh, batch_size=130) # Start training process
model.evaluate(x_test,y_test_oh)
print("Best ACC  : "+str(metr[1]))

batches = np.array([10,20,30,40,50,60,70,80,90,100,110,120,130,140,150])
plt.xlabel("Batch size")
plt.ylabel("Accuracy in %")
plt.title("Accuracy against different batch sizes")
plt.scatter(batches,accs, label="Testing accuracy")
plt.scatter(batches,train_accs, label= "Training accuracy")
plt.legend()
plt.show()

print(best_batch)
print(best_acc)

model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100,activation="relu"),#relu drops accuracy
            tf.keras.layers.Dense(10,activation="relu")
        ])
model.compile(optimizer="Adam", loss="mse", metrics=["accuracy"])
model.fit(x=x_train, y=y_train_oh) # Start training process
metr = model.evaluate(x_test,y_test_oh)
print(metr[1])

model1 = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100,activation="relu"),#relu drops accuracy
            tf.keras.layers.Dense(10,activation="relu")
        ])
model1.compile(optimizer="Adam", loss="mse", metrics=["accuracy"])
model1.fit(x=x_train, y=y_train_oh) # Start training process
metr = model1.evaluate(x_test,y_test_oh)
print(metr[1])

model3 = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100,activation="relu"),#relu drops accuracy
            tf.keras.layers.Dense(10,activation="relu")
        ])
model3.compile(optimizer="Adam", loss="mse", metrics=["accuracy"])
model3.fit(x=x_train, y=y_train_oh) # Start training process
metr = model3.evaluate(x_test,y_test_oh)
print(metr[1])

model4 = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100,activation="relu"),#relu drops accuracy
            tf.keras.layers.Dense(10,activation="relu")
        ])
model4.compile(optimizer="Adam", loss="mse", metrics=["accuracy"])
model4.fit(x=x_train, y=y_train_oh) # Start training process
metr = model4.evaluate(x_test,y_test_oh)
print(metr[1])

metr = model4.evaluate(x_test,y_test_oh)
print(metr[1])

metr = model4.evaluate(x_test,y_test_oh)
print(metr[1])

evaluations_epochs_test = np.array([])
evaluations_epochs_train = np.array([])
epoch_tr_loss = np.array([])
epoch_ts_loss= np.array([])
best_acc=0
best_loss=1
for i in range(5,101,5):
  epochs_model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(90,activation="relu"),#relu drops accuracy
            tf.keras.layers.Dense(10,activation="relu")
        ])
  epochs_model.compile(optimizer="nadam", loss="mse", metrics=["accuracy"])
  epochs_model.fit(x=x_train, y=y_train_oh,epochs=i, batch_size=150)
  evaluation_test = epochs_model.evaluate(x_test,y_test_oh)
  evaluation_train = epochs_model.evaluate(x_train,y_train_oh)
  evaluations_epochs_test = np.append(evaluations_epochs_test,evaluation_test[1])
  evaluations_epochs_train = np.append(evaluations_epochs_train,evaluation_train[1])
  epochs_tr_loss = np.append(evaluations_epochs_test,evaluation_test[0])
  epochs_ts_loss = np.append(evaluations_epochs_train,evaluation_train[0])
  if evaluation_test[1]>best_acc:
    epochs_model.save("epochs_"+str(i)+".h5")
    best_acc=evaluation_test[1]
    best_loss = evaluation_test[0]

epochs = np.array(range(5,101,5))
print(best_acc)

f, ax = plt.subplots(1)
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy in %")

plt.title("Accuracy against different Epochs")

plt.plot(epochs,evaluations_epochs_test*100, label="Testing accuracy")
plt.plot(epochs,evaluations_epochs_train*100, label= "Training accuracy")
plt.legend()
plt.show()

epochs_model = tf.keras.models.load_model("batch_130.h5")
epochs_model.compile(optimizer="Adam", loss="mse", metrics=["accuracy"])
epochs_model.fit(x=x_train, y=y_train_oh, epochs=1)
testsss = epochs_model.evaluate(x_test, y_test_oh)
print(testsss[1])

print(best_acc)

losses = ["mse", ""]

model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(90,activation="relu"),#relu drops accuracy

            tf.keras.layers.Dense(10,activation="relu")
        ])
model.compile(optimizer="nadam",loss="mse",metrics=["accuracy"])
model.fit(x_train,y_train_oh,epochs=20)
model.evaluate(x_test,y_test_oh)

j=110
best_acc = 0
best_neurons = 0
neurons = np.empty(10)
accs = np.empty(10)
train_accs = np.empty(10)

for i in range(10):
  model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(j,activation="relu"),#relu drops accuracy
            tf.keras.layers.Dense(10,activation="relu")
        ])
  model.compile(optimizer="nadam", loss="mse", metrics=["accuracy"])
  model.fit(x=x_train, y=y_train_oh) # Start training process
  metr = model.evaluate(x_test,y_test_oh)
  train_metr = model.evaluate(x_train,y_train_oh)
  if metr[1]>best_acc:
    best_neurons = j
    best_acc = metr[1]
    model.save("neurons_"+str(j))
  accs[i] = metr[1]
  neurons[i] = j
  train_accs[i] = train_metr[1]
  j=j+10
print("THE Accuracy : " + str(best_acc))
print("THE neurons  : " + str(best_neurons))
print(accs)
print(best_neurons)

plt.xlabel("Number of Neurons in Hidden Layer")
plt.ylabel("Accuracy in %")
plt.title("Accuracy against number of hidden layer neurons")
plt.plot(neurons,accs*100, label="Testing accuracy")
plt.plot(neurons,train_accs*100, label= "Training accuracy")
plt.legend()
plt.show()

model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(90,activation="relu"),#relu drops accuracy
            tf.keras.layers.Dense(10,activation="relu")
        ])
model.compile(optimizer="nadam", loss="CategoricalCrossentropy", metrics=["accuracy"])
model.fit(x=x_train, y=y_train_oh,epochs=20, batch_size = 10000) # Start training process
model.save("final.h5")

model.evaluate(x_test,y_test_oh)

