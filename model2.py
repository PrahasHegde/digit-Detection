#imports
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

# Workaround for the DistributedDatasetInterface error
from tensorflow.python.keras.engine import data_adapter

def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

data_adapter._is_distributed_dataset = _is_distributed_dataset

# Check tf.keras version
print(tf.keras.__version__)

# Loading MNIST dataset
mnist = tf.keras.datasets.mnist

# Splitting into train and test
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Data Exploration
print(X_train.shape)
print(X_test.shape)

# X_train is 60000 rows of 28x28 values; we reshape it to 60000 x 784. 
RESHAPED = 784 # 28x28 = 784 neurons
X_train = X_train.reshape(60000, RESHAPED) 
X_test = X_test.reshape(10000, RESHAPED) 

# Data is converted into float32 to use 32-bit precision when training a neural network 
X_train = X_train.astype('float32')
X_test = X_test.astype('float32') 

# Normalizing the input to be within the range [0,1]
X_train /= 255
X_test /= 255

print(X_train.shape[0], 'train samples') 
print(X_test.shape[0], 'test samples') 

# One-hot representation of the labels.
Y_train = tf.keras.utils.to_categorical(Y_train, 10) 
Y_test = tf.keras.utils.to_categorical(Y_test, 10)

# Load model2
# Most common type of model is a stack of layers
model_2 = Sequential()
N_hidden = 64

# Adds a densely-connected layer with 64 units to the model
model_2.add(Dense(N_hidden, name='dense_layer', input_shape=(784,), activation='relu'))

# Adding another dense layer
model_2.add(Dense(N_hidden, name='dense_layer_2', activation='relu'))

# Add an output layer with 10 output units (10 different classes)
model_2.add(Dense(10, name='dense_layer_3', activation='softmax'))

# Compiling the model
model_2.compile(optimizer='SGD', 
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Training the model
training = model_2.fit(X_train, Y_train, batch_size=64, epochs=100, validation_split=0.2)

# List all data in training
print(training.history.keys())

# Summarize training for accuracy
plt.plot(training.history['accuracy'])
plt.plot(training.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Summarize training for loss
plt.plot(training.history['loss'])
plt.plot(training.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Evaluate the model
test_loss, test_acc = model_2.evaluate(X_test, Y_test)
print('Test accuracy:', test_acc)
