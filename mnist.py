import tkinter as tk
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow_datasets as tfds
import plotly.graph_objects as go

import time

# Function to create a progress bar
def update_progress_bar(iteration, total, bar_length=50):
    progress = (iteration / total)
    arrow = '#' * int(round(progress * bar_length))
    spaces = ' ' * (bar_length - len(arrow))
    print(f'\rProgress: [{arrow + spaces}] {progress * 100:.2f}% ', end='', flush=True)


# Here we use a sequential model.
# Feel free to play with the model to your liking
def define_model():
	model = Sequential()
	model.add(Dense(128, input_shape=(784, ), activation='relu', name='dense_1'))
	model.add(Dense(64, activation='relu', name='dense_2'))
	model.add(Dense(32, activation='relu', name='dense_3'))
	model.add(Dense(10, activation='linear', name='dense_output'))
	model.compile(optimizer='Adam', loss='mse', metrics=['mae'])
	model.summary()
	return model

data_type = 'float32'

#my_data = genfromtxt('data/MNIST_CSV/mnist_train.csv', delimiter=',', dtype=float)
#my_data = np.loadtxt('my_data_loaded.txt', dtype=data_type)


def load_data(split = 'train'):
	ds = tfds.load(name="mnist", split=split)
	ds_generator = tfds.as_numpy(ds)  # Convert `tf.data.Dataset` to Python generator

	#put everything in simple numpy arrays
	x_train = np.empty((len(ds_generator),28*28), dtype=data_type)
	y_train = np.zeros((len(ds_generator),10),dtype=data_type)

	print("preparing data")
 
	for i,data in enumerate(ds_generator):
		x_train[i,:] = data['image'].flatten()
		#As we have 10 neurons in the output layer we set the activated digit to one
		y_train[i,data['label']] = 1 #e.g. label = 0 --> 0th neuron will be set to one
  
		if i%500 == 0:
			update_progress_bar(i, len(ds_generator))

	#normalize data
	x_train = x_train/255.0
	return (x_train, y_train)

# Curtesy to 
# https://towardsdatascience.com/keras-101-a-simple-and-interpretable-neural-network-model-for-house-pricing-regression-31b1a77f05ae
def show_training_loss(history):
	fig = go.Figure()
	fig.add_trace(go.Scattergl(y=history.history['loss'],
						name='Train'))
	fig.add_trace(go.Scattergl(y=history.history['val_loss'],
						name='Valid'))
	fig.update_layout(height=500, width=700,
					xaxis_title='Epoch',
					yaxis_title='Loss')
	fig.show()

model = define_model()
x_train, y_train = load_data('train')

# train and save model
history = model.fit(x_train, y_train, epochs=6, validation_split=0.1)

show_training_loss(history)
model.save("models/mnist")

####################################################################
#evaluate
x_eval, y_eval = load_data('test')
_, acc = model.evaluate(x_eval, y_eval, verbose=0)
print(f"Model accuracy on test data set is {acc}")

####################################################################
from draw import Drawer
root = tk.Tk()
app = Drawer(root, model)
app.run()