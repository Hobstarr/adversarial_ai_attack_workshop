import random                         #
import matplotlib.pyplot as plt       # useful python tools
import numpy as np                    #

import tensorflow as tf                     # Tensorflow is gogles model building
import tensorflow_datasets as tfds          # library, supports distributed computing
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, Reshape

# cleverhans is a model attacking and defending tool, full info on github
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

from absl import app, flags                 # specific tools and custom utils 
from easydict import EasyDict               # from utils.py file
from utils import ld_mnist, ld_mnist_onehot, create_dataset, Neural_Net, call_model

FLAGS = flags.FLAGS # setting up command line parameters for simulating command line.

#############################################
######### MNIST Fingerprint Dataset #########
#############################################

# Try using mnist from keras
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
train_dataset = create_dataset(x_train, y_train)
val_dataset = create_dataset(x_val, y_val)

# What does this dataset look like? 
# Plot the first 100 entries:
fig, ax = plt.subplots(10,10) # Build 10 x 10 grid
fig.suptitle('MNIST Dataset: First 100 Entries', fontsize = '12')
for i, ax in enumerate(ax.flatten()): 
    ax.imshow(x_train[i], cmap ='gray_r')
    ax.axis('off')
fig.show()

#######################################
######### Create Target Model #########
#######################################

# Create secret model using random subset of training data,
# random number of layers [1-10], random nodes in layers [32-256]
# TODO: Randomise dataset - simple solution: new create_dataset
secret_layers = random.randint(1,10)
secret_layers_list = [random.randint(32,256) for x in range(secret_layers)]
secret_layers_list.sort(reverse = True)

secret_model = Neural_Net(layers = secret_layers, layers_list = secret_layers_list)
secret_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
              loss = tf.losses.CategoricalCrossentropy(from_logits = True),
              metrics = ['accuracy'])

history = secret_model.fit(
    train_dataset.repeat(),
    epochs=10,
    steps_per_epoch=500,
    validation_data = val_dataset.repeat(),
    validation_steps = 2 
)

secret_train_loss, secret_train_acc = secret_model.evaluate(train_dataset)
secret_test_loss, secret_test_acc = secret_model.evaluate(val_dataset)
print(f'Secret Model Performance \n\
Train Accuracy: {secret_train_acc:.4f} \n\
Test Accuracy : {secret_test_acc:.4f}')
      
##########################################
######### Create Surrogate Model #########
##########################################

# Create surrogate model - same task (identifying fingerprints)
# Create a very simple model
model = Neural_Net(layers = 3, layers_list = [128,64,32])
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
              loss = tf.losses.CategoricalCrossentropy(from_logits = True),
              metrics = ['accuracy'])

# Using the created datasets              #TODO: probably only need one of these
history = model.fit(
    train_dataset.repeat(),
    epochs=10,
    steps_per_epoch=500,
    validation_data = val_dataset.repeat(),
    validation_steps = 2 
)

train_loss, train_acc = model.evaluate(train_dataset)
test_loss, test_acc = model.evaluate(val_dataset)
print(f'Secret Model Performance \n\
Train Accuracy: {train_acc:.4f} \n\
Test Accuracy : {test_acc:.4f}')

# TODO: Move to utils
def call_model(model, example):
    one_hot = model(example.reshape(1,28,28))
    return np.argmax(one_hot)

train_disagree_dict = {}
for i in range(60000):
    if call_model(model, x_train[i]) != call_model(secret_model, x_train[i]):
        dict_key = str(i)
        train_disagree_dict[dict_key] = x_train[i]
        # print(f'The {i}th entry is predicted differently')

# TODO: Add prediction of each model (to show what models predict as)
# What does this dataset look like? 
# Plot the first 100 entries:
fig, ax = plt.subplots(5,5) # Build 5 x 5 grid
fig.suptitle('MNIST Dataset: Ambiguous entries', fontsize = '12')
for i, ax in enumerate(ax.flatten()):
    dict_entry = list(train_disagree_dict.keys())[i]
    ax.imshow(train_disagree_dict[dict_entry], cmap ='gray_r')
    ax.axis('off')
fig.show()

# Code for predicting each model on an example [for use in above TODO]
call_model(model, x_train[0])
call_model(secret_model, x_train[0])

#################################
######### Gradient Tape #########
#################################

# Quick intro to gradient tape
x = tf.constant(1.0) # Need persistent to call it twice
with tf.GradientTape(persistent = True) as tape:
  tape.watch(x)
  y = 5 * x * x * x
  z = (y * x) /4
dy_dx = tape.gradient(y, x)
dz_dx = tape.gradient(z, x)
print(f'{dy_dx}, {dz_dx}')

# Build an FGSM attack
input_label = y_train[0]
input_label = tf.cast(input_label, tf.int32)
input_label = tf.one_hot(input_label, 10)
input_label = tf.reshape(input_label, (1,10))

input_image = x_train[0]
input_image = input_image.reshape(1,28,28)
input_image = tf.cast(input_image, tf.float32)
input_image = tf.convert_to_tensor(input_image)

softmax = tf.keras.layers.Softmax()
softmax(model(input_image))

loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits = True)
tf.one_hot
def create_adversarial_pattern(input_image, input_label):
    input_label = input_label.reshape(-1)
    input_label = tf.cast(input_label, tf.int32)
    one_hot_label = tf.one_hot(input_label, 10)

    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = loss_object(input_label, prediction)

    gradient = tape.gradient(gradient)
    return(gradient)

create_adversarial_pattern(x_train[0], y_train[0])


input_label = y_train[0].reshape(-1)
input_label = tf.cast(input_label, tf.int32)
one_hot_label = tf.one_hot(input_label, 10)


input_image = x_train[0]
input_image = input_image.reshape(1,28,28)
input_image = tf.cast(input_image, tf.float32)
input_image = tf.convert_to_tensor(input_image)
model(input_image)







########## UNUSED CODE, TODO ##########
# Quite a lot of heuristics from model.metrics to history.History [outputs]
model.metrics

# Cleverhans data type for training / running cleverhans examples
# Try using mnist from tfds ['slightly different']
data = ld_mnist_onehot()

# Using load_mnist datasets easydict
history = model.fit(
    data['train'].repeat(),
    epochs=10,
    steps_per_epoch=500,
    validation_data = data['test'],
    validation_steps = 2 
)
#######################################


# Build model - test/train
# Build model - attacks
# Build model - adversarial training
# Build model - attacks

# Running the standalone code in the terminal : (need to define main() function)
# Here we define flags and then 'run' in the command line with set params
if __name__ == "__main__":
    flags.DEFINE_integer("nb_epochs", 5, "Number of epochs.")
    flags.DEFINE_float("eps", 0.3, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_bool(
        "adv_train", False, "Use adversarial training (on PGD adversarial examples)."
    )
    app.run(main)