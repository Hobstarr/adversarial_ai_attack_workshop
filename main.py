# TODO: edit load_data data_dir in utils.py
# TODO: Randomise dataset - simple solution: new create_dataset
# TODO: Add prediction of each model (to show what models predict as)

import random                         #
import matplotlib.pyplot as plt       # useful python tools
import numpy as np                    #
import time                           #

import tensorflow as tf  # Tensorflow is googles model building library, supports distributed computing
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape

# cleverhans is a model attacking and defending tool, full info on github
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

from easydict import EasyDict               # from utils.py file
from utils import Neural_Net, create_dataset, preprocess_single_for_pert, plot_adv, \
                    plot_prob_dist

#############################################
######### Part 1: ML Model FUNdamentals #####
#############################################

######### MNIST Fashion Dataset #############
### Building a dataset to work from
### If you'd rather use the fingerprint dataset
### change fashion_mnist to mnist

# use inbuilt mnist or fashion_mnist from keras (comment mnist, uncomment fashion_mnist)
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
# (x_train, y_train), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()
train_dataset = create_dataset(x_train, y_train)
val_dataset = create_dataset(x_val, y_val)

# What does this dataset look like? 
# Plot the first x entries, change the values within subplots to edit:
fig, ax = plt.subplots(5, 5) # Build 5 x 5 grid
fig.suptitle(f'MNIST Dataset: First {len(ax.flatten())} Entries', fontsize = '12')
for i, ax in enumerate(ax.flatten()): 
    ax.imshow(x_train[i], cmap ='gray_r')
    ax.axis('off')
fig.show()

# If using fashion_mnist use this class list:
class_list = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

######### Create Target Model #########
# Create secret model using random subset of training data,
# random number of layers [1-10], random nodes in layers [32-256]
# TODO: Randomise dataset - simple solution: new create_dataset
random_mask = np.random.choice(60000, replace=False, size=10000)
secret_train_dataset = create_dataset(x_train[random_mask], y_train[random_mask])

secret_layers = random.randint(1,10)
secret_layers_list = [random.randint(32,256) for x in range(secret_layers)]
secret_layers_list.sort(reverse = True)

secret_model = Neural_Net(layers = secret_layers, layers_list = secret_layers_list)
secret_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
              loss = tf.losses.CategoricalCrossentropy(),
              metrics = ['accuracy'])

history = secret_model.fit(
    secret_train_dataset.repeat(),
    epochs=10,
    steps_per_epoch=100,
    validation_data = val_dataset.repeat(),
    validation_steps = 2 
)

secret_train_loss, secret_train_acc = secret_model.evaluate(train_dataset)
secret_test_loss, secret_test_acc = secret_model.evaluate(val_dataset)
print(f'Secret Model Performance \n\
Train Accuracy: {secret_train_acc:.4f} \n\
Test Accuracy : {secret_test_acc:.4f}')
      
######### Create Surrogate Model #########
# To attack a model, we often have to create our own
# model that accomplishes the same task. Due to some
# intriguing properties of neural networks, and other
# types of models it can be found that attacks are
# 'transferable'.
train_random_mask = np.random.choice(60000, replace=False, size=10000)
sub_train_dataset = create_dataset(x_train[train_random_mask], y_train[train_random_mask])

# Create a very simple model
model = Neural_Net(layers = 2, layers_list = [32,16])
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
              loss = tf.losses.CategoricalCrossentropy(),
              metrics = ['accuracy'])

# Using the created datasets
history = model.fit(
    sub_train_dataset.repeat(),
    epochs=10,
    steps_per_epoch=100,
    validation_data = val_dataset.repeat(),
    validation_steps = 2 
)

train_loss, train_acc = model.evaluate(train_dataset)
test_loss, test_acc = model.evaluate(val_dataset)
print(f'Secret Model Performance \n\
Train Accuracy: {train_acc:.4f} \n\
Test Accuracy : {test_acc:.4f}')

# How do these models differ, what is ambiguous for our models?
model_preds = model(x_train)
secret_preds = secret_model(x_train)
train_disagree_dict = {}
for i in range(2000):
    if not (model_preds.numpy()[i] == secret_preds.numpy()[i]).all():
        print(str(i))
        dict_key = str(i)
        train_disagree_dict[dict_key] = x_train[i]
        # print(f'The {i}th entry is predicted differently')

# TODO: Add prediction of each model (to show what models predict as)
# What does this dataset look like? 
# Plot the first n * n entries:
fig, ax = plt.subplots(5, 5) # Build 5 x 5 grid
fig.suptitle('MNIST Dataset: Ambiguous entries', fontsize = '12')
for i, ax in enumerate(ax.flatten()):
    dict_entry = list(train_disagree_dict.keys())[i]
    ax.imshow(train_disagree_dict[dict_entry], cmap ='gray_r')
    ax.axis('off')
fig.show()

#############################################
######### Part 2: Crafting ML Model Attacks #
#############################################

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

#################################
#########  FGSM Attack  #########
loss_object = tf.keras.losses.CategoricalCrossentropy()
def create_adversarial_pattern(input_image, input_label):
    with tf.GradientTape(persistent = True) as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        print(prediction)
        loss = loss_object(input_label, prediction)

    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    return(signed_grad)

# Pick a target, our aim is to misclassify this image
target_image = 10
input_image, input_label = preprocess_single_for_pert(x_val[0], y_val[0])
plt.imshow(input_image[0]) # the subscript gives us the values from a tensor
plt.xlabel(f'Our model predicts this image: {np.argmax(model(input_image))}')
plt.show()

# Build perturbation mask and show it
perturbations = create_adversarial_pattern(input_image, input_label)
plt.imshow(perturbations[0])
plt.show()

# Show image
# Since this attack is 'not-targetted' we are trying to 
# increase the loss with respect to ther input image
eps = 0.07
adv_x = input_image + (eps * perturbations)
plt.imshow(adv_x[0])
plt.xlabel(f'Our model predicts this image: {np.argmax(model(adv_x))}')
plt.show()

eps = 0.10        # value of change to make to input
target_image = 100           # index of image in validation set
input_image, input_label = preprocess_single_for_pert(x_val[target_image], y_val[target_image])
perturbations = create_adversarial_pattern(input_image, input_label)
plot_adv(input_image, eps, perturbations, model, secret_model)

#############################################
######### Part 3: What can we Learn? ########
#############################################

#################################
#### FGSM Attack Heuristics #####
def plot_prob_dist(image_index, model):
    input_image, input_label = preprocess_single_for_pert(x_val[target_image], y_val[target_image])
    perturbations = create_adversarial_pattern(input_image, input_label)

    start = time.time()
    model_prob_dict = {}
    for i in range(10):
        model_prob_dict[str(i)] = []
    for eps in range(0, 1000):
        adv_x = input_image + (eps/1000 * perturbations)
        predicted_probs = model(adv_x).numpy()[0]
        for i in range(10):
            model_prob_dict[str(i)].append(predicted_probs[i])
    end = time.time()
    print(end - start)
        
    fig = plt.figure(figsize = (8,8))
    for i in range(10):
        dict_entry = str(i)
        plt.plot([x / 1000 for x in list(range(1000))], model_prob_dict[dict_entry], label = dict_entry)
    fig.legend()
    plt.ylabel('probability')
    plt.xlabel('eps')
    plt.title('class probability vs perturbation value')
    fig.show()

target_image = 215
plot_prob_dist(target_image, model)
plot_prob_dist(target_image, secret_model)

eps = 0.5                     
target_image = 215          # index of image in validation set
input_image, input_label = preprocess_single_for_pert(x_val[target_image], y_val[target_image])
perturbations = create_adversarial_pattern(input_image, input_label)
plot_adv(input_image, eps, perturbations, model, secret_model)

