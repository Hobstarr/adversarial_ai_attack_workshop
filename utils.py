import numpy as np
import tensorflow as tf
from easydict import EasyDict
from tensorflow.keras import Model
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, Reshape
import matplotlib.pyplot as plt

@tf.autograph.experimental.do_not_convert
def preprocess(x, y):
  '''helper function for create_dataset'''
  x = tf.cast(x, tf.float32) / 255.0
  y = tf.cast(y, tf.int64)
  return x, y

@tf.autograph.experimental.do_not_convert
def create_dataset(xs, ys, n_classes=10):
  ''' build dataset from list of inputs, labels, and n_classes '''
  ys = tf.one_hot(ys, depth=n_classes)
  return tf.data.Dataset.from_tensor_slices((xs, ys)) \
    .map(preprocess) \
    .shuffle(len(ys)) \
    .batch(128)

@tf.autograph.experimental.do_not_convert
def ld_mnist():
    """Load training and test data."""

    def convert_types(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label
    

    dataset, info = tfds.load(
        "mnist", data_dir="gs://tfds-data/datasets", with_info=True, as_supervised=True
    )
    mnist_train, mnist_test = dataset["train"], dataset["test"]
    mnist_train = mnist_train.map(convert_types).shuffle(60000).batch(128)
    mnist_test = mnist_test.map(convert_types).shuffle(10000).batch(128)
    return EasyDict(train=mnist_train, test=mnist_test)

@tf.autograph.experimental.do_not_convert
def ld_mnist_onehot():
    """Load training and test data."""

    def convert_types(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255

        label = tf.one_hot(label, depth = 10)
        label = tf.cast(label, tf.int64)
        return image, label
    

    dataset, info = tfds.load(
        "mnist", data_dir="gs://tfds-data/datasets", with_info=True, as_supervised=True
    )
    mnist_train, mnist_test = dataset["train"], dataset["test"]
    mnist_train = mnist_train.map(convert_types).shuffle(60000).batch(128)
    mnist_test = mnist_test.map(convert_types).shuffle(10000).batch(128)
    return EasyDict(train=mnist_train, test=mnist_test)

# Determine how to build a model:
class Neural_Net(Model):
    def __init__(self, layers = 3, layers_list = [128, 64, 32]):
        super(Neural_Net, self).__init__()
        layers_dict = {}

        layers_dict['layer_0'] = Reshape(target_shape=(28 * 28,), input_shape=(28, 28))
        for i in range(layers):
            dict_entry = 'layer_' + str(i+1)
            layers_dict[dict_entry] = Dense(units = layers_list[i], 
                                            activation = 'relu')
        
        final_layer = 'layer_' + str(layers+1)
        layers_dict[final_layer] = Dense(10, activation = 'softmax')

        self.layers_dict = layers_dict

    def call(self, x):
        x = x
        for i in range(len(self.layers_dict)):
            key = list(self.layers_dict.keys())[i]
            
            #print(self.layers_dict[key])
            
            x = self.layers_dict[key](x)
        return(x)
    
def call_model(model, example, verbose = False):
    one_hot = model(example.reshape(1,28,28))
    if verbose == True:
        print(np.argmax(one_hot))
    return(np.argmax(one_hot))

def preprocess_single_for_pert(input_image, input_label):
    """
    Takes a single input_image, input_label and returns a converted
    version of each that can be input to plot_adv, or create a 
    perturbation mask in create_adversarial_pattern
    :param input_image: an array of the image
    :param input_label: an integer valued label
    :return: converted input_image, input_label
    """

    input_label = tf.cast(input_label, tf.int32)
    input_label = tf.one_hot(input_label, 10)
    input_label = tf.reshape(input_label, (1,10))

    input_image = input_image / 255.0
    input_image = input_image.reshape(1,28,28)
    input_image = tf.cast(input_image, tf.float32)
    input_image = tf.convert_to_tensor(input_image)
    return input_image, input_label

def plot_adv(input_image, eps, perturbations, model, secret_model):
    """
    Plots the original image with model predictions, the perturbation mask
    and the final adversarial image with model predictions.
    :param input_image: a tensor of the input image
    :param eps: the value of epsiol (value to perturb image)
    :param perturbations: the perturbation mask used to edit image
    :param model: the surrogate model (or first predictive model)
    :param secret_model: the secret model (or second predictive model)
    """

    adv_x = input_image + (eps * perturbations)

    fig, ax = plt.subplots(1, 3, figsize = (8,3))
    fig.suptitle(f'My First Adversarial Sample', fontsize = '12')

    ax[0].imshow(input_image[0])
    ax[0].set_xlabel(f'Surrogate Model: {np.argmax(model(input_image))} \n\
    Target Model: {np.argmax(secret_model(input_image))}')

    ax[1].imshow(perturbations[0] * 0.5 + 0.5)
    ax[1].set_xlabel(f'eps = {eps}')

    ax[2].imshow(adv_x[0])
    ax[2].set_xlabel(f'Surrogate Model: {np.argmax(model(adv_x))} \n\
    Target Model: {np.argmax(secret_model(adv_x))}')

    for ax in ax.flatten():
        ax.set_xticks([]) 
        ax.set_yticks([])
        #ax.axis('off')
    fig.show()