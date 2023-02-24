import tensorflow as tf
from easydict import EasyDict
from tensorflow.keras import Model
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, Reshape

@tf.autograph.experimental.do_not_convert
def preprocess(x, y):
  x = tf.cast(x, tf.float32) / 255.0
  y = tf.cast(y, tf.int64)
  return x, y

@tf.autograph.experimental.do_not_convert
def create_dataset(xs, ys, n_classes=10):
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
    
def call_model(model, example):
    one_hot = model(example.reshape(1,28,28))
    print(np.argmax(one_hot))