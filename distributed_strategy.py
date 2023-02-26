
import tensorflow as tf
import numpy as np
import os

os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "4"
# if different type of GPU then use cross_device_ops
strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
print("number of devices :  {}".format(strategy.num_replicas_in_sync))
# create dataset :

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images[..., None]  # add channel dimension
test_images = test_images[..., None]
train_images = train_images / np.float32(255)  # normalizing
test_images = test_images / np.float32(255)
BUFFER_SIZE = len(train_images)  #

ds = tf.data.Dataset.from_tensor_slices(images)


def scale(image, label):  # preprocess dataset
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label


BATCH_SIZE_PER_REPLICA = 64  # batch size for each replica
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync  # total batch size

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(
    GLOBAL_BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)

# dataset for distributed strategy
train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

# model for Guided CTC
def create_model():
    model = tf.keras.models.Sequential([])
    return model


# custom training inside stategy
with strategy.scope()
    model = create_model()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                reduction=tf.keras.losses.Reduction.NONE)


    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)


    test_loss = tf.keras.metrics.Mean(name='test_loss')
    training_accuracy = tf.keras.metrics.SparseCategoricalCrossentropy(name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalCrossentropy(name='test_accuracy')
    optimizer = tf.keras.optimizers.Adam()

train_mnist.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
with strategy.scope():
    model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
EPOCHS = 10
for epoch in EPOCHS:
    total_loss = 0.0
    num_batches = 0
    for batch in train_dist_dataset:
        total_loss += distributed_train_step(batch)
        num_batches += 1
    total_loss = total_loss / num_batches


@tf.function
def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
    tf.print(per_replica_losses.values)
    foo = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    tf.print(foo)
    return foo


def train_step(inputs):
    images, labels = inputs
    with tf.GradientTape as tape:
        predictions = model(images, traininig=True)
        loss = compute_loss(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_accuracy.update_state(labels, predictions)
    return loss



########################
@tf.function
def distributed_test_step(dataset_inputs):
    return strategy.run(test_step, args=(dataset_inputs))


def test_step(inputs):
    image, label = inputs
    predictions = model(images, training=False) # model for inference
    t_loss = loss_object(labels, predictions)  #
    total_loss.update_state(t_loss)
    test_accuracy.update_state(labels, predictions)

    #######################


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten

input = Input(shape=(28, 28))
x = Flatten()(input)
x = Dense(128, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)
func_model = Model(inputs=input, outputs=predictions)
model._layers = [layer for layer in model._layers if isinstance(layer, Layer)]
# single input 2 output
loss = {'y_1': 'mse', 'y_2': 'mse'}

loss, loss_y1, loss_y2, acc_y1, acc_y2 = model.evaluate() #single input 2 output


class my_model(tf.keras.models):
    def __init__(self):

#huber loss implementation by function
def my_loss_function(y_true, y_pred):
    threshold = 1
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= threshold
    small_error_loss = tf.square(error) / 2
    big_error_loss = threshold * (tf.abs(error) - threshold / 2)
    return tf.where(is_small_error, small_error_loss, big_error_loss)

#
model.compile(loss=my_loss_function)

#huber loss implementation by class
class myhuberloss(tf.keras.losses.Loss):

    def __init__(self, threshold, reduction=losses_utils.ReductionV2.AUTO, name=None):
        super().__init__(reduction, name)
        self.threshold = threshold

    def call(self, y_true, y_pred):
        error = y_true - y_pred

        self.threshold


# custom layer
# 1)lambda tf.keras.layers.Lambda(lambda x:tf.abs(x)) #not trainable maybe
# 2)
class SimpleDense(tf.keras.layers.Layer):

    def __init__(self, units=32):
        super(SimpleDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(name='kernel', initial_value=w_init(shape=(input_shape[-1],
                                                                        self.units),
                                                                 dtype='float32'),
                             trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(name='bias',
                             initial_value=b_init(shape=(self.units,), dtype='float32'),
                             trainable=true)

    def call(self, inputs):
        return tf.matmul(self.w * inputs + self.b)

    my_dense = SimpleDense(units=1)
    x = tf.ones

#custom model
class CNNRES(tf.keras.models.Model):
    def __init__(self, layers, filters, activation=None, **kwargs):
        super(CNNRES, self).__init__(**kwargs)
        self.hidden = [tf.keras.layers.Conv2D(filters, (3, 3), activation='relu') for _ in range(layers)]

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        return x + inputs


class Myres(Model):
    def __init__(self, **kwargs):
        self.hidden1 = Dense(30, activation='relu')
        self.hidden2 = CNNRes(2, 32)

    def call(self, inputs):
        pass


###
tf.nn.ctc_loss(
    labels,
    logits,
    label_length,
    logit_length,
    logits_time_major=True,
    unique=None,
    blank_index=None,
    name=None
)

