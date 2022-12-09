import tensorflow as tf
import utils.sparsemax as spm
import datasets
import numpy as np
import keras


full, train, test = datasets._mulan('scene', pre_split=True)
print(train.targets)
inputs = np.array(train.inputs)
targets = np.array(train.targets)
print(targets.shape)
print(inputs.shape)


class MultinomialRegressor(tf.Module):

  def __init__(self, n_class, feature_dim):
    self.built = False
    self.n_class = n_class
    self.layers = []
    # self.layers.append(Dense(input_dim=x.shape[-1], output_size=n_class))
    self.rand_w = tf.random.uniform(shape=[feature_dim, self.n_class], seed=22)
    self.w = tf.Variable(initial_value=self.rand_w, name='w',trainable=True)
    self.b = tf.Variable(initial_value=tf.zeros([n_class]), name='b', trainable=True)

    # self.w = self.add_weight(
    #         shape=(feature_dim, self.n_class), initializer="random_normal", trainable=True
    #     )
    # self.b = self.add_weight(shape=(n_class,), initializer="zeros", trainable=True)
  def __call__(self, x, train=True):
    # Initialize the model parameters on the first call

    # if not self.built:
    #   # Randomly generate the weights
    #   self.rand_w = tf.random.uniform(shape=[x.shape[-1], self.n_class], seed=22)
    #   self.w = tf.Variable(rand_w, name='w')
    #   self.b = tf.Variable(tf.zeros([n_class]), name='b')
    #   self.built = True
    # Compute the model output
    z = tf.matmul(tf.cast(x, 'float32'), self.w) + self.b

    sparsemax_pred = spm.sparsemax(z)
    return sparsemax_pred

  def get_w(self):
    return self.rand_w
    


def train(batch_size, epochs):
  step = 0
  full, train, test = datasets._mulan('scene', pre_split=True)
  regressor = MultinomialRegressor(6, train.inputs.shape[1])
  print(regressor.trainable_variables)
  train_inputs = np.array(train.inputs)
  train_targets = np.array(train.targets)
  loss_fn = tf.keras.losses.CategoricalCrossentropy()
  optimizer = tf.keras.optimizers.Adam()

  for j in range(epochs):
    for i in range(train_inputs.shape[0] // batch_size):
      step += 1;
      x_batch = train_inputs[:(i+1)*batch_size,:]
      y_batch = train_targets[:(i+1)*batch_size,:]
      with tf.GradientTape() as tape:

        pred_batch = regressor(x_batch)
        # print(pred_batch[0])
        loss_value = loss_fn(y_batch, pred_batch)

      grads = tape.gradient(loss_value, regressor.trainable_variables)
      optimizer.apply_gradients(zip(grads, regressor.trainable_variables))

      if step % 5 == 0:
              print(
                  "Training loss (for one batch) at step %d: %.4f"
                  % (step, float(loss_value))
              )
              print("Seen so far: %s samples" % ((step + 1) * batch_size))


train(5, 10)






    # return tf.nn.softmax(z)


