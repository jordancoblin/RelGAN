import tensorflow as tf
import utils.sparsemax as spm
import datasets


# (full, train, test) = datasets.all_datasets[0]
print(datasets._mulan('scene', pre_split=True))


class MultinomialRegression(tf.Module):

  def __init__(self, n_class):
    self.built = False
    self.n_class = n_class

  def __call__(self, x, train=True):
    # Initialize the model parameters on the first call
    dataset = _mulan()
    if not self.built:
      # Randomly generate the weights
      rand_w = tf.random.uniform(shape=[x.shape[-1], self.n_class], seed=22)
      self.w = tf.Variable(rand_w)
      self.built = True
    # Compute the model output
    z = tf.matmul(x, self.w)

    return spm.sparsemax(z)
    # return tf.nn.softmax(z)


