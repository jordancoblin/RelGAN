import numpy as np
import numpy.testing as np_test
import unittest
import sparsemax
import tensorflow as tf

class TestSparsemax(unittest.TestCase):
    def test_sparsemax(self):
        tf.enable_eager_execution()

        z = tf.constant([2.5, 0.2, 0.1, 3, 0.1, 2.5])
        s = sparsemax.sparsemax(z)

        expected = np.array([0.1667, 0.0000, 0.0000, 0.6667, 0.0000, 0.1667])
        np_test.assert_array_almost_equal(s.numpy(), expected, decimal=4)

    def test_sparsegen_lin(self):
        tf.enable_eager_execution()

        z = tf.constant([2.5, 2.0, 0.1, 3, 0.1, 2.5])

        # Is equivalent to sparsemax for lambda = 0
        s = sparsemax.sparsegen_lin(z, lam=-0.)
        expected = sparsemax.sparsemax(z)
        np_test.assert_array_almost_equal(s.numpy(), expected.numpy(), decimal=4)

        # Becomes less sparse as lambda is decreased
        s2 = sparsemax.sparsegen_lin(z, lam=-2.0)
        s2_nonzero = tf.count_nonzero(s2)
        sparsemax_nonzero = tf.count_nonzero(s)
        self.assertGreater(s2_nonzero.numpy(), sparsemax_nonzero.numpy())
    
    # Test auto-diff gradient with a custom-defined gradient calculation
    def test_gradient(self):
        tf.enable_eager_execution()

        x = tf.constant([[0.5, 0.1]])
        w = tf.Variable([[0.1, 0.3, 0.2, 0.8, 0.1], [0.5, 0.1, 0.9, 0.4, 0.2]], 
                trainable=True, name='weights')
        y = tf.constant([[0, 0, 1, 0, 0]], dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            y_pred_orig = sparsemax.sparsemax(tf.matmul(x, w))
            loss_orig = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=y_pred_orig, labels=y
            ))

            y_pred_custom = sparsemax.sparsemax_custom_grad(tf.matmul(x, w))
            loss_custom = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=y_pred_custom, labels=y
            ))

        grads_orig = tape.gradient(loss_orig, w)
        grads_custom = tape.gradient(loss_custom, w)
        np_test.assert_array_equal(grads_orig.numpy(), grads_custom.numpy())

if __name__ == '__main__':
    unittest.main()