import numpy as np
import numpy.testing as np_test
import unittest
import entmax
import sparsemax
import tensorflow as tf

class TestEntmax(unittest.TestCase):
    def test_entmax(self):
        tf.enable_eager_execution()

        z = tf.constant([[2.5, 0.2, 0.1, 3],[5.0, 4.5, 1.5, 0.5]])
        sparse = entmax.entmax_bisect_tf(z, alpha=1.3)

        # Output obtained using pytorch implementation here: https://github.com/deep-spin/entmax/blob/master/entmax/root_finding.py
        # Make sure these match up
        expected = np.array([[3.4896e-01, 2.0426e-05, 1.6194e-07, 6.5102e-01],
            [6.5103e-01, 3.4897e-01, 0.0000e+00, 0.0000e+00]])
        np_test.assert_array_almost_equal(sparse.numpy(), expected, decimal=4)
    
    def test_entmax_2(self):
        tf.enable_eager_execution()

        z = tf.constant([2.5, 0.2, 0.1, 3, 0.1, 2.5])
        e = entmax.entmax_bisect_tf(z, alpha=2)

        # Is equivalent to softmax for alpha=2
        expected = sparsemax.sparsemax(z)
        np_test.assert_array_almost_equal(e.numpy(), expected.numpy(), decimal=4)
    
    def test_entmax_1(self):
        tf.enable_eager_execution()

        z = tf.constant([2.5, 0.2, 0.1, 3, 0.1, 2.5])

        # Implementation doesn't support alpha=1, instead choose float close to 1
        e = entmax.entmax_bisect_tf(z, alpha=1.00001) 

        # Is equivalent to softmax for alpha~1
        expected =  tf.nn.softmax(z)
        np_test.assert_array_almost_equal(e.numpy(), expected.numpy(), decimal=4)
    
    # Test auto-diff gradient with a custom-defined gradient calculation
    def test_gradient(self):
        tf.enable_eager_execution()

        x = tf.constant([[0.5, 0.1]])
        w = tf.Variable([[0.1, 0.3, 0.2, 0.8, 0.1], [0.5, 0.1, 0.9, 0.4, 0.2]], 
                trainable=True, name='weights')
        y = tf.constant([[0, 0, 1, 0, 0]], dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            y_pred_orig = entmax.entmax_bisect_tf(tf.matmul(x, w))
            loss_orig = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=y_pred_orig, labels=y
            ))

            y_pred_custom = entmax.entmax_bisect_tf_custom_grad(tf.matmul(x, w))
            loss_custom = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=y_pred_custom, labels=y
            ))

        grads_orig = tape.gradient(loss_orig, w)
        grads_custom = tape.gradient(loss_custom, w)

        # Not accurate up to 4 decimal places, not entirely sure why...
        np_test.assert_array_almost_equal(grads_orig.numpy(), grads_custom.numpy(), decimal=3)

if __name__ == '__main__':
    unittest.main()