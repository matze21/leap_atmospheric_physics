import tensorflow as tf
from keras import backend as K

def r2_scoretf(y_true, y_pred):
    sum_squares_residuals = tf.reduce_sum(tf.square(y_true - y_pred), axis=0)
    sum_squares_total = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)), axis=0)
    r2 = 1 - (sum_squares_residuals / sum_squares_total)
    return r2 #tf.reduce_mean(r2)

def r2_scoreTrain(y_true, y_pred):
    sum_squares_residuals = tf.reduce_sum(tf.square(y_true - y_pred), axis=0)
    sum_squares_total = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)), axis=0)
    r2 = (sum_squares_residuals / sum_squares_total) # alwaysPositive, the smaller the better
    return tf.reduce_mean(r2)

class RSquaredMetric(tf.keras.metrics.Metric):
    def __init__(self, name='r_squared', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_sum_squares = None#self.add_weight(name='total_sum_squares', initializer='zeros', shape=shape)
        self.residual_sum_squares = None#self.add_weight(name='residual_sum_squares', initializer='zeros', shape=shape)
        self.num_samples = self.add_weight(name="num_samples", initializer='zeros',dtype=tf.int32)
 
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)
        
        sum_squares_residuals = tf.reduce_sum(tf.square(y_true - y_pred), axis=0)
        sum_squares_total = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true, axis=0)), axis=0)
        sum_squares_total = tf.where(tf.equal(sum_squares_total, 0.0), tf.ones_like(sum_squares_total), sum_squares_total)
        
        if self.total_sum_squares is None:
            self.total_sum_squares = self.add_weight(name='total_sum_squares', initializer='zeros', shape=sum_squares_total.shape)
            self.residual_sum_squares = self.add_weight(name='residual_sum_squares', initializer='zeros', shape=sum_squares_residuals.shape)

        self.total_sum_squares.assign_add(sum_squares_total)
        self.residual_sum_squares.assign_add(sum_squares_residuals)

    def result(self):
        r_squared = 1 - (self.residual_sum_squares / self.total_sum_squares)
        r_squared = tf.where(tf.math.is_nan(r_squared), tf.ones_like(r_squared), r_squared)
        return tf.reduce_mean(r_squared)

    def reset_state(self):
        self.total_sum_squares.assign(tf.zeros_like(self.total_sum_squares))
        self.residual_sum_squares.assign(tf.zeros_like(self.residual_sum_squares))