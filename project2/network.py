
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


class Network(object):
    def __init__(self, dataset, learning_rate=0.5):
        # lr = 0.5
        self.dataset = dataset
        self.input_number =
        self.output_number = 
        self.neural_layer = []
        # Get the input vector length
        self.num_classes = dataset.train.labels.shape[1]
        self.learning_rate = learning_rate

    def softmax(self, x):
        return tf.nn.softmax(x)

    def relu(self, x):
        return tf.nn.relu(x)

    def sigmoid(self, x):
        return tf.sigmoid(x)

    def build_graph(self, learning_rate=0.5):
        self.x = tf.placeholder(tf.float32, [None, self.input_number])
        # W represents the transformation into a layer
        # W = tf.Variable(tf.zeros([784, 10]))
        # this is the biases
        # b = tf.Variable(tf.zeros([self.num_classes]))
        current_value = self.x
        for i in range(len(self.neural_layer)):
            current_value = tf.matmul(current_value, self.neural_layer[i]['W']) +\
                                      self.neural_layer[i]['b']
            if i == len(self.neural_layer)-1:
                # Idea is to not pump it through sigmoid, but the softmax for classification.
                continue
            # current_value = self.softmax(current_value)
            # current_value = self.sigmoid(current_value)
            current_value = self.relu(current_value)
        self.y = self.softmax(current_value)
        # self.y = current_value
        self.y_ = tf.placeholder(tf.float32, [None, self.num_classes])
        self.cross_entropy = tf.reduce_mean(
            -tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))
        self.train_step = tf.train.GradientDescentOptimizer(
            self.learning_rate).minimize(self.cross_entropy)

    def train(self, epoch_number=1000, batch_number=100):
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for _ in range(epoch_number):
            batch_xs, batch_ys = self.dataset.train.next_batch(batch_number)
            self.sess.run(self.train_step,
                          feed_dict={self.x: batch_xs,
                                     self.y_: batch_ys})

    def define_layers(self, output_numbers):
        for i in range(len(output_numbers)-1):
            self.neural_layer += [
                {'W': tf.Variable(tf.random_uniform(
                 [output_numbers[i], output_numbers[i+1]], minval=-1, maxval=1)),
                 'b':tf.Variable(tf.random_uniform([output_numbers[i+1]], minval=-1, maxval=1))}]

        # TODO: insert bias at each level.
        # NOTE: Originale
        # self.neural_layer += [tf.Variable(tf.zeros([input_number, self.output_number]))]
        # NOTE: Elliot Greenless uses tf.truncated_normal([input_size, output_size], stdev=1))
        return

    def eval(self):
        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        score = self.sess.run(accuracy,
                              feed_dict={self.x: self.dataset.test.images,
                                         self.y_: self.dataset.test.labels})
        return score

# if __name__ == "__main__":
#     p = Network()
#     p.train()
#     p.eval()
