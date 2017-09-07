# Python 3.6
from tensorflow.examples.tutorials.mnist import input_data
from network import Network


def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    model = Network(mnist)
    model.build_graph(learning_rate=0.5)
    model.train()
    model.eval()
    return

main()