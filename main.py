#加载自定义模块
import trainModel as tm
from tensorflow.examples.tutorials.mnist import input_data

def main(argv=None):
	mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
	tm.train(mnist)


if __name__ == '__main__':
	main()
	