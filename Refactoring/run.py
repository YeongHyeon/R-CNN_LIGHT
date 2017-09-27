import sys, os, inspect, argparse
PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(PACK_PATH)
sys.path.append(PACK_PATH+"/source")

import developed
developed.print_stamp()

import matplotlib.pyplot as plt
import tensorflow as tf

# custom modules
import utility
import functions
import data_handler
import model
import sub_procedure

def main():

    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']

    print("")

    if((not(data_handler.check())) or (FLAGS.make)):
        path = raw_input("Enter the source path: ")
        data_handler.make(path=path, height=32, width=32, extensions=extensions, clone=FLAGS.boost)

    dataset = data_handler.load()

    sess = tf.InteractiveSession()

    data_size, height, width, channel = dataset.train.data_size
    classes = dataset.train.class_num

    data = tf.placeholder(tf.float32, shape=[None, data_size])
    label = tf.placeholder(tf.float32, shape=[None, classes])
    training = tf.placeholder(tf.bool)

    train_step, accuracy, loss, prediction = model.convolution_neural_network(x=data, y_=label, training=training, height=height, width=width, channel=channel, classes=classes)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    sub_procedure.training_process(sess=sess, dataset=dataset, x=data, y_=label, training=training, train_step=train_step, accuracy=accuracy, loss=loss, saver=saver, batch_size=FLAGS.batch, epochs=FLAGS.epochs)
    sub_procedure.prediction_process(sess=sess, dataset=dataset, x=data, y_=label, training=training, prediction=prediction, saver=saver, validation=FLAGS.validation)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--make', type=bool, default=False, help='Default: False. Enter True to update the dataset.')
    parser.add_argument('--boost', type=int, default=0, help='Default: 0. ')
    parser.add_argument('--batch', type=int, default=10, help='Default: 10. Batches per iteration, the number of data to be training and testing.')
    parser.add_argument('--epochs', type=int, default=100, help='Default: 100')
    parser.add_argument('--validation', type=int, default=0, help='Default: 0')
    FLAGS, unparsed = parser.parse_known_args()

    main()
