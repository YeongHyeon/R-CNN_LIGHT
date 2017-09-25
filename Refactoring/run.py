import source.developed as developed
developed.print_stamp()

import argparse

import matplotlib.pyplot as plt
import tensorflow as tf

# custom modules
import source.utility as utility
import source.functions as functions
import source.mash_plot as mash_plot
import source.data_handler as data_handler
import source.model as model
import source.sub_procedure as sub_procedure

def main():

    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']

    print("")

    if((not(data_handler.check())) or (FLAGS.make)):
        path = input("Enter the source path: ")
        utility.data_booster(path=path, clone=FLAGS.boost, extentions=extentions)
        data_handler.make(path=path, imsize=32, extentions=extentions)

    dataset = data_handler.load()

    sess = tf.InteractiveSession()

    data_size = dataset.train.data_size
    classes = dataset.train.class_num

    data = tf.placeholder(tf.float32, shape=[None, data_size])
    label = tf.placeholder(tf.float32, shape=[None, classes])
    training = tf.placeholder(tf.bool)

    train_step, accuracy, loss, prediction = model.convolution_neural_network(x=data, y_=label, training=training, data_size=data_size, classes=classes)

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
