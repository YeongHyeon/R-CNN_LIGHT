import os, sys, inspect, shutil

import tensorflow as tf
import numpy as np

import utility as utility

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def training_process(sess=None, dataset=None,
                     x=None, y_=None, training=None,
                     train_step=None, accuracy=None, loss=None, saver=None,
                     batch_size=0, epochs=0):

    print("\n** Training process start!")

    te_am = dataset.test.amount
    if(batch_size > te_am):
        batch_size = te_am

    epoch_step = 1
    if(epochs < 100):
        epoch_step = 1
    elif(epochs < 1000):
        epoch_step = 10
    else:
        epoch_step = int(epochs/100)

    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []

    print("\n Training to "+str(epochs)+" epochs | Batch size: %d\n" %(batch_size))

    tra_am = dataset.train.amount
    for i in range(epochs):

        start = 0
        end = start + batch_size

        if(i%epoch_step == 0): # compute accuracy and loss
            if(start == 0):
                train_batch = dataset.train.next_batch(batch_size=batch_size)
            test_batch = dataset.test.next_batch(batch_size=batch_size)

            sys.stdout.write(" Evaluation        \r")
            sys.stdout.flush()

            train_accuracy = accuracy.eval(feed_dict={x:train_batch[0], y_:train_batch[1], training:False})
            test_accuracy = accuracy.eval(feed_dict={x:test_batch[0], y_:test_batch[1], training:False})
            train_loss = loss.eval(feed_dict={x:train_batch[0], y_:train_batch[1], training:False})
            test_loss = loss.eval(feed_dict={x:test_batch[0], y_:test_batch[1], training:False})

            train_acc_list.append(train_accuracy)
            test_acc_list.append(test_accuracy)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)

            print(" Epoch [ %d / %d ]\n Accuracy  train: %.5f  |  test: %.5f" %(i, epochs, train_accuracy, test_accuracy))
            print(" CE loss   train: %.5f  |  test: %.5f" %(train_loss, test_loss))
            print("")

        while(True): # 1 epoch
            sys.stdout.write(" Loading next batch\r")
            sys.stdout.flush()
            train_batch = dataset.train.next_batch(batch_size=batch_size, start=start, end=end)

            sys.stdout.write(" Training          \r")
            sys.stdout.flush()
            sess.run(train_step, feed_dict={x:train_batch[0], y_:train_batch[1], training:True})

            start = end
            end = start + batch_size
            if(start >= tra_am):
                break

    utility.save_graph_as_image(train_list=train_acc_list, test_list=test_acc_list, ylabel="accuracy")
    utility.save_graph_as_image(train_list=train_loss_list, test_list=test_loss_list, ylabel="loss")

    test_batch = dataset.test.next_batch(batch_size=batch_size)
    test_accuracy = accuracy.eval(feed_dict={x:test_batch[0], y_:test_batch[1], training:False})
    test_loss = loss.eval(feed_dict={x:test_batch[0], y_:test_batch[1], training:False})
    print("\n Final Test accuracy, loss  | %.5f\t %.5f\n" %(test_accuracy, test_loss))

    utility.refresh_directory(PACK_PATH+"/checkpoint")
    saver.save(sess, PACK_PATH+"/checkpoint/checker")

def prediction_process(sess=None, dataset=None,
                       x=None, y_=None, training=None,
                       prediction=None, saver=None,
                       validation=0):

    print("\n** Prediction process start!")

    val_am = dataset.validation.amount
    if(validation == 0):
        val_loop = val_am
    else:
        val_loop = validation
        if(val_loop > val_am):
            val_loop = val_am

    correct = 0
    if(os.path.exists(PACK_PATH+"/checkpoint/checker.index")):
        saver.restore(sess, PACK_PATH+"/checkpoint/checker")

        f = open(PACK_PATH+"/dataset/labels.txt", 'r')
        content = f.readlines()
        f.close()
        for idx in range(len(content)):
            content[idx] = content[idx][:len(content[idx])-1] # rid \n

        print("\n Prediction to "+str(val_loop)+" times")

        prob_list = []
        prob_matrix = np.empty((0, dataset.validation.class_num), float)

        line_cnt = 0
        tmp_label = 0
        for i in range(val_loop):
            valid_batch = dataset.validation.next_batch(batch_size=1, nth=line_cnt)
            line_cnt += 1

            if(tmp_label != int(np.argmax(valid_batch[1]))):
                prob_list.append(prob_matrix)
                prob_matrix = np.empty((0, dataset.validation.class_num), float)
                tmp_label = int(np.argmax(valid_batch[1]))

            prob = sess.run(prediction, feed_dict={x:valid_batch[0], training:False})
            prob_matrix = np.append(prob_matrix, np.asarray(prob), axis=0)

            print("\n Prediction")
            print(" Real:   "+str(content[int(np.argmax(valid_batch[1]))]))
            print(" Guess:  "+str(content[int(np.argmax(prob))])+"  %.2f %%" %(np.amax(prob)*100))

            if(content[int(np.argmax(valid_batch[1]))] == content[int(np.argmax(prob))]):
                correct = correct + 1
        prob_list.append(prob_matrix)

        print("\n Accuracy: %.5f" %(float(correct)/float(val_loop)))
        utility.save_confusion(save_as="confusion", labels=content, lists=prob_list, size=dataset.validation.class_num)
    else:
        print("You must training first!")
