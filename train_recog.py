import idx2numpy
import pandas as pd
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# import tensorflow as tf

# np.random.seed(1337)

desired_width = 170
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width, precision=3)
train_lab_file = r'emnist-balanced-train-labels-idx1-ubyte'
train_img_file = r'emnist-balanced-train-images-idx3-ubyte'
test_lab_file = r'emnist-balanced-test-labels-idx1-ubyte'
test_img_file = r'emnist-balanced-test-images-idx3-ubyte'

# train_lab_file = r'emnist-mnist-train-labels-idx1-ubyte'
# train_img_file = r'emnist-mnist-train-images-idx3-ubyte'
# test_lab_file = r'emnist-mnist-test-labels-idx1-ubyte'
# test_img_file = r'emnist-mnist-test-images-idx3-ubyte'

train_labels = idx2numpy.convert_from_file(train_lab_file)
train_imgs = idx2numpy.convert_from_file(train_img_file)
test_labels = idx2numpy.convert_from_file(test_lab_file)
test_imgs = idx2numpy.convert_from_file(test_img_file)

# num = 4
#
# flipped = np.transpose(train_imgs[num])
# print(flipped)
# print(train_labels[num])

#
# feature_columns = [tf.feature_column.numeric_column('x', shape=train_imgs.shape[1:])]
#
# estimator = tf.estimator.DNNClassifier(
#     feature_columns=feature_columns,
#     hidden_units=[300, 100],
#     n_classes=10,
#     model_dir = '/train/DNN')
########################################

# import tensorflow as tf
# import numpy as np


# from tensorflow.examples.tutorials.mnist import input_data
# import input_data

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# def next_batch(num, data, labels):
#     idx = np.arange(0, len(data))
#     np.random.shuffle(idx)
#     idx = idx[:num]
#     data_shuffle = [data[i] for i in idx]
#     labels_shuffle = [labels[i] for i in idx]
#     return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def one_hot(single_value):
    newList = np.zeros(47)
    for i in range(0, 47):
        newList[i] = 0
    newList[single_value] = 1
    return newList



# print(mnist.train.labels[0])
# print(mnist.train.images[0])
#
# print(train_imgs[0])
# print(train_labels[0])

format_train_imgs = np.zeros_like((train_imgs), dtype=np.float64)
format_train_labels = np.zeros((len(train_labels),47))
train_imgs_pre = np.zeros((len(train_imgs), len(train_imgs[0])*len(train_imgs[0])), dtype=np.float64)
for item in range(0, len(train_imgs)):
    format_train_imgs[item] = np.transpose(train_imgs[item])
    format_train_imgs[item] = np.true_divide(format_train_imgs[item], 255)
    train_imgs_pre[item] = format_train_imgs[item].flatten()
    format_train_labels[item] = one_hot(train_labels[item])

# print(train_imgs_pre[0])
# print(format_train_labels[0])


# print(mnist.test.images[0])
# print(mnist.test.labels[0])

format_test_imgs = np.zeros_like((test_imgs), dtype=np.float64)
format_test_labels = np.zeros((len(test_labels),47))
test_imgs_pre = np.zeros((len(test_imgs), len(test_imgs[0])*len(test_imgs[0])), dtype=np.float64)

for item in range(0, len(test_imgs)):
    format_test_imgs[item] = np.transpose(test_imgs[item])

    format_test_imgs[item] = np.true_divide(format_test_imgs[item], 255)
    test_imgs_pre[item] = format_test_imgs[item].flatten()
    format_test_labels[item] = one_hot(test_labels[item])

# print(test_imgs_pre[3])
# print(format_test_labels[3])



# Python optimisation variables
learning_rate = 0.5
epochs = 10
batch_size = 100

# declare the training data placeholders
# input x - for 28 x 28 pixels = 784
x = tf.placeholder(tf.float32, [None, 784], name='x')
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 47], name='y')

# now declare the weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([300]), name='b1')
# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([300, 47], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([47]), name='b2')


saver = tf.train.Saver()


# calculate the output of the hidden layer
hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)

# now calculate the hidden layer output - in this case, let's use a softmax activated
# output layer
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))
tf.identity(y_, name="y_")

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))

# add an optimiser
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# start the session
with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    total_batch = int(len(train_labels) / batch_size)

    for epoch in range(epochs):
        avg_cost = 0
        currArea = 0

        for i in range(total_batch):
            # batch_x, batch_y = next_batch(batch_size, train_imgs, train_labels)
            batch_x = []
            batch_y = []
            # print(currArea, currArea + batch_size)
            for rx in range(currArea, currArea+batch_size):
                batch_x.append(train_imgs_pre[rx])
                batch_y.append(format_train_labels[rx])

            currArea += batch_size

            _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))

    test_img_fin = []
    test_labels_fin = []
    for i in range(0, len(test_imgs)):
        test_temp_im = test_imgs[i].transpose()
        test_temp_im = test_temp_im.flatten()
        test_img_fin.append(test_temp_im)

        one_hot_test_y = one_hot(test_labels[i])
        test_labels_fin.append(one_hot_test_y)

    # print(sess.run(accuracy, feed_dict={x: test_img_fin, y: test_labels_fin}))
    print(test_imgs_pre[0])
    print(sess.run(tf.math.argmax(y_[0]), feed_dict={x: [test_imgs_pre[0]]}))
    print(sess.run(accuracy, feed_dict={x: test_imgs_pre, y: format_test_labels}))
    saver.save(sess, 'emnist_balanced_model')


