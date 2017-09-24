import os
import numpy as np
import tensorflow as tf
import cv2
from random import shuffle

normalize_image = False
image_shape = tuple()

with open('classifier_images/classifier_images.csv', 'r') as f:
    t = f.readlines()
    labels_dict = {}
    # compile the labels from the csv file into a dict with the file name as keys
    for i in t:
        s = i.split(",")
        if s:
            labels_dict[s[0]] = 2 if s[1].strip() == 'Green' else 1


def process_images(n=[], labels=[]):
    # loop through the folder and compile the labels read with the file names
    for i in os.listdir('classifier_images'):
        # in case there are other files in this folder
        if not i.endswith('.jpg'):
            print(i)
            continue

        image = cv2.imread(r'classifier_images\{}'.format(i))[:400:]

        resize_image = cv2.resize(image, (100, 50))
        hls_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2HLS)
        normalize_image = (hls_image.astype(np.float32) / 255.0) + 0.01
        # visualise the individual processed images here if necessary
        # plt.imshow(normalize_image, interpolation='nearest')
        # plt.show()

        n.append(normalize_image)
        if i in labels_dict:
            labels.append(labels_dict[i])

    image_shape = normalize_image.shape if normalize_image is not False else (0, 0, 0)
    return n, labels


features, labels = process_images()
# the first 1/10 of the images are set as the validation set. this set is always the same
# and is never shuffled
X_valid, y_valid = features[:int(len(features)/10)], labels[:int(len(features)/10)]
features, labels = features[int(len(features)/10):], labels[int(len(features)/10):]
# shuffle and set another 1/10 to test set
X_test, y_test = features[:int(len(features)/10)], labels[:int(len(features)/10)]
features, labels = features[int(len(features)/10):], labels[int(len(features)/10):]
# set batch processing parameters
nb_epoch = 100
batch_size = 12

X_train, y_train = features, labels
X_test, y_test = X_test, y_test

### Replace each question mark with the appropriate value.

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape
num_channels = 3  # grey scale

# TODO: How many unique classes/labels there are in the dataset.
n_classes = max(y_train) - min(y_train) + 1

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)



# Visualizations will be shown in the notebook.
# plt.imshow(X_train[np.random.randint(0, len(X_train))], interpolation='nearest', cmap=cm.brg)


from tensorflow.contrib.layers import flatten

filter_size = 5
depth = 12
depth2 = 24


def model(data):
    layer1_weights = tf.Variable(tf.truncated_normal(
        [filter_size, filter_size, num_channels, depth], mean=0, stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros(depth))
    conv1 = tf.nn.conv2d(data, filter=layer1_weights, strides=[1, 1, 1, 1], padding='VALID')
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    keep_prob1 = tf.Variable(0.5)  # dropout layer
    conv1 = tf.nn.relu(tf.nn.dropout(conv1 + layer1_biases, keep_prob1))

    layer2_weights = tf.Variable(tf.truncated_normal(
        [filter_size, filter_size, depth, depth2], mean=0, stddev=0.1))
    layer2_biases = tf.Variable(tf.zeros(depth2))
    conv2 = tf.nn.conv2d(conv1, filter=layer2_weights, strides=[1, 1, 1, 1], padding='VALID')
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    keep_prob2 = tf.Variable(0.5)  # dropout layer
    conv2 = tf.nn.relu(tf.nn.dropout(conv2 + layer2_biases, keep_prob2))

    fc0 = flatten(conv2)
    fc1_W = tf.Variable(tf.truncated_normal(shape=(4752, 1200), mean=0, stddev=0.1))
    fc1_b = tf.Variable(tf.zeros(1200))
    fc1 = tf.nn.relu(tf.matmul(fc0, fc1_W) + fc1_b)

    fc2_W = tf.Variable(tf.truncated_normal(shape=(1200, 400), mean=0, stddev=0.1))
    fc2_b = tf.Variable(tf.zeros(400))
    fc2 = tf.nn.relu(tf.matmul(fc1, fc2_W) + fc2_b)

    fc3_W = tf.Variable(tf.truncated_normal(shape=(400, n_classes),  mean=0, stddev=0.1))
    fc3_b = tf.Variable(tf.zeros(n_classes))

    return tf.matmul(fc2, fc3_W) + fc3_b


# Variables and Input data.
tf_train_dataset = tf.placeholder(tf.float32, shape=(None, image_shape[0], image_shape[1], num_channels))
tf_train_labels = tf.placeholder(tf.int32, shape=(None))

# Training computation.
one_hot_y = tf.one_hot(tf_train_labels, n_classes)
logits = model(tf_train_dataset)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits))

# introducing variable learning rate
global_step = tf.Variable(0)  # count the number of steps taken.
learning_rate = tf.train.exponential_decay(0.000221, global_step, 1000, 0.96)

# Optimizer.
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_operation = optimizer.minimize(loss, global_step=global_step)
# Predictions for the training, validation, and test data.
# train_prediction = tf.nn.softmax(logits)
# valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
# test_prediction = tf.nn.softmax(model(tf_test_dataset))

### Train your model here.
EPOCHS = 100

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={tf_train_dataset: batch_x, tf_train_labels: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...", "\n")
    for i in range(EPOCHS):
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={tf_train_dataset: batch_x, tf_train_labels: batch_y})

        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        test_accuracy = evaluate(X_test, y_test)
        print("Test Accuracy = {:.3f}".format(test_accuracy))

    saver = tf.train.Saver()
    save_path = saver.save(sess, r'C:\Users\ardav\Documents\python\CarND-Semantic-Segmentation\ts')

    builder = tf.saved_model.builder.SavedModelBuilder(r'C:\Users\ardav\Documents\python\CarND-Semantic-Segmentation\ts')
    predict_tensor_inputs_info = tf.saved_model.utils.build_tensor_info(tf_train_dataset)
    predict_tensor_scores_info = tf.saved_model.utils.build_tensor_info(model(tf_train_dataset))

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'images': predict_tensor_inputs_info},
            outputs = {'scores': predict_tensor_scores_info},
            method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map={'predict_images': prediction_signature},
        legacy_init_op=legacy_init_op)
    builder.save()

    print("Model saved to {}".format(save_path))



### Load step

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('ts.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    sess.run(tf.global_variables_initializer())
    softmax_predictions = sess.run(tf.nn.softmax(logits), feed_dict={tf_train_dataset: X_valid,
                                                                     tf_train_labels: y_valid})
    accuracy = sess.run(accuracy_operation, feed_dict={tf_train_dataset: X_valid,
                                                                     tf_train_labels: y_valid})
    print("Real Image Test Accuracy = {:.3f}".format(accuracy))

    sess.run()
