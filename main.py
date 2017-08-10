import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    def_graph = tf.get_default_graph()

    in_tensor = def_graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = def_graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = def_graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = def_graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = def_graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return in_tensor, keep_prob, layer3, layer4, layer7
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    l3_num_outputs = int(vgg_layer3_out.shape[-1])
    l4_num_outputs = int(vgg_layer4_out.shape[-1])
    l7_num_outputs = int(vgg_layer7_out.shape[-1])

    l3_ker_size = (4, 4)
    l3_weights = tf.Variable(tf.truncated_normal([64, 64, 3, num_classes], mean=0, stddev=0.1))
    l3_biases = tf.Variable(tf.zeros(num_classes))
    l3_deconv = tf.layers.conv2d_transpose(vgg_layer3_out, num_classes, kernel_size=l3_ker_size, strides=(8, 8),
                                           padding='valid')
    l3_keep_prob = tf.Variable(0.5)  # dropout layer
    l3_relu = tf.nn.relu(tf.matmul(l3_weights, l3_deconv) + l3_biases)
    l3_dropout = tf.nn.dropout(l3_relu, l3_keep_prob)

    l4_ker_size = (4, 4)
    l4_weights = tf.Variable(tf.truncated_normal([64, 64, 3, num_classes], mean=0, stddev=0.1))
    l4_biases = tf.Variable(tf.zeros(num_classes))
    l4_deconv = tf.layers.conv2d_transpose(vgg_layer4_out, num_classes, kernel_size=l4_ker_size, strides=(4, 4),
                                           padding='valid')
    l4_keep_prob = tf.Variable(0.5)  # dropout layer
    l4_relu = tf.nn.relu(tf.matmul(l4_weights, l4_deconv) + l4_biases)
    l4_dropout = tf.nn.dropout(l4_relu, l4_keep_prob)


    l7_ker_size = (16, 16)
    l7_weights = tf.Variable(tf.truncated_normal([64, 64, 3, num_classes], mean=0, stddev=0.1))
    l7_biases = tf.Variable(tf.zeros(num_classes))
    l7_deconv = tf.layers.conv2d_transpose(vgg_layer7_out, num_classes, kernel_size=l7_ker_size, strides=(1, 1),
                                           padding='valid')
    l7_keep_prob = tf.Variable(0.5)  # dropout layer
    l7_relu = tf.nn.relu(tf.matmul(l7_weights, l7_deconv) + l7_biases)
    l7_dropout = tf.nn.dropout(l7_relu, l7_keep_prob)

    return l3_dropout + l4_dropout + l7_dropout
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    return None, None, None
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    pass
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function

        # TODO: Train NN using the train_nn function

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
