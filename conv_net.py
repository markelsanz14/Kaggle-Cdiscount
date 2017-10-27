import tensorflow as tf
from get_next_batch import get_next_batch

def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv_net(category_to_int, int_to_category):
    num_iterations = 20000
    tr_batch_size = 32
    val_batch_size = 1024

    x = tf.placeholder(tf.float32, [None, 180*180*3])
    x_image = tf.reshape(x, [-1, 180, 180, 3])

    y_ = tf.placeholder(tf.float32, [None, 5270])

    # Build the graph for the deep net
    
    # First convolutional layer
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    # New size of images is 90x90

    # Second convolutional layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    # New size of images is 45x45

    # Third convolutional layer
    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    # New size of images is 23x23
    h_pool3_flat = tf.reshape(h_pool3, [-1, 23*23*128])

    # Fully connected layer 1
    W_fc1 = weight_variable([23*23*128, 8192])
    b_fc1 = bias_variable([8192])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # Dropout layer
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Output layer
    W_fc2 = weight_variable([8192, 5270])
    b_fc2 = bias_variable([5270])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # Optimizer + Loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    print('Starting training...')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    for step in range(1, num_iterations):
        # Get new batch
        tr_batch_x, tr_batch_y = get_next_training_batch(tr_batch_size, category_to_int)

        if step % 100 == 0:
            train_d = {x: tr_batch_x, y_: tr_batch_y, keep_prob: 1.0}
            train_acc = accuracy.eval(feed_dict=train_d)
            print('step {}, training accuracy {}'.format(step, train_acc))

            # Get validation batch and print accuracy
            val_batch_x, val_batch_y = get_next_validation_batch(val_batch_size, category_to_int)
            val_d = {x: val_batch_x, y_: val_batch_y, keep_prob: 1.0}
            val_acc = accuracy.eval(feed_dict=val_d)
            print('step {}, validation accuracy {} \n'.format(step, val_acc))

        train_d = {x: tr_batch_x, y_: tr_batch_y, keep_prob: 0.5}
        train_step.run(feed_dict=train_d)

    val_batch_x, val_batch_y = get_next_batch(4*val_batch_size, category_to_int)
    train_d = {x: val_batch_x, y_: val_batch_y, keep_prob: 1.0}
    final_accuracy = accuracy.eval(feed_dict=train_d)
    print('FINAL training accuracy {}'.format(final_accuracy))

