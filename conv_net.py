import tensorflow as tf
from get_next_batch import get_next_training_batch, get_next_validation_batch

def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def max_pool_4x4(x):
    """max_pool_4x4 downsamples a feature map by 4X."""
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],strides=[1, 4, 4, 1], padding='SAME')


def weight_variable(shape, name):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv_net(category_to_int, int_to_category):
    num_iterations = 20000
    tr_batch_size = 128
    val_batch_size = 128
    num_prints = 250
    num_saves = 500

    x = tf.placeholder(tf.float32, [None, 180*180*3], name='x')
    x_image = tf.reshape(x, [-1, 180, 180, 3])
    y_ = tf.placeholder(tf.float32, [None, 5270], name='y_')

    # First convolutional layer
    W_conv1 = weight_variable([2, 2, 3, 32], 'W_conv1')
    b_conv1 = bias_variable([32], 'b_conv1')
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_4x4(h_conv1)
    # New size of images is 45x45

    # Second convolutional layer
    W_conv2 = weight_variable([4, 4, 32, 64], 'W_conv2')
    b_conv2 = bias_variable([64], 'b_conv2')
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    # New size of images is 23x23

    # Third convolutional layer
    W_conv3 = weight_variable([6, 6, 64, 128], 'W_conv3')
    b_conv3 = bias_variable([128], 'b_conv3')
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    # New size of images is 12x12
    h_pool3_flat = tf.reshape(h_pool3, [-1, 12*12*128])

    # Fully connected layer 1
    W_fc1 = weight_variable([12*12*128, 2048], 'W_fc1')
    b_fc1 = bias_variable([2048], 'b_fc1')
    h_fc1 = tf.nn.softmax(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # Dropout layer
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Output layer
    W_fc2 = weight_variable([2048, 5270], 'W_fc2')
    b_fc2 = bias_variable([5270], 'b_fc2')
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='output_layer')

    # Optimizer + Loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)
    learning_r = tf.placeholder(tf.float32, shape=[], name='learning_rate')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_r)
    train_step = optimizer.minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(prediction)

    saver = tf.train.Saver()

    print('Starting training...')
    sess = tf.Session()
    print('Session obtained')
    sess.run(tf.global_variables_initializer())
    print('variables initialized')

    with sess.as_default():
        for step in range(1, num_iterations+1):
            # Get new batch
            tr_batch_x, tr_batch_y = get_next_training_batch(tr_batch_size, category_to_int)
            # PRINT RESULT
            if step % num_prints == 0:
                train_d = {x: tr_batch_x, y_: tr_batch_y, keep_prob: 1.0}
                train_acc = accuracy.eval(feed_dict=train_d)

                # Get validation batch and print accuracy
                val_batch_x, val_batch_y = get_next_validation_batch(val_batch_size, category_to_int)
                val_d = {x: val_batch_x, y_: val_batch_y, keep_prob: 1.0}
                val_acc = accuracy.eval(feed_dict=val_d)
                print('step {}, training accuracy {}, validation accuracy {}'.format(step, train_acc, val_acc))

            # TRAIN MODEL
            train_d = {x: tr_batch_x, y_: tr_batch_y, learning_r:0.5, keep_prob: 0.5}
            sess.run(train_step, feed_dict=train_d)

            # Save model
            if step % num_saves == 0:
                saver.save(sess, '../models/model')
                print('Model saved')


        # Save final model
        saver.save(sess, '../models/model_final')
        val_batch_x, val_batch_y = get_next_validation_batch(4*val_batch_size, category_to_int)
        train_d = {x: val_batch_x, y_: val_batch_y, keep_prob: 1.0}
        final_accuracy = accuracy.eval(feed_dict=train_d)
        print('FINAL validation accuracy {}'.format(final_accuracy))

