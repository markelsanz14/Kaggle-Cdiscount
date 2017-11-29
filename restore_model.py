import tensorflow as tf
from get_next_batch import get_next_training_batch, get_next_validation_batch

def restore_model(category_to_int, int_to_category):
    first_iteration = 200
    last_iteration = 300
    tr_batch_size = 8
    val_batch_size = 32
    num_prints = 50
    num_saves = 100

    print('Starting training...')
    
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('../models/model_final.meta')
        saver.restore(sess, tf.train.latest_checkpoint('../models/'))
        
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('x:0')
        y_ = graph.get_tensor_by_name('y_:0')
        W_conv1 = graph.get_tensor_by_name('W_conv1:0')
        b_conv1 = graph.get_tensor_by_name('b_conv1:0')
        W_conv2 = graph.get_tensor_by_name('W_conv2:0')
        b_conv2 = graph.get_tensor_by_name('b_conv2:0')
        W_fc1 = graph.get_tensor_by_name('W_fc1:0')
        b_fc1 = graph.get_tensor_by_name('b_fc1:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        W_fc2 = graph.get_tensor_by_name('W_fc2:0')
        b_fc2 = graph.get_tensor_by_name('b_fc2:0')
        y_conv = graph.get_tensor_by_name('output_layer:0')
    
        # Optimizer + Loss
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv)
        loss = tf.reduce_mean(cross_entropy)
        train_step = tf.get_collection('optimizer:0')

        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(prediction)
    

        tr_batch_x, tr_batch_y = get_next_training_batch(tr_batch_size, category_to_int)
        train_d = {x:tr_batch_x, y_:tr_batch_y, keep_prob:1.0}
        res = loss.eval(feed_dict=train_d)
        for step in range(first_iteration+1, last_iteration+1):
            # Get new batch
            tr_batch_x, tr_batch_y = get_next_training_batch(tr_batch_size, category_to_int)
            if step % num_prints == 0:
                train_d = {x: tr_batch_x, y_: tr_batch_y, keep_prob: 1.0}
                train_acc = accuracy.eval(feed_dict=train_d)

                # Get validation batch and print accuracy
                val_batch_x, val_batch_y = get_next_validation_batch(val_batch_size, category_to_int)
                val_d = {x: val_batch_x, y_: val_batch_y, keep_prob: 1.0}
                val_acc = accuracy.eval(feed_dict=val_d)
                print('step {}, training accuracy {}, validation accuracy {}'.format(step, train_acc, val_acc))

            train_d = {x: tr_batch_x, y_: tr_batch_y, keep_prob: 0.5}
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

