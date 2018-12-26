#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import os
import sys
import time

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

import h5py
import numpy as np

# Basic model parameters as external flags.
FLAGS = None

def placeholder_inputs(batch_size):
    """
    Generate placeholder variables to represent the input tensors.

    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.

    Args:
        batch_size: The batch size will be baked into both placeholders.

    Returns:
        images_placeholder: Images placeholder.
        labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train of test data sets.
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder

def fill_feed_dict(data_set, images_pl, labels_pl):
    """
    Fills the feed_dict for training the given step.

    A feed_dict takes the form of:
    feed_dict  {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }

    Args:
        data_set: The set of images and labels, from input_data.read_data_sets()
        images_pl: The images placeholder, from placeholder_inputs().
        lables_pl: The labels placeholder, from placeholder_inputs().

    Returns:
        feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # 'batch size' examples.
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed
    }
    return feed_dict

def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set):
    """
    Run one evaluation against the full epoch of data.

    Args:
        sess: The session in which the model has been trained.
        eval_correct: The Tensor that returns the number of correct predictions.
        images_placeholder: The images placeholder.
        labels_placeholder: The labels placeholder.
        data_set: The set of images and labels to evaluate, from input_data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for _ in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('Num examples: %d, Num correct: %d, Precision @ 1: %0.04f' % (num_examples, true_count, precision))

def run_training():
    """ Train MNIST for a number of steps. """
    # Get the sets of images and labels for training, validation, and test on MNIST.
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

    # Tell Tensorflow that the model will be built into te default Graph.
    with tf.Graph().as_default():
        # Generate placeholder for the images and labels.
        images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

        # Build a Graph that computes predictions from the inference model.
        logits = mnist.inference(images_placeholder, FLAGS.hidden1, FLAGS.hidden2)

        # Add to the Graph the Ops for loss calculation.
        loss = mnist.loss(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = mnist.training(loss, FLAGS.learning_rate)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = mnist.evaluation(logits, labels_placeholder)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        # And then after everything is built:

        # Run the Op to initialize the variables.
        sess.run(init)

        # Start the training loop.
        for step in range(FLAGS.max_steps):
            start_time = time.time()

            # Fill a feed dictionary with the actual set of images nad labesl
            # for this particular training step.
            feed_dict = fill_feed_dict(data_sets.train, images_placeholder, labels_placeholder)

            # Run one step of the model. The return values are the activations
            # from the 'train_op' (which is discarded) and the 'loss' Op. To
            # inspect the values of your Ops or variables, you may include then in the list
            # passed to sess.run() and the value tensors will be returned in the tuple from the call.
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step+1) % 1000 == 0 or (step+1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.train)
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.validation)
                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.test)

def raw_dataset():
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)
    hdf5 = h5py.File('mnist_raw.hdf5', 'w')
    grp_train = hdf5.create_group('train')
    grp_valid = hdf5.create_group('validation')
    ft = h5py.File('test.hdf5', 'w')
    grp_test = ft.create_group('test')

    train_input = None
    train_output = None

    for _ in range(550):
        images_feed, labels_feed = data_sets.train.next_batch(FLAGS.batch_size, FLAGS.fake_data)
        train_input = np.append(train_input, images_feed, axis=0) if train_input is not None else np.array(images_feed)
        train_output = np.append(train_output, labels_feed) if train_output is not None else np.array(labels_feed)

    valid_input = None
    valid_output = None

    for _ in range(50):
        images_feed, labels_feed = data_sets.validation.next_batch(FLAGS.batch_size, FLAGS.fake_data)
        valid_input = np.append(valid_input, images_feed, axis=0) if valid_input is not None else np.array(images_feed)
        valid_output = np.append(valid_output, labels_feed) if valid_output is not None else np.array(labels_feed)

    grp_train.create_dataset('input', data=train_input)
    grp_train.create_dataset('output', data=train_output)
    grp_valid.create_dataset('input', data=valid_input)
    grp_valid.create_dataset('output', data=valid_output)
    
    grp_test.create_dataset('input', data=data_sets.test.images)
    grp_test.create_dataset('output', data=data_sets.test.labels)

def split_dataset():
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)
    f0 = h5py.File('mnist0.hdf5', 'w')
    f1 = h5py.File('mnist1.hdf5', 'w')
    f2 = h5py.File('mnist2.hdf5', 'w')
    ft = h5py.File('test.hdf5', 'w')
    grp0_train = f0.create_group('train')
    grp0_validation = f0.create_group('validation')
    grp1_train = f1.create_group('train')
    grp1_validation = f1.create_group('validation')
    grp2_train = f2.create_group('train')
    grp2_validation = f2.create_group('validation')
    grp_test = ft.create_group('test')

    toggle = 0
    flag = False

    for _ in range(550):
        images_feed, labels_feed = data_sets.train.next_batch(FLAGS.batch_size, FLAGS.fake_data)

        if flag:
            if toggle == 0:
                train_input0 = np.append(train_input0, images_feed, axis=0)
                train_output0 = np.append(train_output0, labels_feed)
            elif toggle == 1:
                train_input1 = np.append(train_input1, images_feed, axis=0)
                train_output1 = np.append(train_output1, labels_feed)
            elif toggle == 2:
                train_input2 = np.append(train_input2, images_feed, axis=0)
                train_output2 = np.append(train_output2, labels_feed)
        else:
            if toggle == 0:
                train_input0 = np.array(images_feed)
                train_output0 = np.array(labels_feed)
            elif toggle == 1:
                train_input1 = np.array(images_feed)
                train_output1 = np.array(labels_feed)
            elif toggle == 2:
                train_input2 = np.array(images_feed)
                train_output2 = np.array(labels_feed)
                flag = True
        
        toggle = (toggle + 1) % 3

    toggle = 0
    flag = False

    for _ in range(50):
        images_feed, labels_feed = data_sets.validation.next_batch(FLAGS.batch_size, FLAGS.fake_data)

        if flag:
            if toggle == 0:
                validation_input0 = np.append(validation_input0, images_feed, axis=0)
                validation_output0 = np.append(validation_output0, labels_feed)
            elif toggle == 1:
                validation_input1 = np.append(validation_input1, images_feed, axis=0)
                validation_output1 = np.append(validation_output1, labels_feed)
            elif toggle == 2:
                validation_input2 = np.append(validation_input2, images_feed, axis=0)
                validation_output2 = np.append(validation_output2, labels_feed)
        else:
            if toggle == 0:
                validation_input0 = np.array(images_feed)
                validation_output0 = np.array(labels_feed)
            elif toggle == 1:
                validation_input1 = np.array(images_feed)
                validation_output1 = np.array(labels_feed)
            elif toggle == 2:
                validation_input2 = np.array(images_feed)
                validation_output2 = np.array(labels_feed)
                flag = True

        toggle = (toggle + 1) % 3

    grp0_train.create_dataset('input', data=train_input0)
    grp0_train.create_dataset('output', data=train_output0)
    grp1_train.create_dataset('input', data=train_input1)
    grp1_train.create_dataset('output', data=train_output1)
    grp2_train.create_dataset('input', data=train_input2)
    grp2_train.create_dataset('output', data=train_output2)
    grp0_validation.create_dataset('input', data=validation_input0)
    grp0_validation.create_dataset('output', data=validation_output0)
    grp1_validation.create_dataset('input', data=validation_input1)
    grp1_validation.create_dataset('output', data=validation_output1)
    grp2_validation.create_dataset('input', data=validation_input2)
    grp2_validation.create_dataset('output', data=validation_output2)
    grp_test.create_dataset('input', data=data_sets.test.images)
    grp_test.create_dataset('output', data=data_sets.test.labels)

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    if FLAGS.split:
        split_dataset()
    else:
        raw_dataset()
    run_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=2000,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--hidden1',
        type=int,
        default=128,
        help='Number of units in hidden layer 1.'
    )
    parser.add_argument(
        '--hidden2',
        type=int,
        default=32,
        help='Number of units in hidden layer 2.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size. Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default='data',
        #default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow/mnist/input_data'),
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='log',
        #default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow/mnist/logs/fully_connected_feed'),
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--fake_data',
        default=False,
        help='If true, uses fake data for unit testing.',
        action='store_true'
    )
    parser.add_argument(
        '--split',
        type=bool,
        default=False,
        help='If true, MNIST data will be splitted into 3 groups.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)