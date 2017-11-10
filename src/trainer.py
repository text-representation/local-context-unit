"""
Author	:

Date	:

Brief	: 
"""

import sys
import os.path
import time
import random
import numpy as np
import numpy.random
import tensorflow as tf

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True

base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
lib_dir = os.path.abspath(os.path.join(base_dir, 'lib'))
sys.path.append(lib_dir)
from tedll.utils import config
from tedll.utils import data_utils

import region_emb_classify

class Trainer(object):
    """Trainer"""
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.train_op = self.make_train_op()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=500)
    
    def load_saver_and_predict(self):
        """load_saver_and_predict"""
        checkpoint_dir = os.path.join(base_dir, self.config.trainer.checkpoint_dir)
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        
        print 'checkpoint file is ', checkpoint_file
        with tf.Session(config=config_tf) as sess:
            self.saver.restore(sess, checkpoint_file)
            acc, loss = self.evaluate_on_data(sess,\
                    self._sequence_test, self._label_test)
            print >> sys.stderr, 'acc is %f ' % (acc)
    
    def load_data(self):
        """load_data"""
        indexes = [0, 1]
        lengths = [1, self.config.model.max_sequence_length]

        print >> sys.stderr, "Load label data ..."
        data_path = os.path.join(base_dir, self.config.trainer.label_data)
        labels, sequence_train = \
                data_utils.read_data(data_path, indexes, lengths)

        print >> sys.stderr, "Load dev data ..."

        data_path = os.path.join(base_dir, self.config.trainer.dev_data)
        self._label_dev, self._sequence_dev = \
                data_utils.read_data(data_path, indexes, lengths)

        data_path = os.path.join(base_dir, self.config.trainer.test_data)
        print >> sys.stderr, "Load test data ..."
        
        self._label_test, self._sequence_test = \
                data_utils.read_data(data_path, indexes, lengths)

        self._train_batches = data_utils.batch_iter(list(zip(sequence_train, labels)), \
                self.config.trainer.batch_size, self.config.trainer.max_epoch)

    def evaluate_on_data(self, sess, sequence, labels):
        """evaluate_on_data
        Args:
            sess(type):
            sequence(type):
            labels:
        Returns:
            type:
        """
        logits = []
        loss = 0
        batch_size = self.config.trainer.batch_size
        for i in xrange(len(sequence) / batch_size):
            t_sequence = sequence[i * batch_size: i * batch_size + batch_size]
            t_label = labels[i * batch_size: i * batch_size + batch_size]
            feed_dict = {self.model.sequence: t_sequence, self.model.label: t_label}
            t_logits, t_loss = sess.run([self.model.logits_op, self.model.loss_op], \
                    feed_dict=feed_dict)
            loss += t_loss
            logits.extend(t_logits)
        loss /= len(sequence)
        pred = map(np.argmax, logits)
        true = map(lambda x:x[0], labels)
        res = []
        res.append(self.evaluate(pred, true))
        res.append(loss)
        return res

    def evaluate(self, pred, true):
        """evaluate

        Args:
            pred(list): predict result
            true(list): real result
        Returns:
            4-element Tuple: .
        """
        TP = len(filter(lambda (p, t): p == t, zip(pred, true))) * np.float(1.0)
        acc = TP / len(pred)
        return acc

    def make_train_op(self):
        """make_train_op"""
        adam_optimizer = tf.train.AdamOptimizer(learning_rate=self.config.trainer.learning_rate)
        tvars = tf.trainable_variables()
        train_op = adam_optimizer.apply_gradients(\
                    adam_optimizer.compute_gradients(self.model.loss_op, tvars),
                        global_step=self.global_step)
        return train_op

    def train(self):
        """train"""
        current_epoch = 0
        best_dev_acc = 0
        best_test_acc = 0
        best_epoch = 0

        checkpoint_dir = os.path.join(base_dir, self.config.trainer.checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
 
        with tf.Session(config=config_tf) as sess:
            sess.run(tf.global_variables_initializer())
            while True:
                epoch, epoch_percent, batch_slots  = self._train_batches.next()
                batch_sequence, batch_label = zip(*batch_slots)
                if epoch >= self.config.trainer.max_epoch:
                    print >> sys.stderr, "Training Done!"
                    exit(0)
                
                feed_dict = {
                        self.model.sequence: batch_sequence,
                        self.model.label: batch_label}

                fetch_dict = [self.train_op,
                        self.model.loss_op,
                        self.global_step]

                _, batch_loss, global_step = \
                        sess.run(fetch_dict, feed_dict=feed_dict)

                if global_step % 10 == 1:
                    output_format = 'epoch:{0}[{1:.2f}%] batch_loss:{2}| global_step:{3}'
                    output = [epoch, epoch_percent, \
                            batch_loss / self.config.trainer.batch_size, global_step]
                    print >> sys.stderr, output_format.format(*output)


                # End of an epoch
                if current_epoch < epoch:
                    #self.saver.save(sess, os.path.join(checkpoint_dir, 'model.cpkt'),\
                    #        global_step=global_step)
                    # Test on dev
                    feed_dict = {self.model.sequence: self._sequence_dev,
                            self.model.label: self._label_dev}
                    dev_acc, loss = \
                            self.evaluate_on_data(sess, self._sequence_dev, self._label_dev)
                    format_str = 'epoch {0} finished. Performance on {1}:'\
                            ' [acc:{2}, loss:{3}]'
                    output = [current_epoch, 'Dev', dev_acc, loss]
                    print >> sys.stderr, format_str.format(*output)
                    test_acc, loss = \
                            self.evaluate_on_data(sess, self._sequence_test, self._label_test)
                    if dev_acc > best_dev_acc:
                        best_dev_acc = dev_acc
                        best_epoch = current_epoch
                        best_test_acc = test_acc

                    output = [current_epoch, 'Test', test_acc, loss]
                    print >> sys.stderr, format_str.format(*output)
                    print >> sys.stderr, 'Best Epoch: %d---Best test acc: %f---Best dev acc: %f---'\
                            % (best_epoch, best_test_acc, best_dev_acc)

                    current_epoch = epoch


def main():
    """main"""
    config_path = sys.argv[1]
    config_file = os.path.join(base_dir, config_path)
    main_config = config.Config(config_file=config_file)

    model = region_emb_classify.RegionEmbeddingClassify(main_config.model)
    trainer = Trainer(model, main_config)
    if main_config.model.predict is False:
        trainer.load_data()
        trainer.train()
    else:
        trainer.load_test()
        trainer.load_saver_and_predict()

if '__main__' == __name__:
    main()

