import json
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import cPickle as pickle
import os
from datetime import datetime, timedelta
import pandas as pd




class CNN():

    def __init__(self, sess):
        
        self.sess = sess

        # self.train_size = 6445
        # self.val_size = 1693
        self.epochs = 50
        self.batch_size = 1

        self.l2_lambda = 1e-4
        self.lr = 1e-4

        self.d = 100
        self.l = 3

        self.max_mid_term = 30
        self.max_long_term = 100


        self.checkpoint_dir = 'checkpoints/'
        self.build_model()

    def build_model(self):
        # self.long_term_events_count = tf.placeholder(tf.int32, name='long_term_events_count')
        # self.mid_term_events_count = tf.placeholder(tf.int32, name='mid_term_events_count')
        # self.short_term_events_count = tf.placeholder(tf.int32, name='short_term_events_count')

        # self.long_term_events = tf.placeholder(tf.float32, [self.batch_size, self.long_term_events_count.eval(), self.d], name='long_term_events')
        # self.mid_term_events = tf.placeholder(tf.float32, [self.batch_size, self.mid_term_events_count.eval(), self.d], name='mid_term_events')
        # self.short_term_events = tf.placeholder(tf.float32, [self.batch_size, self.short_term_events_count.eval(), self.d], name='short_term_events')

        self.long_term_events = tf.placeholder(tf.float32, [self.batch_size, self.max_long_term, self.d], name='long_term_events')
        self.mid_term_events = tf.placeholder(tf.float32, [self.batch_size, self.max_mid_term, self.d], name='mid_term_events')
        self.short_term_events = tf.placeholder(tf.float32, [self.batch_size, None, self.d], name='short_term_events')

        self.labels = tf.placeholder(tf.float32, [self.batch_size], name='labels')


        self.long_term_conv = self.conv1d(self.long_term_events, output_dim=self.d, name='long_term_conv')
        self.mid_term_conv = self.conv1d(self.mid_term_events, output_dim=self.d, name='mid_term_conv')
        # print self.long_term_conv.shape


        # self.long_term_pool = tf.reshape(tf.nn.max_pool(tf.expand_dims(self.long_term_conv, -1), [1, tf.shape(self.long_term_conv)[1], 1, 1], [1, 1, 1, 1], 'VALID', name='long_term_pool'), [self.batch_size, self.d])
        # self.mid_term_pool = tf.reshape(tf.nn.max_pool(tf.expand_dims(self.mid_term_conv, -1), [1, tf.shape(self.mid_term_conv)[1], 1, 1], [1, 1, 1, 1], 'VALID', name='mid_term_pool'), [self.batch_size, self.d])
        # self.short_term_pool = tf.reshape(tf.nn.avg_pool(tf.expand_dims(self.short_term_events, -1), [1, tf.shape(self.short_term_events)[1], 1, 1], [1, 1, 1, 1], 'VALID', name='short_term_pool'), [self.batch_size, self.d])
        # print self.long_term_pool.shape

        self.long_term_pool = tf.reduce_max(self.long_term_conv, axis=2)
        self.mid_term_pool = tf.reduce_max(self.mid_term_conv, axis=2)
        self.short_term_pool = tf.reduce_mean(self.short_term_events, axis=1)

        self.v = tf.concat([self.long_term_pool, self.mid_term_pool, self.short_term_pool], axis=1)
        print self.v.shape

        self.y = tf.layers.dense(self.v, 100, activation=tf.nn.sigmoid)
        # print self.y.shape

        self.logits = tf.reshape(tf.layers.dense(self.y, 1, activation=tf.nn.sigmoid), [self.batch_size], name='preds')
        # print self.logits.shape

        # print tf.trainable_variables()

        self.ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
        self.l2_loss = sum([tf.reduce_sum(tf.square(var)) for var in tf.trainable_variables()])

        self.loss = self.ce_loss + (self.l2_lambda * self.l2_loss)

        self.saver = tf.train.Saver()


    # THIS ACTUALLY MAKES NO FREAKING SENSE AT ALL
    def conv1d(self, input, output_dim=1, k_w=3, d_w=1, stddev=0.02, name="conv1d"):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [k_w, input.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv1d(input, w, stride=d_w, padding='SAME')

            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

            return conv


    def train(self, data, val_data):
        cnn_optim = tf.train.AdamOptimizer(self.lr).minimize(self.loss, var_list=tf.trainable_variables())

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        counter = 0
        start_time = time.time()

        losses = []
        ce_losses = []
        l2_losses = []
        tps = []
        fps = []
        tns = []
        fns = []

        for epoch in xrange(self.epochs):
            length = len(data)
            batch_idxs = length // self.batch_size

            batch_loss = 0
            batch_ce_loss = 0
            batch_l2_loss = 0
            batch_per_error = 0
            batch_tp = 0
            batch_fp = 0
            batch_tn = 0
            batch_fn = 0

            for idx in xrange(0, batch_idxs):
                batch_data = data[idx*self.batch_size:(idx+1)*self.batch_size]

                feed_dict = {   
                                # self.short_term_events_count: len(batch_data[0][0]),
                                # self.mid_term_events_count: len(batch_data[0][1]),
                                # self.long_term_events_count: len(batch_data[0][2]),
                                self.short_term_events: [batch_data[0][0]],
                                self.mid_term_events: [batch_data[0][1]],
                                self.long_term_events: [batch_data[0][2]],
                                self.labels: batch_data[0][3]
                            }
                # if idx == 0:
                #     print feed_dict[self.corrupt]
                vlist = tf.trainable_variables()
                vnames = [v.name for v in vlist]
                # print vnames
                __ = self.sess.run([cnn_optim, self.logits] + vlist, feed_dict=feed_dict)

                _ = __[0]
                logits = __[1]
                label = batch_data[0][3]

                ce_loss = self.ce_loss.eval(feed_dict)
                l2_loss = self.l2_loss.eval(feed_dict)
                loss = self.loss.eval(feed_dict)

                # print loss, u_loss, l2_loss
                batch_loss += loss
                batch_ce_loss += ce_loss
                batch_l2_loss += l2_loss
                if label[0] == 1 and logits > .5:
                    batch_tp += 1.0
                if label[0] == 0 and logits > .5:
                    batch_fp += 1.0
                if label[0] == 0 and logits <= .5:
                    batch_tn += 1.0
                if label[0] == 1 and logits <= .5:
                    batch_fn += 1.0
                
                counter += 1
                print('Train Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, ce_loss: %.8f, l2_loss: %.8f' \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, loss, ce_loss, l2_loss))

                if np.mod(counter, 1000) == 1:
                    print logits, feed_dict[self.labels]

            batch_loss /= batch_idxs
            batch_ce_loss /= batch_idxs
            batch_l2_loss /= batch_idxs
            losses += [[batch_loss[0]]]
            ce_losses += [[batch_ce_loss[0]]]
            l2_losses += [[self.l2_lambda * batch_l2_loss]]
            tps += [[batch_tp]]
            fps += [[batch_fp]]
            tns += [[batch_tn]]
            fns += [[batch_fn]]
            # print losses

# -------------------VALIDATION----------------------

            # val_length = min(len(val_data), self.val_size)
            val_length = len(val_data)
            val_batch_idxs = val_length // self.batch_size

            batch_loss = 0
            batch_ce_loss = 0
            batch_l2_loss = 0
            batch_per_error = 0
            batch_tp = 0
            batch_fp = 0
            batch_tn = 0
            batch_fn = 0

            for idx in xrange(0, val_batch_idxs):
                batch_data = val_data[idx*self.batch_size:(idx+1)*self.batch_size]

                feed_dict = {
                                self.short_term_events: [batch_data[0][0]],
                                self.mid_term_events: [batch_data[0][1]],
                                self.long_term_events: [batch_data[0][2]],
                                self.labels: batch_data[0][3]
                            }
                ce_loss = self.ce_loss.eval(feed_dict)
                l2_loss = self.l2_loss.eval(feed_dict)
                loss = self.loss.eval(feed_dict)
                logits = self.logits.eval(feed_dict)

                label = batch_data[0][3]

                # print loss, u_loss, l2_loss
                batch_loss += loss
                batch_ce_loss += ce_loss
                batch_l2_loss += l2_loss
                if label[0] == 1 and logits > .5:
                    batch_tp += 1.0
                if label[0] == 0 and logits > .5:
                    batch_fp += 1.0
                if label[0] == 0 and logits <= .5:
                    batch_tn += 1.0
                if label[0] == 1 and logits <= .5:
                    batch_fn += 1.0

                print('Val Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, ce_loss: %.8f, l2_loss: %.8f' \
                    % (epoch, idx, val_batch_idxs,
                        time.time() - start_time, loss, ce_loss, l2_loss))

            batch_loss /= val_batch_idxs
            batch_ce_loss /= val_batch_idxs
            batch_l2_loss /= val_batch_idxs
            losses[-1] += [batch_loss[0]]
            ce_losses[-1] += [batch_ce_loss[0]]
            l2_losses[-1] += [self.l2_lambda * batch_l2_loss]
            tps[-1] += [batch_tp]
            fps[-1] += [batch_fp]
            tns[-1] += [batch_tn]
            fns[-1] += [batch_fn]

            losses_arr = np.array(losses)
            ce_losses_arr = np.array(ce_losses)
            l2_losses_arr = np.array(l2_losses)

            tps_arr = np.array(tps)
            fps_arr = np.array(fps)
            tns_arr = np.array(tns)
            fns_arr = np.array(fns)

            # print losses_arr.shape, losses_arr[:,0].shape, losses_arr[:,0]

            plt.figure(figsize=(16,10))
            plt.plot([i+1 for i in range(len(losses_arr))], losses_arr[:,0], label='total loss', color='b')
            plt.plot([i+1 for i in range(len(ce_losses_arr))], ce_losses_arr[:,0], label='ce loss', color='g')
            plt.plot([i+1 for i in range(len(l2_losses_arr))], l2_losses_arr[:,0], label='l2 loss', color='r')
            plt.legend()
            plt.savefig('trn_losses.png')

            plt.figure(figsize=(16,10))
            plt.plot([i+1 for i in range(len(tps_arr))], tps_arr[:,0], label='tp', color='b')
            plt.plot([i+1 for i in range(len(fps_arr))], fps_arr[:,0], label='fp', color='g')
            plt.plot([i+1 for i in range(len(tns_arr))], tns_arr[:,0], label='tn', color='r')
            plt.plot([i+1 for i in range(len(fns_arr))], fns_arr[:,0], label='fn', color='c')
            plt.legend()
            plt.savefig('trn_pns.png')

            plt.figure(figsize=(16,10))
            plt.plot([i+1 for i in range(len(tps_arr))], np.divide((tps_arr+tns_arr)[:,0], (tps_arr+tns_arr+fps_arr+fns_arr)[:,0]), label='acc', color='b')
            plt.plot([i+1 for i in range(len(tps_arr))], np.divide(tps_arr[:,0], (tps_arr+fps_arr)[:,0]), label='precision', color='g')
            plt.plot([i+1 for i in range(len(tps_arr))], np.divide(tps_arr[:,0], (tps_arr+fns_arr)[:,0]), label='recall', color='r')
            plt.legend()
            plt.savefig('trn_acc.png')


            plt.figure(figsize=(16,10))
            plt.plot([i+1 for i in range(len(losses_arr))], losses_arr[:,1], label='total loss', color='b')
            plt.plot([i+1 for i in range(len(ce_losses_arr))], ce_losses_arr[:,1], label='ce loss', color='g')
            plt.plot([i+1 for i in range(len(l2_losses_arr))], l2_losses_arr[:,1], label='l2 loss', color='r')
            plt.legend()
            plt.savefig('val_losses.png')

            plt.figure(figsize=(16,10))
            plt.plot([i+1 for i in range(len(tps_arr))], tps_arr[:,1], label='tp', color='b')
            plt.plot([i+1 for i in range(len(fps_arr))], fps_arr[:,1], label='fp', color='g')
            plt.plot([i+1 for i in range(len(tns_arr))], tns_arr[:,1], label='tn', color='r')
            plt.plot([i+1 for i in range(len(fns_arr))], fns_arr[:,1], label='fn', color='c')
            plt.legend()
            plt.savefig('val_pns.png')

            plt.figure(figsize=(16,10))
            plt.plot([i+1 for i in range(len(tps_arr))], np.divide((tps_arr+tns_arr)[:,1], (tps_arr+tns_arr+fps_arr+fns_arr)[:,1]), label='acc', color='b')
            plt.plot([i+1 for i in range(len(tps_arr))], np.divide(tps_arr[:,1], (tps_arr+fps_arr)[:,1]), label='precision', color='g')
            plt.plot([i+1 for i in range(len(tps_arr))], np.divide(tps_arr[:,1], (tps_arr+fns_arr)[:,1]), label='recall', color='r')
            plt.legend()
            plt.savefig('val_acc.png')

            plt.figure(figsize=(16,10))
            plt.plot([i+1 for i in range(len(losses_arr))], losses_arr[:,0], label='train total loss', color='b')
            plt.plot([i+1 for i in range(len(ce_losses_arr))], ce_losses_arr[:,0], label='train ce loss', color='g')
            plt.plot([i+1 for i in range(len(l2_losses_arr))], l2_losses_arr[:,0], label='train l2 loss', color='r')
            plt.plot([i+1 for i in range(len(losses_arr))], losses_arr[:,1], label='val total loss', linestyle='--', color='b')
            plt.plot([i+1 for i in range(len(ce_losses_arr))], ce_losses_arr[:,1], label='val ce loss', linestyle='--', color='g')
            plt.plot([i+1 for i in range(len(l2_losses_arr))], l2_losses_arr[:,1], label='val l2 loss', linestyle='--', color='r')
            plt.legend()
            plt.savefig('losses.png')

            plt.figure(figsize=(16,10))
            plt.plot([i+1 for i in range(len(tps_arr))], tps_arr[:,0], label='train tp', color='b')
            plt.plot([i+1 for i in range(len(fps_arr))], fps_arr[:,0], label='train fp', color='g')
            plt.plot([i+1 for i in range(len(tns_arr))], tns_arr[:,0], label='train tn', color='r')
            plt.plot([i+1 for i in range(len(fns_arr))], fns_arr[:,0], label='train fn', color='c')
            plt.plot([i+1 for i in range(len(tps_arr))], tps_arr[:,1], label='val tp', linestyle='--', color='b')
            plt.plot([i+1 for i in range(len(fps_arr))], fps_arr[:,1], label='val fp', linestyle='--', color='g')
            plt.plot([i+1 for i in range(len(tns_arr))], tns_arr[:,1], label='val tn', linestyle='--', color='r')
            plt.plot([i+1 for i in range(len(fns_arr))], fns_arr[:,1], label='val fn', linestyle='--', color='c')
            plt.legend()
            plt.savefig('pns.png')

            plt.figure(figsize=(16,10))
            plt.plot([i+1 for i in range(len(tps_arr))], np.divide((tps_arr+tns_arr)[:,0], (tps_arr+tns_arr+fps_arr+fns_arr)[:,0]), label='train acc', color='b')
            plt.plot([i+1 for i in range(len(tps_arr))], np.divide(tps_arr[:,0], (tps_arr+fps_arr)[:,0]), label='train precision', color='g')
            plt.plot([i+1 for i in range(len(tps_arr))], np.divide(tps_arr[:,0], (tps_arr+fns_arr)[:,0]), label='train recall', color='r')
            plt.plot([i+1 for i in range(len(tps_arr))], np.divide((tps_arr+tns_arr)[:,1], (tps_arr+tns_arr+fps_arr+fns_arr)[:,1]), label='val acc', linestyle='--', color='b')
            plt.plot([i+1 for i in range(len(tps_arr))], np.divide(tps_arr[:,1], (tps_arr+fps_arr)[:,1]), label='val precision', linestyle='--', color='g')
            plt.plot([i+1 for i in range(len(tps_arr))], np.divide(tps_arr[:,1], (tps_arr+fns_arr)[:,1]), label='val recall', linestyle='--', color='r')
            plt.legend()
            plt.savefig('acc.png')



def main(_):

    with open('1000epochs3/run_info.p', 'r') as f:
        run_info = pickle.load(f)

    # ticks = ['AAPL', 'GOOGL', 'IBM', 'AMZN', 'CSCO', 'NVDA', 'MSFT', 'QCOM', 'INTC', 'AMD']
    ticks = ['AAPL', 'GOOGL', 'IBM', 'AMZN', 'CSCO', 'MSFT', 'INTC']

    # min_date = datetime.strptime('20121109', '%Y%m%d')
    # val_date = datetime.strptime('20171016', '%Y%m%d')
    min_date = datetime.strptime('20121109', '%Y%m%d')
    val_date = datetime.strptime('20171001', '%Y%m%d')
    max_date = datetime.strptime('20171016', '%Y%m%d')

    # delta = max_date - min_date



    data = {}
    tick_data = {}


    for t in ticks:
        tick_data[t] = np.loadtxt('stock_data/{}_2006-01-01_to_2017-11-01.csv'.format(t), skiprows=1, delimiter=',', dtype=object,
                        converters={1: lambda x: datetime.strptime(x, "%d-%b-%y")})


    # unique_dates = set()

    for k in run_info['info2index']:
        d = datetime.strptime(k[1], '%Y%m%d')
        # unique_dates.add((k[0], d))
        for i in range(31):
            date = d + timedelta(days=i)
            if date < max_date and (k[0], date) not in data:
                data[(k[0], date)] = [[] for _ in range(3)] + [-1]
        if d < max_date:
            data[(k[0], d)][0] += [k]
        for i in range(8):
            date = d + timedelta(days=i)
            if date < max_date:
                data[(k[0], date)][1] += [k]
        for i in range(31):
            date = d + timedelta(days=i)
            if date < max_date:
                data[(k[0], date)][2] += [k]
    # for k in data:
    #     print len(data[k][0]), len(data[k][1]), len(data[k][2])
    # print 'unique dates:', len(unique_dates)

    for t in ticks:
        for i in range(1, len(tick_data[t])):
            # print (t, tick_data[t][i][1]), (t, tick_data[t][i][1]) in data
            if (t, tick_data[t][i][1]) in data:
                if tick_data[t][i][5] < tick_data[t][i - 1][5]: #try ternary classifier
                    data[(t, tick_data[t][i][1])][3] = 1
                else:
                    data[(t, tick_data[t][i][1])][3] = 0

    print len(data)

    weekends = []
    for k in data:
        if data[k][3] == -1:
            # if k[1].date().weekday() < 5 and k[0] in ticks:
            #     print k[0], k[1], k[1].date().weekday() >= 5
            weekends += [k]

    for k in weekends:
        del data[k]

    print len(data)

    u = np.loadtxt('1000epochs3/u/u_epoch_500.csv')

    emb_data = {}
    up = 0
    down = 0
    # rip = 0
    for k in data:
        st = np.array([u[run_info['info2index'][data[k][0][i]]] for i in range(len(data[k][0]))])
        mt = np.array([u[run_info['info2index'][data[k][1][i]]] for i in range(len(data[k][1]))])
        lt = np.array([u[run_info['info2index'][data[k][2][i]]] for i in range(len(data[k][2]))])
        label = np.array(data[k][3]).reshape((1,))
        if data[k][3] == 1:
            up += 1
        else:
            down += 1
        if st.shape == (0,):
            st = np.zeros((1, 100))
            # rip += 1
        if mt.shape == (0,):
            mt = np.zeros((30, 100))
        if mt.shape[0] < 30:
            mt_temp = np.zeros((30, 100))
            mt_temp[:len(mt)] = mt
            mt = mt_temp
        if lt.shape[0] < 100:
            lt_temp = np.zeros((100, 100))
            lt_temp[:len(lt)] = lt
            lt = lt_temp

        # print st.shape, mt.shape, lt.shape
        emb_data[k] = [st, mt, lt, label]
    # print emb_data.values()[0]
    # print up, down

    train_data = {}
    val_data = {}

    # st = [0 for i in range(101)]
    # mt = [0 for i in range(101)]
    # lt = [0 for i in range(101)]
    for k in emb_data:
        # st[len(emb_data[k][0])] += 1
        # mt[len(emb_data[k][1])] += 1
        # lt[len(emb_data[k][2])] += 1
        if k[1] >= val_date:
            val_data[k] = emb_data[k]
        else:
            train_data[k] = emb_data[k]

    print len(train_data), len(val_data)

    # print st
    # print mt
    # print lt

    # print len([k[1] for k in data.keys()])
    # plt.hist([k[1] for k in data.keys()])
    # plt.show()

    # x = []

    # for k in emb_data:
    #     if np.count_nonzero(emb_data[k][0]) == 0:
    #         continue
    #     x += [[k[0], k[1]] + list(np.mean(emb_data[k][0], axis=0)) + [emb_data[k][3][0]]]
    # # x = np.array(x)
    # print len(x), len(x[0])
    # print x[0]

    # df = pd.DataFrame(x)
    # print df
    # df.to_csv('tic_date_emb_label.csv')
    # df = pd.read_csv('tic_date_emb_label.csv')
    # # print df
    # print df.iloc[0,3:103], type(df.iloc[0,3:104])


    # exit()

    with tf.Session() as sess:
        model = CNN(sess)

        # model.train(emb_data.values())
        model.train(train_data.values(), val_data.values())

if __name__ == '__main__':
    tf.app.run()
