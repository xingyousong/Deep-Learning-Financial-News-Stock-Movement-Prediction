import json
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import cPickle as pickle
import os




class NTN():

    def __init__(self, sess, std=1):
        
        self.sess = sess

        self.train_size = 6193
        self.epochs = 1000
        self.batch_size = 1

        self.l2_lambda = 1e-4
        self.lr = 1e-4

        self.t_std = 1/np.sqrt(self.train_size)
        self.u_std = std

        self.d = 100
        self.k = 100

        # self.e_index = 0


        self.checkpoint_dir = 'checkpoints/'
        self.build_model()

    def build_model(self):

        self.events = tf.placeholder(tf.float32, [self.batch_size, 3, self.d], name='event_triples')
        self.e_index = tf.placeholder(tf.int32, name='e_index')

        self.corrupt =  tf.placeholder(tf.float32, [self.d])
        self.events_corrupt = tf.concat([tf.reshape(self.corrupt, [self.batch_size, 1, self.d]), self.events[:,1:,:]], 1, name='corrupt_event_triples')


        self.w = tf.get_variable('W', [self.k, 2*self.d], tf.float32, tf.random_normal_initializer(stddev=0))
        self.b = tf.get_variable('b', [self.k], tf.float32, tf.random_normal_initializer(stddev=0))
        self.u = tf.get_variable('u', [self.train_size, self.k], tf.float32, tf.random_normal_initializer(mean=0, stddev=self.u_std))

        # print self.events_corrupt.shape, self.events.shape

        self.us = self.network(self.events[0])
        self.us_corrupt = self.network(self.events_corrupt[0], reuse=True)

        self.score = tf.matmul(self.us, tf.reshape(self.u[self.e_index], [self.k, 1])) / self.k
        self.score_corrupt = tf.matmul(self.us_corrupt, tf.reshape(self.u[self.e_index], [self.k, 1])) / self.k

        self.u_loss = tf.maximum(1 - self.score + self.score_corrupt, 0)
        self.l2_loss = sum([tf.reduce_sum(tf.square(var)) for var in tf.trainable_variables() if var.name != 'u:0'])
        print [var for var in tf.trainable_variables() if var.name != 'u:0']

        # print self.u_loss.shape, self.l2_loss.shape

        # print tf.trainable_variables()

        self.loss = self.u_loss + (self.l2_lambda * self.l2_loss)

        self.saver = tf.train.Saver()


    def neural_turing_layer(self, input, d, k, scope=None, name=''):
        t = tf.get_variable('T'+name, [d, d, k], tf.float32, tf.random_normal_initializer(stddev=self.t_std))
        temp = tf.matmul(input[:1,:],tf.reshape(t,[d,d*k]))
        btp = tf.reshape(tf.matmul(input[1:,:],tf.reshape(temp,[d,k])), [1, k])
        # print self.w.shape, tf.reshape(input, [2*d, 1]).shape
        ff = tf.reshape(tf.matmul(self.w, tf.reshape(input, [2*d, 1])), [1, k])
        return tf.nn.tanh(btp + ff + self.b)


    def network(self, input, reuse=False):
        with tf.variable_scope('ntn') as scope:
            if reuse:
                scope.reuse_variables()
            r1 = self.neural_turing_layer(input[:2,:], self.d, self.k, scope='ntn', name='r1')
            r2 = self.neural_turing_layer(input[1:,:], self.d, self.k, scope='ntn', name='r2')

            rs = tf.concat([r1, r2], 0)
            u = self.neural_turing_layer(rs, self.k, self.k, scope='ntn', name='u')

            return u


    def train(self, data=None, We=[]):
        ntn_optim = tf.train.AdamOptimizer(self.lr).minimize(self.loss, var_list=tf.trainable_variables())

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        counter = 0
        start_time = time.time()


        losses = []
        u_losses = []
        l2_losses = []
        u_corr_stds = []
        u_mean_stds = []
        scores = []
        score_corrupts = []

        for epoch in xrange(self.epochs):
            length = min(len(data), self.train_size)
            batch_idxs = length // self.batch_size

            batch_loss = 0
            batch_u_loss = 0
            batch_l2_loss = 0
            batch_score = 0
            batch_score_corrupt = 0

            for idx in xrange(0, batch_idxs):
                batch_data = data[idx*self.batch_size:(idx+1)*self.batch_size]

                feed_dict = {self.events: batch_data, self.e_index: self.batch_size*idx, self.corrupt: We[np.random.randint(0, len(We))]*5}
                # if idx == 0:
                #     print feed_dict[self.corrupt]
                vlist = tf.trainable_variables()
                vnames = [v.name for v in vlist]
                # print vnames
                __ = self.sess.run([ntn_optim, self.us, self.us_corrupt, self.u, self.score, self.score_corrupt] + vlist, feed_dict=feed_dict)
                _ = __[0]
                us_sample = __[1]
                corrupt_sample = __[2]
                u = __[3]
                score = __[4]
                score_corrupt= __[5]
                vs = __[6:]

                u_loss = self.u_loss.eval(feed_dict)
                l2_loss = self.l2_loss.eval(feed_dict)
                loss = self.loss.eval(feed_dict)

                # print loss, u_loss, l2_loss
                batch_loss += loss
                batch_u_loss += u_loss
                batch_l2_loss += l2_loss
                batch_score += score
                batch_score_corrupt += score_corrupt
                
                counter += 1
                print('Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, u_loss: %.8f, l2_loss: %.8f' \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, loss, u_loss, l2_loss))

                if np.mod(counter, 1000) == 1:
                    if self.batch_size*idx+3 < len(u):
                        print 'U1:', u[self.batch_size*idx]
                        print 'U2:', u[self.batch_size*idx+1]
                        print 'U3:', u[self.batch_size*idx+2]
                        print 'U4:', u[self.batch_size*idx+3]
                    # print 'SAMPLE U:', us_sample
                    # print 'CORRUPT U:', corrupt_sample
                    print 'SAMPLE SCORE:', score, np.dot(u[self.batch_size*idx], us_sample.reshape(self.k)), np.dot(u[0], us_sample.reshape(self.k))
                    print 'CORRUPT SCORE:', score_corrupt, np.dot(u[self.batch_size*idx], corrupt_sample.reshape(self.k)), np.dot(u[0], corrupt_sample.reshape(self.k))
                    # print vs

                    x = {}
                    for i in range(len(vnames)):
                        if vnames[i] == 'u:0':
                            continue
                        x[vnames[i]] = vs[i]
                        # print vnames[i], vs[i].shape
                    w = x['W:0']
                    b = x['b:0']
                    t1 = x['ntn/Tr1:0']
                    t2 = x['ntn/Tr2:0']
                    t3 = x['ntn/Tu:0']

                    def layer(input, t, w, b, d=100, k=100):
                        temp = np.dot(input[:1,:],np.reshape(t,[d,d*k]))
                        btp = np.reshape(np.dot(input[1:,:],np.reshape(temp,[d,k])), [1, k])
                        ff = np.reshape(np.dot(w, np.reshape(input, [2*d, 1])), [1, k])
                        return np.tanh(btp + ff + b)

                    # for i in range(len(emb)):
                    r1 = layer(data[idx*self.batch_size][:2], t1, w, b)
                    r1_c = layer(np.array([feed_dict[self.corrupt], data[idx*self.batch_size][1]]), t1, w, b)
                    r1_c1 = layer(np.array([We[np.random.randint(0, len(We))]*5, data[idx*self.batch_size][1]]), t1, w, b)
                    r1_c2 = layer(np.array([We[np.random.randint(0, len(We))]*5, data[idx*self.batch_size][1]]), t1, w, b)
                    r1_c3 = layer(np.array([We[np.random.randint(0, len(We))]*5, data[idx*self.batch_size][1]]), t1, w, b)
                    r2 = layer(data[idx*self.batch_size][1:], t2, w, b)
                    r3 = layer(np.array([r1, r2]), t3, w, b)
                    r3_c = layer(np.array([r1_c, r2]), t3, w, b)
                    r3_c1 = layer(np.array([r1_c1, r2]), t3, w, b)
                    r3_c2 = layer(np.array([r1_c2, r2]), t3, w, b)
                    r3_c3 = layer(np.array([r1_c3, r2]), t3, w, b)
                    print 'actual manual score:', r3.dot(u[self.batch_size*idx])[0]
                    print 'actual corrupt scores:', r3_c.dot(u[self.batch_size*idx])[0]
                    print 'other corrupt scores:', r3_c1.dot(u[self.batch_size*idx])[0], r3_c2.dot(u[self.batch_size*idx])[0], r3_c3.dot(u[self.batch_size*idx])[0]


                if np.mod(counter, 3000) == 1:
                    np.savetxt('u.csv', u)

                    prods = u.dot(u.T)

                    # print prods.shape
                    # print prods.reshape((1000000)).shape

                    for i in range(length):
                        prods[i,i] = 0

                    u_corr_stds += [np.std(prods.reshape((length**2)))]
                    u_mean_stds += [np.mean(np.std(u, axis=1))]


            np.savetxt('u/u_epoch_{}.csv'.format(epoch+1), u)

            batch_loss /= batch_idxs
            batch_u_loss /= batch_idxs
            batch_l2_loss /= batch_idxs
            batch_score /= batch_idxs
            batch_score_corrupt /= batch_idxs
            losses += [batch_loss[0][0]]
            u_losses += [batch_u_loss[0][0]]
            l2_losses += [self.l2_lambda * batch_l2_loss]
            scores += [batch_score[0][0]]
            score_corrupts += [batch_score_corrupt[0][0]]
            # print losses

            plt.figure()
            plt.plot([i+1 for i in range(len(losses))], losses, label='total loss')
            plt.plot([i+1 for i in range(len(u_losses))], u_losses, label='u loss')
            plt.plot([i+1 for i in range(len(l2_losses))], l2_losses, label='l2 loss')
            plt.plot([i+1 for i in range(len(scores))], scores, label='score')
            plt.plot([i+1 for i in range(len(score_corrupts))], score_corrupts, label='score_c')
            plt.plot([i+1 for i in range(len(score_corrupts))], [-s for s in score_corrupts], label='-score_c')
            plt.legend()
            plt.savefig('losses.png')

            plt.figure()
            plt.plot([(i+1)/2 for i in range(len(u_corr_stds))], u_corr_stds)
            plt.savefig('u_corr_stds.png')

            plt.figure()
            plt.plot([(i+1)/2 for i in range(len(u_corr_stds))], u_mean_stds)
            plt.savefig('u_mean_stds.png')

            if (epoch+1) % 10 == 0:
                np.savetxt('u/u_epoch_{}.csv'.format(epoch+1), u)

                with open('train_stats.p', 'w') as outfile:
                    pickle.dump({
                        'losses': losses,
                        'u_losses': u_losses,
                        'l2_losses': l2_losses,
                        'score': scores,
                        'score_corrupts': score_corrupts,
                        'u_corr_stds': u_corr_stds,
                        'u_mean_stds': u_mean_stds,
                    }, outfile)

            if (epoch+1 >= 0) and ((epoch+1) % 10 == 0):
                with open('checkpoints/vars_epoch_{}.p'.format(epoch+1), 'w') as outfile:
                    vout = {}
                    for i in range(len(vnames)):
                        if vnames[i] == 'u:0':
                            continue
                        vout[vnames[i]] = vs[i]
                        # print vnames[i], vs[i].shape
                    pickle.dump(vout, outfile)







def main(_):
    
    we_file = 'we.npz'
    w2i_file = 'w2i.json'

    concat = False
    npz = np.load(we_file)
    W1 = npz['arr_0']
    W2 = npz['arr_1']
    with open(w2i_file) as f:
        word2idx = json.load(f)

    V = len(word2idx)
    if concat:
        We = np.hstack([W1, W2.T])
        print 'We.shape:', We.shape
        assert(V == We.shape[0])
    else:
        We = (W1 + W2.T) / 2

    We_dict = {}
    for i in range(len(We)):
        We_dict[i] = We[i]
    We_mat = We
    We = We_dict

    # print We[word2idx['google']]
    # print We[word2idx['apple']]
    word_index = max(word2idx.values()) + 1

    with open('openie.p', 'r') as infile:
        openie = pickle.load(infile)

    # print openie.items()[0]

    index = 0
    info2index = {}
    index2info = []


    word_emb = []

    for k in openie:
        info2index[k] = index
        index += 1
        index2info += [k]

        emb = np.zeros((3, 100))
        for e in openie[k]:
            sub = e['subject'].split(' ')
            for w in sub:
                if w not in word2idx:
                    word2idx[w] = word_index
                    word_index += 1
                if word2idx[w] not in We:
                    We[word2idx[w]] = np.random.randn(100) * .1
                emb[0] = We[word2idx[w]]
            rel = e['relation'].split(' ')
            for w in rel:
                if w not in word2idx:
                    word2idx[w] = word_index
                    word_index += 1
                if word2idx[w] not in We:
                    We[word2idx[w]] = np.random.randn(100) * .1
                emb[1] = We[word2idx[w]]
            obj = e['object'].split(' ')
            for w in obj:
                if w not in word2idx:
                    word2idx[w] = word_index
                    word_index += 1
                if word2idx[w] not in We:
                    We[word2idx[w]] = np.random.randn(100) * .1
                emb[2] = We[word2idx[w]]
        emb /= len(openie[k])
        word_emb += [emb]

    with open('run_info.p', 'w') as outfile:
        pickle.dump({
            'We_mat': We_mat,
            'We_dict': We_dict,
            'word2idx': word2idx,
            'word_emb': word_emb,
            'info2index': info2index,
            'index2info': index2info
        }, outfile)

    word_emb = np.array(word_emb)
    print word_emb.shape
    print index, word_index

    print np.std(We_mat, axis=1).shape
    print np.mean(np.std(We_mat, axis=1))
    # std = np.mean(np.std(word_emb, axis=2))
    print np.max(We_mat), np.min(We_mat), np.mean(np.abs(We_mat))

    std = .5

    with open('checkpoints/vars_epoch_0.p', 'r') as f:
        x = pickle.load(f)
        for k in x:
            print k, x[k].shape

    # print word_emb[0]
    # print word_emb[1]

    # exit(1)

    with tf.Session() as sess:
        model = NTN(sess, std)

        model.train(word_emb*5, We)

if __name__ == '__main__':
    tf.app.run()
