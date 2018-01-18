import numpy as np
import matplotlib.pyplot as plt
import time
import cPickle as pickle
import seaborn


with open('1000epochs3/train_stats.p', 'r') as f:
    x = pickle.load(f)

plt.figure(figsize=(7, 5))
plt.plot([i+1 for i in range(len(x['losses']))], x['losses'], label='Total Loss')
plt.plot([i+1 for i in range(len(x['losses']))], x['u_losses'], label='Margin Loss')
plt.plot([i+1 for i in range(len(x['losses']))], x['l2_losses'], label='L2 Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
# plt.show()
plt.savefig('ntn_losses.png')

exit(1)


n = 6193

u = np.loadtxt('u/u_epoch_650.csv')

# print u.shape

# prods = u.dot(u.T)

# print prods.shape
# print prods.reshape((n**2)).shape


# for i in range(n):
#   prods[i,i] = 0

# print np.std(prods.reshape((n**2)))

with open('checkpoints/vars_epoch_650.p', 'r') as f:
    x = pickle.load(f)
    # for k in x:
    #     print k, x[k]

w = x['W:0']
b = x['b:0']
t1 = x['ntn/Tr1:0']
t2 = x['ntn/Tr2:0']
t3 = x['ntn/Tu:0']

with open('run_info.p', 'r') as infile:
    run_info = pickle.load(infile)
    We = run_info['We_mat']

emb = np.array(run_info['word_emb'])*5

print emb.shape


def layer(input, t, w, b, d=100, k=100):
    temp = np.dot(input[:1,:],np.reshape(t,[d,d*k]))
    btp = np.reshape(np.dot(input[1:,:],np.reshape(temp,[d,k])), [1, k])
    ff = np.reshape(np.dot(w, np.reshape(input, [2*d, 1])), [1, k])
    return np.tanh(btp + ff + b)

for i in range(len(emb)):
    r1 = layer(emb[i][:2], t1, w, b)
    r1_c1 = layer(np.array([We[np.random.randint(0, len(We))]*5, emb[i][1]]), t1, w, b)
    r1_c2 = layer(np.array([We[np.random.randint(0, len(We))]*5, emb[i][1]]), t1, w, b)
    r1_c3 = layer(np.array([We[np.random.randint(0, len(We))]*5, emb[i][1]]), t1, w, b)
    r2 = layer(emb[i][1:], t2, w, b)
    r3 = layer(np.array([r1, r2]), t3, w, b)
    r3_c1 = layer(np.array([r1_c1, r2]), t3, w, b)
    r3_c2 = layer(np.array([r1_c2, r2]), t3, w, b)
    r3_c3 = layer(np.array([r1_c3, r2]), t3, w, b)
    print r3.dot(u[i]), r3_c1.dot(u[i]), r3_c2.dot(u[i]), r3_c3.dot(u[i])
    # print r3
    # print r3_c1
    # print r3_c2
    # print r3_c3
    if i == 10:
        break

# exit(1)            

means = []
stds = []

start_time = time.time()

for i in range(1, 251, 10):
    if i%100 == 0:
        print i, time.time() - start_time
    u = np.loadtxt('u/u_epoch_{}.csv'.format(i))
    prods = u.dot(u.T)

    # print prods.shape
    # print prods.reshape((n**2)).shape

    for i in range(n):
        prods[i,i] = 0

    stds += [np.std(prods.reshape((n**2)))]
    means += [np.mean(prods.reshape((n**2)))]



# plt.imshow(prods, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.show()

# plt.hist(prods.reshape((n**2)))
# plt.show()

plt.plot(stds)
plt.plot(means)
plt.show()

r = np.random.randn(n, 100) * .5

prods = r.dot(r.T)


for i in range(n):
    prods[i,i] = 0


print np.std(prods.reshape((n**2)))


plt.imshow(prods, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()

plt.hist(prods.reshape((n**2)))
plt.show()