import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
class Batch:
    def __init__(self, data, batchSize):
        self.data = np.copy(data)
        self.count = 0
        self.batchSize = batchSize
    
    def getNextBatch(self):
        if self.count >= len(self.data) / self.batchSize:
            self.count = 0
            np.random.shuffle(self.data)
        start = self.count * self.batchSize
        end = start + self.batchSize
        toReturn = self.data[start:end]
        self.count += 1
        return toReturn

LEARNING_RATE_START = 0.0001
LEARNING_RATE_END   = 0.000001
N_EPOCHS      = 500
anneal_rate = (-1.0 * np.log(LEARNING_RATE_END / LEARNING_RATE_START)) / float(N_EPOCHS)
batchSize = 50

train_data = np.load("../data/trainingDataClassify.npy")
valid_data = np.load("../data/validationDataClassify.npy")
print(len(train_data))
trainBatch = Batch(train_data, batchSize)
epochSize = len(train_data) / batchSize
curr_lr = LEARNING_RATE_START

sess = tf.Session()
beta = 0.001
A = tf.Variable(tf.random_normal([60, 2048], stddev=0.1))
B = tf.Variable(tf.random_normal([2048, 2048], stddev=0.1))
C = tf.Variable(tf.random_normal([2048, 1024], stddev=0.1)) 
D = tf.Variable(tf.random_normal([1024, 1024], stddev=0.1))
E = tf.Variable(tf.random_normal([1024, 2780], stddev=0.1))
biasA = tf.Variable(tf.zeros([2048]), name = 'biasesA')
biasB = tf.Variable(tf.zeros([2048]), name = 'biasesB')
biasC = tf.Variable(tf.zeros([1024]), name = 'biasesC')
biasD = tf.Variable(tf.zeros([1024]), name = 'biasesD')
biasE = tf.Variable(tf.zeros([2780]), name = 'biasesE')
X = tf.placeholder(tf.float32, [None, 60])
y = tf.placeholder(tf.float32, [None, 2780]) 
lr  = tf.placeholder(tf.float32)

regularizer = tf.nn.l2_loss(A) + tf.nn.l2_loss(B) + tf.nn.l2_loss(C) + tf.nn.l2_loss(D)
eval_dict={
X   : valid_data[:,:-1],
y   : valid_data[:,-1:]
}

train_dict = {
    X : train_data[:,:-1],
    y : train_data[:,-1:]
}

hidden1 = tf.nn.relu(tf.matmul(tf.cast(X, tf.float32), A) + biasA)
hidden2 = tf.nn.relu(tf.matmul(hidden1, B) + biasB)
hidden3 = tf.nn.relu(tf.matmul(hidden2, C) + biasC)
hidden4 = tf.nn.relu(tf.matmul(hidden3, D) + biasD)
estimate = tf.nn.softmax(tf.matmul(hidden4, E) + biasE)
#estimate = tf.clip_by_value(estimate, 0, 2779)

metric = tf.reduce_mean(tf.abs(tf.transpose(estimate) - y))
loss = tf.nn.l2_loss(estimate - y)
optimizer = tf.train.AdamOptimizer(lr)
train_step = optimizer.minimize(loss)

saver = tf.train.Saver([A,B,C,D,E,biasA, biasB, biasC, biasD, biasE])

init = tf.global_variables_initializer()
sess.run(init)

# saver = tf.train.import_meta_graph('../weights/ZestimateWeight.meta')
# saver.restore(sess, tf.train.latest_checkpoint('./'))

best_loss = sess.run(loss, eval_dict)
best_training_loss = None
for i in range(N_EPOCHS*epochSize):
    checkpointed = False
    
    if i %  epochSize == 0:
        print("epoch #" + str(i /epochSize))
        valid_loss = sess.run(loss, eval_dict)
        train_loss = sess.run(loss, train_dict)
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_training_loss = train_loss
            saver.save(sess, "../weights/ZestimateWeight")
            if i >= 25*epochSize:  
                curr_lr = LEARNING_RATE_START * np.exp(-1.0 * anneal_rate * i)
            checkpointed = True
        print("checkPointed = " + str(checkpointed))
        print("validationLoss = " + str(valid_loss))
        print("trainLoss = " + str(train_loss))
    currBatch = trainBatch.getNextBatch()
    sess.run(train_step, feed_dict={
       X   : currBatch[:, :-1],
       y   : currBatch[:, -1:],
       lr  : curr_lr
    })
print("BEST LOSS = " + str(best_loss))  
print("Corresponding Train Loss = " + str(best_training_loss)) 
