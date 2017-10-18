# Load pickled data
import pickle
import numpy  as np
import tensorflow as tf
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import random
import csv
from tensorflow.contrib.layers import flatten
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# TODO: Fill this in based on whe8=;re you saved the training and testing data
'''
training_file = "train.p"
validation_file="valid.p"
testing_file = "test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
'''
(X_train,y_train),(X_test,y_test) = cifar10.load_data()
#it's a good idea to flatten the array
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.3,random_state=32,stratify=y_train)
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes =len(np.unique(y_train))


print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.

# Visualizations will be shown in the notebook.
#matplotlib inline
def vis(X_train):
    index = random.randint(0,len(X_train))
    image = X_train[index].squeeze()
    plt.imshow(image)
    print(y_train[index])
#vis(X_train)

def plot_figures(figures,nrows = 1,ncols = 1,labels=None):
    fig, axs = plt.subplots(ncols = ncols,nrows = nrows,figsize=(12,14))
    axs = axs.ravel()# packed as 1-D array
    for index,title in zip(range(len(figures)),figures):#make it as a tuple ,figures is a dict(int:array)
        axs[index].imshow(figures[title],plt.gray())
        if(labels != None):
            axs[index].set_title(labels[index])
        else:
            axs[index].set_title(title)
            
        axs[index].set_axis_off()#turn off
    
    plt.tight_layout()
    
name_values = np.genfromtxt('signnames.csv',skip_header=1,dtype=None,delimiter=',')

figures = {}
labels = {}
for i in range(8):
    index = random.randint(0,n_train-1)
    labels[i] = name_values[y_train[index]][1].decode('ascii')
    figures[i] = X_train[index]
    
plot_figures(figures,4,2,labels)


### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.

#to grayscale
X_train_gray = np.sum(X_train/3,axis= 3,keepdims = True)
X_test_gray = np.sum(X_test/3,axis= 3,keepdims = True)
X_valid_gray = np.sum(X_valid/3,axis=3,keepdims =True)
print(X_train_gray.shape)
#vis(X_train_gray)
X_train = X_train_gray
X_test = X_test_gray
X_valid = X_valid_gray
###normalize
X_train_normalize = (X_train-128)/128
X_test_normalize = (X_test-128)/128
X_valid_normalize = (X_valid-128)/128
print(np.mean(X_train_normalize))
#vis(X_train_normalize)
X_train = X_train_normalize
X_test = X_test_normalize
X_valid = X_valid_normalize


#mix the train and valid

X_train = np.concatenate((X_train,X_valid),axis=0)
y_train = np.concatenate((y_train,y_valid),axis=0)

X_train,y_train = shuffle(X_train,y_train)
X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size = 0.2,random_state=0)

unique_train,counts_train = np.unique(y_train,return_counts=True)
plt.bar(unique_train,counts_train)
plt.grid()
plt.title("train data counts after split")
plt.show()

unique_valid,counts_valid = np.unique(y_valid,return_counts=True)
plt.bar(unique_valid,counts_valid)
plt.grid()
plt.title('valid data counts afte spliting')
plt.show()
#augmenting data

### Define your architecture here.

keep_prob = tf.placeholder(tf.float32) 
def LeNet(x):
    mu = 0
    sigma = 0.1
    
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5,5,1,6),mean=mu,stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x,conv1_W,strides=[1,1,1,1],padding = 'VALID') + conv1_b
    
    conv1 = tf.nn.relu(conv1)
    
    conv1 = tf.nn.max_pool(conv1,ksize = [1,2,2,1],strides=[1,2,2,1],padding='VALID')
    
    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)
    
    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)
    #changed that
    fc1 = tf.nn.dropout(fc1,keep_prob)
    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)
    #changed that
    fc2 = tf.nn.dropout(fc2,keep_prob)
    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 10), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits

    ### Train your model here.
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)
#keep_prob = tf.placeholder(tf.float32)
#change EPOCHS 10-27, Bathc_size 128-156, rate 0.001 - 0.00097
EPOCHS = 30
BATCH_SIZE = 128

rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y,keep_prob: 0.90})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

### Calculate and report the accuracy on the training and validation set.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y,keep_prob:0.90})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
### Once a final model architecture is selected, 
    saver.save(sess, './lenet')
    print("Model saved")
### the accuracy on the test set should be calculated and reported as well.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver2 = tf.train.import_meta_graph('./lenet.meta')
    saver2.restore(sess, "./lenet")
    test_accuracy = evaluate(X_test, y_test)
    print("Test Set Accuracy = {:.3f}".format(test_accuracy))
### Feel free to use as many code cells as needed.