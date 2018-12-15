import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER_NODE = 500
MOVING_AVERAGE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
LEARNING_RATE_BASE = 0.8
BATCH_SIZE = 100
LEARNING_RATE_DECAY = 0.99
TRAINING_STEP = 5000

#forward_propagation
def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1) + biases1)
        return tf.matmul(layer1,weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2)) + avg_class.average(biases2)

#training_model
def train(mnist):
    x = tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')

    #parameter
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER_NODE],stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER_NODE]))
    weights2 = tf.Variable(tf.truncated_normal([LAYER_NODE,OUTPUT_NODE],stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

    y = inference(x,None,weights1,biases1,weights2,biases2)

    #exponential moving average
    global_step = tf.Variable(0,trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    average_y = inference(x,variable_averages,weights1,biases1,weights2,biases2)

    #cross_entropy
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #regularization
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)

    #loss function
    loss = cross_entropy_mean + regularization

    #exponential dacay  / learning rate
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,
                                               mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    train_op = tf.group(train_step,variable_averages_op)

    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
        test_feed = {x:mnist.test.images,y_:mnist.test.labels}

        for i in range(TRAINING_STEP):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy,feed_dict=validate_feed)
                print("after %d training step,validation accuracy using average model is %g"%(i,validate_acc))
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})
        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print('after %d training step,test accuracy using average moedel is %g'%(TRAINING_STEP,test_acc))

def main(argv=None):
    mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()