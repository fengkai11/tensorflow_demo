import tensorflow as tf
a = tf.Variable(1.0,name = 'a')
b = tf.Variable(2.0,name = 'b')
c = tf.add(a,b)
t = tf.truncated_normal_initializer(stddev = 0.1,seed = 1)
v = tf.get_variable('v',[1],initializer=t)
tl = tf.nn.l2_loss(v)
tf.add_to_collection('loss',v)
tf.add_to_collection('loss',tl)

input = tf.Variable(tf.random_normal([1,3,3,5]))
filter = tf.Variable(tf.random_normal([2,2,5,1]))
op1 = tf.nn.conv2d(input,filter,strides = [1,1,1,1],padding = 'SAME')
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tmp = sess.run(op1)
    print(tmp.shape)