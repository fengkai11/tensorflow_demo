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

y2 = tf.convert_to_tensor([[0, 0, 1, 0]], dtype=tf.int64)
y_2 = tf.convert_to_tensor([[-2.6, -1.7, 3.2, 0.1]], dtype=tf.float32)
t= tf.argmax(y2,1)
c2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits= y_2, labels=tf.argmax(y2,1))
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tmp = sess.run(t)
    print(tmp.shape)