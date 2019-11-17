#coding:utf-8
#REF:tensorflow变量作用域（https://www.cnblogs.com/MY0213/p/9208503.html）
import tensorflow as tf
import numpy as np
from PIL import Image
'''
variable scope:变量作用域；
变量一般就是模型参数，如果需要共享可以设置为全局变量，但这样会影响封装；
通过设置作用域可以共享变量名；
tf.get_variable(<name>,<shape>,<initializer>)创建或者返回给定名称的变量；
tf.variable_scope(<scope_name>)管理传给get_variable()的变量名的作用域；
'''
def conv_relu(input,kernel_shape,bias_shpape):
    weights = tf.get_variable("weights",kernel_shape,initializer=tf.random_normal_initializer())
    tf.add_to_collection('net_weights', weights)
    biases = tf.get_variable("biases",bias_shpape,initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input,weights,strides=[1,1,1,1],padding = 'SAME')
    return tf.nn.relu(conv+biases)
'''
当我们需要两个卷积层时就可以通过tf.variable_scope去区分变量；
'''
def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        net = conv_relu(input_images,[5,5,3,32],[32])
    with tf.variable_scope("conv2"):
        net = conv_relu(net,[5,5,32,32],[32])
    return net



if __name__ == "__main__":
    # input = np.random.rand(1,224,224,3)#np.random.rand(size);
    im = Image.open(r'D:\demo.jpg')
    im = np.asanyarray(im)
    im = np.float32(im)
    im =  im[np.newaxis,:,:,:]
    print(im.shape)
    '''
    在新的作用域中重复使用第一次使用的变量
    '''
    with tf.variable_scope("image_filter") as scope:
        pred = my_image_filter(im)
        scope.reuse_variables()
        # pred2 = my_image_filter(im)
    #end

    '''
    工作机制
    变量都是通过 作用域/变量名 来标识的；
    默认reues是false，每次都是创建新的变量；
    '''
    # 创建新的变量；
    with tf.variable_scope("foo"):
        v = tf.get_variable("v",[1])

    #reuse
    with tf.variable_scope("foo",reuse = True):
        v2 = tf.get_variable("v",[1])
    print(v.name)
    print(v2.name)
    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        '''
        获取变量：
        tf.add_to_collection：把变量放入一个集合，把很多变量变成一个列表
        tf.get_collection：从一个结合中取出全部变量，是一个列表
        '''
        wgt = tf.get_collection('net_weights')
        wgt = sess.run(wgt)
        print(len(wgt))
        exit(-1)
        rst = sess.run(pred)
        rst = np.squeeze(rst)
        print(rst.shape)
        img = Image.fromarray(np.uint8(rst[:,:,2]))
        img.show()
'''
Summary:
tf.get_variable(...)
tf.variable_scpe(...)
tf.add_to_collection(...)
tf.get_collection(...)

Image.open()
im.show()
im = np.asanyarray(im)
im = np.float32(im)
im =  im[np.newaxis,:,:,:]
im = np.expand_dim(),增加维度；
np.squeeze(),减少无用的维度；
'''