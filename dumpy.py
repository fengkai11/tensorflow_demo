from __future__ import absolute_import
from __future__ import division
import os
import numpy as np
import tensorflow as tf
import cv2
import re


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir','',""""Directory where to write event logs""")

tf.app.flags.DEFINE_integer('max_steps',10000,""""Number of batch to run""")
tf.app.flags.DEFINE_string('data_dir',r'D:\my_code\project\tensorflow_demo\data\cifar-10-batches-bin',"""Patch to cifar10 directory""")
tf.app.flags.DEFINE_integer('batch_size',1,"""Number of images to process in a batch""")

IMAGES_SIZE = 24
NUM_CLASS = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
TOWER_NAME = 'tower'

def read_cifar10(filename_queue):
    """
    TODO: change another way to read data
    :param file_name_queue:
    :return: a class,contain key words:
    height:32
    width:32
    depth:3
    key :
    label:
    uint8images:[height,width,depth]
    """
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3

    image_bytes = result.height*result.width*result.depth
    record_bytes = label_bytes+ image_bytes
    reader = tf.FixedLengthRecordReader(record_bytes = record_bytes)
    result.key,value = reader.read(filename_queue)
    record_im = tf.decode_raw(value,tf.uint8)
    result.label = tf.cast(tf.slice(record_im,[0],[label_bytes]),tf.int32)
    depth_major = tf.reshape(tf.slice(record_im,[label_bytes],[image_bytes]),[result.depth,result.height,result.width])
    result.uint8image = tf.transpose(depth_major,[1,2,0])
    return result
def distorted_inputs(batch_size):
    """

    :param batch_size:
    :return:
    images: Images,4D tensor of [batch_size,height,width,3]
    labels:labels. 1D tensor of [batch_size]
    """
    if not FLAGS.data_dir:
        raise ValueError('Please set a data_dir')
    filenames = [os.path.join(FLAGS.data_dir,'data_batch_%d.bin'%i) for i in range(1,6)]
    for item in filenames:
        if not os.path.exists(item):
            raise ValueError('%s is not exist'%item)
    filename_queue = tf.train.string_input_producer(filenames)
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image,tf.float32)

    height = IMAGES_SIZE
    width = IMAGES_SIZE
    distorted_images = tf.random_crop(reshaped_image,[height,width,3])
    distorted_images = tf.image.random_flip_left_right(distorted_images)
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN*min_fraction_of_examples_in_queue)
    return _generate_image_and_label_batch(read_input.uint8image, read_input.label,
                                         min_queue_examples, batch_size)

def _generate_image_and_label_batch(image,label,min_queue_examples,batch_size):
    num_preprocess_threads = 16
    images,label_batch = tf.train.shuffle_batch([image,label],batch_size=batch_size,
                                                num_threads = num_preprocess_threads,
                                                capacity = min_queue_examples+3*batch_size,
                                                min_after_dequeue= min_queue_examples)
    tf.summary.image('images',images)
    return images,tf.reshape(label_batch,[batch_size])

def _variable_with_weight_decay(name,shape,stddev,wd):
    """

    :param name: variable name
    :param shape:
    :param stddev:
    :param wd:
    :return:tensor
    """
    var = _variable_on_cpu(name,shape,tf.truncated_normal_initializer(stddev = stddev))
    if wd is not None:
        weight_decay = tf.multipy(tf.nn.l2_loss(var),wd,name='weight_loss')
        tf.add_to_collection('losses',weight_decay)
    return var
def _variable_on_cpu(name,shape,initializer):
    """

    :param name:
    :param shape:
    :param initializer:
    :return:
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name,shape,initializer = initializer)
    return var
def _activation_summary(x):
    """
    :param x:
    :return:
    """
    tensor_name = re.sub('%s_[0-9]*/'%TOWER_NAME,' ',x.op.name)
    tf.summary.histogram(tensor_name+'/activation',x)
    tf.summary.scalar(tensor_name+'/sparsity',tf.nn.zero_fraction(x))
def inference(images):
    """

    :param images:
    :return: Logits
    """
    #conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',shape = [5,5,3,64],stddev = 1e-4,wd = 0.0)
        conv = tf.nn.conv2d(images,kernel,[1,1,1,1],padding = 'SAME')
        biases = _variable_on_cpu('biases',[64],tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(bias,name = scope.name)
        _activation_summary(conv1)
    pool1 = tf.nn.max_pool(conv1,ksize = [1,3,3,1],strides = [1,2,2,1],padding = 'SAME',name = 'pool1')
    # norm1 = tf.nn.lrn()
    #TODO:BN
    with tf.variable_sope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',shape = [5,5,64,64],stddev = 1e-4,wd = 0.0)
        conv = tf.nn.conv2d(pool1,kernel,[1,1,1,1],padding = 'SAME')
        biases  = _variable_on_cpu('biases',[64],tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(bias,name = scope.name)
        _activation_summary(conv2)
    pool2 = tf.nn.max_pool(conv2,ksize = [1,3,3,1],strides = [1,2,2,1],padding = 'SAME',name = 'pool2')




    
if __name__ == "__main__":
    images, labels = distorted_inputs(batch_size=FLAGS.batch_size)
    im = images[0]

    with tf.Session() as sess:

        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                imgs = sess.run(im)
                print(imgs.shape)
                r, g, b = cv2.split(imgs)
                imgs = cv2.merge([b, g, r])
                cv2.imshow('test', imgs)
                cv2.waitKey(0)
        except tf.errors.OutOfRangeError:
            print('done')
        finally:
            coord.request_stop()
        coord.join(thread)


