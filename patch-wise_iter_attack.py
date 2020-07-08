"""Implementation of sample attack."""
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from utils import *
from attack_method import *
from tqdm import tqdm
from tensorpack import TowerContext
from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2, densenet, fdnets
from tensorpack.tfutils import get_model_loader
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope

slim = tf.contrib.slim

tf.flags.DEFINE_string('checkpoint_path', './models', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('input_csv', 'dataset/dev_dataset.csv', 'Input directory with images.')

tf.flags.DEFINE_string('input_dir', 'dataset/images/', 'Input directory with images.')

tf.flags.DEFINE_string('output_dir', 'output/', 'Output directory with images.')

tf.flags.DEFINE_float('max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer('num_classes', 1001, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer('num_iter', 10, 'Number of iterations.')

tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer('image_resize', 330, 'Height of each input images.')

tf.flags.DEFINE_integer('batch_size', 10, 'How many images process at one time.')

tf.flags.DEFINE_float('amplification_factor', 5.0, 'To amplifythe step size.')

tf.flags.DEFINE_float('momentum', 1.0, 'Momentum.')

tf.flags.DEFINE_float('prob', 0.7, 'probability of using diverse inputs.')

FLAGS = tf.flags.FLAGS

model_checkpoint_map = {
    'inception_v3': os.path.join(FLAGS.checkpoint_path, 'inception_v3.ckpt'),
    'adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'adv_inception_v3_rename.ckpt'),
    'ens3_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens3_adv_inception_v3_rename.ckpt'),
    'ens4_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens4_adv_inception_v3_rename.ckpt'),
    'inception_v4': os.path.join(FLAGS.checkpoint_path, 'inception_v4.ckpt'),
    'inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'inception_resnet_v2_2016_08_30.ckpt'),
    'ens_adv_inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'ens_adv_inception_resnet_v2_rename.ckpt'),
    'resnet_v2_101': os.path.join(FLAGS.checkpoint_path, 'resnet_v2_101.ckpt'),
    'vgg_16': os.path.join(FLAGS.checkpoint_path,'vgg_16.ckpt'),
    'resnet_v2_152': os.path.join(FLAGS.checkpoint_path,'resnet_v2_152.ckpt'),
    'adv_inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'adv_inception_resnet_v2_rename.ckpt'),
    'resnet_v2_50': os.path.join(FLAGS.checkpoint_path,'resnet_v2_50.ckpt'),
    'densenet': os.path.join(FLAGS.checkpoint_path, 'tf-densenet161.ckpt'),
    'X101-DA': os.path.join(FLAGS.checkpoint_path, 'X101-DenoiseAll_rename.npz'),
    'R152-B': os.path.join(FLAGS.checkpoint_path, 'R152_rename.npz'),
    'R152-D': os.path.join(FLAGS.checkpoint_path, 'R152-Denoise_rename.npz'),
}

P_kern, kern_size = project_kern(7)
T_kern = gkern(15, 3)

def graph(x, y, i, x_max, x_min, grad, amplification):
    one_hot = tf.one_hot(y, FLAGS.num_classes)
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    momentum = FLAGS.momentum
    	
    # amplification factor
    beta = alpha * FLAGS.amplification_factor
    gamma = beta


    # DIM: https://arxiv.org/abs/1803.06978
    # input_diversity(FLAG, x)
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
              x, num_classes = FLAGS.num_classes, is_training = False)
    auxlogits_v3 = end_points_v3['AuxLogits']

    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits_v4, end_points_v4 = inception_v4.inception_v4(
                x, num_classes = FLAGS.num_classes, is_training = False)
    auxlogits_v4 = end_points_v4['AuxLogits']

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_resnet, end_points_resnet = resnet_v2.resnet_v2_152(
                x, num_classes = FLAGS.num_classes, is_training = False)

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits_Incres, end_points_IR = inception_resnet_v2.inception_resnet_v2(
                x, num_classes = FLAGS.num_classes, is_training = False)
    auxlogits_IR = end_points_IR['AuxLogits']

    logits = (logits_v3 + logits_v4 + logits_resnet + logits_Incres) / 4.0
    auxlogits = (auxlogits_v3 + auxlogits_v4 + auxlogits_IR ) / 3.0
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=1.0)


    noise = tf.gradients(cross_entropy, x)[0]

    # TI-FGSM: https://arxiv.org/pdf/1904.02884.pdf
    # noise = tf.nn.depthwise_conv2d(noise, T_kern, strides=[1, 1, 1, 1], padding='SAME')

    # MI-FGSM: https://arxiv.org/pdf/1710.06081.pdf
    # noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
    # noise = momentum * grad + noise

    # Project cut noise
    amplification += beta * tf.sign(noise)
    cut_noise = tf.clip_by_value(abs(amplification) - eps, 0.0, 10000.0) * tf.sign(amplification)
    projection = gamma * tf.sign(project_noise(cut_noise, P_kern, kern_size))	

    # Occasionally, when the adversarial examples are crafted for an ensemble of networks with residual block by combined methods, 
    # you may neet to comment the following line to get better result.
    amplification += projection

    x = x + beta * tf.sign(noise) + projection
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)
    
    return x, y, i, x_max, x_min, noise, amplification

def stop(x, y, i, x_max, x_min, grad, amplification):
    num_iter = FLAGS.num_iter
    return tf.less(i, num_iter)

def main(_):
    # Because we normalized the input through "input * 2.0 - 1.0" to [-1,1],
    # the corresponding perturbation also needs to be multiplied by 2
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_classes = FLAGS.num_classes
    sum_ensadv_res_v2 = 0
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        x_input = tf.placeholder(tf.float32, shape = batch_shape)
        adv_img = tf.placeholder(tf.float32, shape = batch_shape)
        y = tf.placeholder(tf.int32, shape = batch_shape[0])
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits_ensadv_res_v2, end_points_ensadv_res_v2 = inception_resnet_v2.inception_resnet_v2(
                    adv_img, num_classes = num_classes, is_training = False, scope = 'EnsAdvInceptionResnetV2')
        pre_ensadv_res_v2 = tf.argmax(logits_ensadv_res_v2, 1)

        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)
        amplification = tf.zeros(shape=batch_shape)
        x_adv, _, _, _, _, _, _ = tf.while_loop(stop, graph, [x_input, y, i, x_max, x_min, grad, amplification])

        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        s5 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        s6 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        s7 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
        s8 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))

        with tf.Session() as sess:
            s1.restore(sess, model_checkpoint_map['inception_v3'])
            s5.restore(sess, model_checkpoint_map['inception_v4'])
            s6.restore(sess, model_checkpoint_map['inception_resnet_v2'])
            s7.restore(sess, model_checkpoint_map['ens_adv_inception_resnet_v2'])
            s8.restore(sess, model_checkpoint_map['resnet_v2_152'])

            import pandas as pd
            dev = pd.read_csv(FLAGS.input_csv)
            for idx in tqdm(range(0, 1000 // FLAGS.batch_size)):
                images, filenames, True_label = load_images(FLAGS.input_dir, dev, idx * FLAGS.batch_size, batch_shape)
                my_adv_images = sess.run(x_adv, feed_dict={x_input: images, y: True_label}).astype(np.float32)
                pre_ensadv_res_v2_ = sess.run(pre_ensadv_res_v2,
                                        feed_dict = {adv_img: my_adv_images})
                sum_ensadv_res_v2 += (pre_ensadv_res_v2_ != True_label).sum()
                save_images(my_adv_images, filenames, FLAGS.output_dir)

    print('The success rate = {}'.format(sum_ensadv_res_v2 / 1000.0))


if __name__ == '__main__':
    tf.app.run()
