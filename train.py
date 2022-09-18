#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(
    description='Train Johnson (2016) style transfer model.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('style_image', help='style image file')
parser.add_argument('saved_model', help='where to save the trained model')

parser.add_argument('--content_weight', type=float, default=1.0,
                    help='content weight')
parser.add_argument('--style_weight', type=float, default=1e-4,
                    help='style weight')
parser.add_argument('--var_weight', type=float, default=1e-6,
                    help='variation weight')
parser.add_argument('--normalization',
                    choices=['batch', 'instance'], default='batch',
                    help='type of normalization' )
parser.add_argument('--epochs', type=int, default=2,
                    help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=4,
                    help='training batch size')

parser.add_argument('--data_dir', default='/tmp',
                    help='dataset directory - requires ~120gb')

args = parser.parse_args()


from os import environ as env
env['TF_CPP_MIN_LOG_LEVEL'] = '2'               # hide info & warnings
env['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'       # grow GPU memory as needed

import tensorflow as tf
import tensorflow_datasets as tfds

from nstesia.io import load_image
from nstesia.johnson_2016 import StyleTransferModel


train_ds = tfds.load('coco/2014', split='train', data_dir=args.data_dir)
train_ds = train_ds.map( lambda data: tf.cast(data['image'], dtype=tf.float32) )
train_ds = train_ds.map( lambda image: tf.image.resize(image, [256,256]) )
train_ds = train_ds.batch(args.batch_size, drop_remainder=True)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)


style_image = load_image(args.style_image)

transfer_model = StyleTransferModel(
    style_image,
    normalization=args.normalization,
    content_weight=args.content_weight,
    style_weight=args.style_weight,
    var_weight=args.var_weight,
)
transfer_model.compile(optimizer='adam')
transfer_model.fit(train_ds, epochs=args.epochs)

transfer_model.save(args.saved_model, save_traces=False)
