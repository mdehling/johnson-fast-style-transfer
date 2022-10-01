#!/usr/bin/env python

import argparse

from os import environ as env
env['TF_CPP_MIN_LOG_LEVEL'] = '2'               # hide info & warnings
env['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'       # grow GPU memory as needed

import tensorflow as tf
import tensorflow_datasets as tfds

from nstesia.io import load_image
from nstesia.johnson_2016 import StyleTransferModel


def parse_args():
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
                        choices=['batch', 'instance'], default='instance',
                        help='type of normalization' )
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='training batch size')

    parser.add_argument('--data_dir', default='/tmp',
                        help='dataset directory - requires ~120gb')

    return parser.parse_args()


def get_train_ds(data_dir='/tmp', batch_size=4):
    ds = tfds.load('coco/2014', split='train', data_dir=data_dir)
    ds = ds.map( lambda data: tf.cast(data['image'], dtype=tf.float32) )
    ds = ds.map( lambda image: tf.image.resize(image, [256,256]) )
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(tf.data.AUTOTUNE)


def train_model(
    style_image_file,
    normalization,
    content_weight, style_weight, var_weight,
    train_ds, epochs,
):
    style_image = load_image(style_image_file)

    model = StyleTransferModel(
        style_image,
        normalization=normalization,
        content_weight=content_weight,
        style_weight=style_weight,
        var_weight=var_weight,
    )
    model.compile(optimizer='adam')
    model.fit(train_ds, epochs=epochs)

    return model


if __name__ == '__main__':

    args = parse_args()

    train_ds = get_train_ds(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    model = train_model(
        args.style_image,
        args.normalization,
        args.content_weight, args.style_weight, args.var_weight,
        train_ds, args.epochs,
    )

    model.save(args.saved_model, save_traces=False)
