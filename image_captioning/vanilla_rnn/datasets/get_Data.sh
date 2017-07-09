#!/bin/bash
wget http://cs231n.stanford.edu/imagenet_val_25.npz

wget "http://cs231n.stanford.edu/coco_captioning.zip"
unzip coco_captioning.zip
rm coco_captioning.zip

wget "http://cs231n.stanford.edu/squeezenet_tf.zip"
unzip squeezenet_tf.zip
rm squeezenet_tf.zip
