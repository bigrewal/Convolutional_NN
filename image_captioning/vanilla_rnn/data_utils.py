import h5py
import numpy as np
import os
import json

URL = '../../../assignment3/cs231n/datasets/coco_captioning'

def load_data(pca=True,max_train=None):
    data = {}
    caption_file = os.path.join(URL, 'coco2014_captions.h5')
    with h5py.File(caption_file, 'r') as f:
        for k, v in f.items():
            data[k] = np.asarray(v)

    if pca:
        train_image_fc = os.path.join(URL, 'train2014_vgg16_fc7_pca.h5')
        val_image_fc = os.path.join(URL, 'val2014_vgg16_fc7_pca.h5')
    else:
        train_image_fc = os.path.join(URL, 'train2014_vgg16_fc7.h5')
        val_image_fc = os.path.join(URL, 'val2014_vgg16_fc7.h5')

    with h5py.File(train_image_fc, 'r') as f:
        data['train_features'] = np.asarray(f['features'])

    with h5py.File(val_image_fc, 'r') as f:
        data['val_features'] = np.asarray(f['features'])

    vocab_file = os.path.join(URL, 'coco2014_vocab.json')

    with open(vocab_file, 'r') as f:
        vocab_data = json.load(f)
        for k, v in vocab_data.items():
            data[k] = v

    train_url_file = os.path.join(URL, 'train2014_urls.txt')
    with open(train_url_file, 'r') as f:
        train_urls = np.asarray([line.strip() for line in f])
    data['train_urls'] = train_urls

    val_url_file = os.path.join(URL, 'val2014_urls.txt')
    with open(val_url_file, 'r') as f:
        val_urls = np.asarray([line.strip() for line in f])
    data['val_urls'] = val_urls

    # Maybe subsample the training data
    if max_train is not None:
        num_train = data['train_captions'].shape[0]
        mask = np.random.randint(num_train, size=max_train)
        data['train_captions'] = data['train_captions'][mask]
        data['train_image_idxs'] = data['train_image_idxs'][mask]

    return data

def sample_minibatch(data, batch_size, split='train'):
    split_size = data['%s_captions' % split].shape[0]
    mask = np.random.choice(split_size, batch_size)

    captions = data['%s_captions' % split][mask]
    image_idxs = data['%s_image_idxs' % split][mask]
    image_features = data['%s_features' % split][image_idxs]
    urls = data['%s_urls' % split][image_idxs]
    return captions, image_features, urls

def decode_captions(captions,index_to_word):
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = index_to_word[captions[i, t]]
            if word != '<NULL>':
                words.append(word)
            if word == '<END>':
                break
        decoded.append(' '.join(words))
    if singleton:
        decoded = decoded[0]

    return decoded