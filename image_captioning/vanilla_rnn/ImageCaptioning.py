import numpy as np
from model import CaptioningRNN
from data_utils import *
from image_utils import *
from rnn import *
from solver import *
import matplotlib.pyplot as plt

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

data = load_data()
# for k,v in data.items():
#     if type(v) == np.ndarray:
#         print(k,type(v),v.shape)
#     else:
#         print(k,type(v),len(v))

# captions,image_index,features,urls = sample_minibatch(data,batch_size=3)

# for i, (caption, url) in enumerate(zip(captions, urls)):
#     plt.imshow(image_from_url(url))
#     plt.axis('off')
#     caption_str = decode_captions(caption, data['idx_to_word'])
#     plt.title(caption_str)
#     plt.show()


#CHECKING ERROR IN STEP FORWARD
# N, D, H = 3, 10, 4
#
# x = np.linspace(-0.4, 0.7, num=N*D).reshape(N, D)
# Wx = np.linspace(-0.1, 0.9, num=D*H).reshape(D, H)
# prev_h = np.linspace(-0.2, 0.5, num=N*H).reshape(N, H)
# Wh = np.linspace(-0.3, 0.7, num=H*H).reshape(H, H)
# b = np.linspace(-0.2, 0.4, num=H)
#
# next_h, _ = step_forward(x,Wx,prev_h,Wh,b)
# expected_next_h = np.asarray([
#   [-0.58172089, -0.50182032, -0.41232771, -0.31410098],
#   [ 0.66854692,  0.79562378,  0.87755553,  0.92795967],
#   [ 0.97934501,  0.99144213,  0.99646691,  0.99854353]])
#
# print('next_h error: ', rel_error(expected_next_h, next_h))

#CHECKING ERROR IN STEP BACKWARD
#---- code needs to be completed for numerical gradient check, come back later and do it

# N, D, W, H = 10, 20, 30, 40
# word_to_idx = {'<NULL>': 0, 'cat': 2, 'dog': 3}
# V = len(word_to_idx)
# T = 13
#
# model = CaptioningRNN(word_to_idx,
#           input_dim=D,
#           wordvec_dim=W,
#           hidden_dim=H,
#           cell_type='rnn',
#           dtype=np.float64)
#
# # Set all model parameters to fixed values
# for k, v in model.params.items():
#     model.params[k] = np.linspace(-1.4, 1.3, num=v.size).reshape(*v.shape)
#
# features = np.linspace(-1.5, 0.3, num=(N * D)).reshape(N, D)
# captions = (np.arange(N * T) % V).reshape(N, T)
#
# loss, grads = model.loss(features, captions)
# expected_loss = 9.83235591003
#
# print('loss: ', loss)
# print('expected loss: ', expected_loss)
# print('difference: ', abs(loss - expected_loss))

np.random.seed(231)

small_data = load_data(max_train=50)

small_rnn_model = CaptioningRNN(
          cell_type='rnn',
          word_to_idx=data['word_to_idx'],
          input_dim=data['train_features'].shape[1],
          hidden_dim=512,
          wordvec_dim=256,
        )

small_rnn_solver = CaptioningSolver(small_rnn_model, small_data,
           update_rule='adam',
           num_epochs=50,
           batch_size=25,
           optim_config={
             'learning_rate': 5e-3,
           },
           lr_decay=0.95,
           verbose=True, print_every=10,
         )

small_rnn_solver.train()

# Plot the training losses
# plt.plot(small_rnn_solver.loss_history)
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.title('Training loss history')
# plt.show()

for split in ['train', 'val']:
    minibatch = sample_minibatch(small_data, split=split, batch_size=2)
    gt_captions, features, urls = minibatch
    gt_captions = decode_captions(gt_captions, data['idx_to_word'])

    sample_captions = small_rnn_model.sample(features)
    sample_captions = decode_captions(sample_captions, data['idx_to_word'])

    for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
        plt.imshow(image_from_url(url))
        plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
        plt.axis('off')
        plt.show()