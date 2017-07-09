import numpy as np
from model import CaptioningRNN
from data_utils import *
from image_utils import *
from rnn import *
from solver import *
import matplotlib.pyplot as plt

np.random.seed(231)

small_data = load_data(max_train=100)

#Overfit the model to check if everything is working
small_rnn_model = CaptioningRNN(
          cell_type='rnn',
          word_to_idx=small_data['word_to_idx'],
          input_dim=small_data['train_features'].shape[1],
          hidden_dim=512,
          wordvec_dim=256,
        )

small_rnn_solver = CaptioningSolver(small_rnn_model, small_data,
           num_epochs=50,
           batch_size=25,
           optim_config={
             'learning_rate': 5e-3,
           },
           lr_decay=0.95,
           verbose=True, print_every=10,
         )

small_rnn_solver.train()


#Run the trained image captioning model

# ===============================================================================

# for split in ['train', 'val']:
#     minibatch = sample_minibatch(small_data, split=split, batch_size=2)
#     gt_captions, features, urls = minibatch
#     gt_captions = decode_captions(gt_captions, small_data['idx_to_word'])
#
#     sample_captions = small_rnn_model.sample(features)
#     sample_captions = decode_captions(sample_captions, small_data['idx_to_word'])
#
#     for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
#         plt.imshow(image_from_url(url))
#         plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
#         plt.axis('off')
#         plt.show()

# ===============================================================================