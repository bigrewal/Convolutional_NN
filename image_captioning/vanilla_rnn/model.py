import numpy as np
from rnn import *
class CaptioningRNN(object):


    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                 hidden_dim=128, cell_type='rnn', dtype=np.float32):

        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)

        # Initialize word vectors
        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
        self.params['W_embed'] /= 100

        # Initialize CNN -> hidden state projection parameters
        self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
        self.params['W_proj'] /= np.sqrt(input_dim)
        self.params['b_proj'] = np.zeros(hidden_dim)

        # Initialize parameters for the RNN
        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params['Wx'] /= np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(dim_mul * hidden_dim)

        # Initialize output to vocab weights
        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
        self.params['W_vocab'] /= np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)


    def loss(self, features, captions):
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # You'll need this
        mask = (captions_out != self._null)

        # Weight and bias for the affine transform from image features to initial
        # hidden state
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']

        # Word embedding matrix
        W_embed = self.params['W_embed']

        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

        # Weight and bias for the hidden-to-vocab transformation.
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        loss, grads = 0.0, {}

        # Step 1
        h0 = np.dot(features, W_proj) + b_proj

        # Step 2
        x, cache_embedding = word_embedding_forward(captions_in, W_embed)

        # Step 3
        if self.cell_type == 'rnn':
            h, cache_rnn = rnn_forward(x, h0, Wx, Wh, b)
        elif self.cell_type == 'lstm':
            h, cache_rnn = lstm_forward(x, h0, Wx, Wh, b)
        else:
            raise ValueError('%s not implemented' % (self.cell_type))

        # Step 4
        scores, cache_scores = temporal_affine_forward(h, W_vocab, b_vocab)

        # Step 5
        loss, dscores = temporal_softmax_loss(
            scores, captions_out, mask, verbose=False)

        # Backward pass
        grads = dict.fromkeys(self.params)

        # Backaward into step 4
        dh, dW_vocab, db_vocab = temporal_affine_backward(
            dscores, cache_scores)

        # Backward into step 3
        if self.cell_type == 'rnn':
            dx, dh0, dWx, dWh, db = rnn_backward(dh, cache_rnn)
        elif self.cell_type == 'lstm':
            dx, dh0, dWx, dWh, db = lstm_backward(dh, cache_rnn)
        else:
            raise ValueError('%s not implemented' % (self.cell_type))

        # Backward into step 2
        dW_embed = word_embedding_backward(dx, cache_embedding)

        # Backward into step 1
        dW_proj = np.dot(features.T, dh0)
        db_proj = np.sum(dh0, axis=0)

        # Gather everythhing in the dict
        grads['W_proj'] = dW_proj
        grads['b_proj'] = db_proj
        grads['W_embed'] = dW_embed
        grads['Wx'] = dWx
        grads['Wh'] = dWh
        grads['b'] = db
        grads['W_vocab'] = dW_vocab
        grads['b_vocab'] = db_vocab


        return loss, grads


    def sample(self, features, max_length=30):
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        # Unpack parameters
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        h0 = np.dot(features, W_proj) + b_proj

        captions[:, 0] = self._start
        prev_h = h0  # Previous hidden state
        prev_c = np.zeros_like(h0)  # Previous cell state
        # Current word (start word)
        capt = self._start * np.ones((N, 1), dtype=np.int32)

        for t in range(max_length):  # Let's go over the sequence

            word_embed, _ = word_embedding_forward(
                capt, W_embed)  # Embedded current word
            if self.cell_type == 'rnn':
                # Run a step of rnn
                h, _ = step_forward(np.squeeze(
                    word_embed), Wx,prev_h, Wh, b)
            elif self.cell_type == 'lstm':
                # Run a step of lstm
                h, c, _ = lstm_step_forward(np.squeeze(
                    word_embed), prev_h, prev_c, Wx, Wh, b)
            else:
                raise ValueError('%s not implemented' % (self.cell_type))

            # Compute the score distrib over the dictionary
            scores, _ = temporal_affine_forward(
                h[:, np.newaxis, :], W_vocab, b_vocab)
            # Squeeze unecessari dimension and get the best word idx
            idx_best = np.squeeze(np.argmax(scores, axis=2))
            # Put it in the captions
            captions[:, t] = idx_best

            # Update the hidden state, the cell state (if lstm) and the current
            # word
            prev_h = h
            if self.cell_type == 'lstm':
                prev_c = c
            capt = captions[:, t]

        return captions
