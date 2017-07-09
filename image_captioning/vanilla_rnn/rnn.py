import numpy as np

def step_forward(x,Wx,prev_h,Wh,b):

    #next_h = tanh(Wh*prev_h + x*Wx +bias)
    forward = np.dot(prev_h, Wh) + np.dot(x, Wx) + b
    next_h = np.tanh(forward)

    cache = (x,Wx,prev_h,Wh,b,next_h,forward)

    return next_h, cache


def step_backward(d_next_h, cache):
    x, Wx, prev_h, Wh, b, next_h, forward = cache

    dforward = (sech(forward) ** 2) * d_next_h

    dx = np.dot(dforward, Wx.T)
    dprev_h = np.dot(dforward, Wh.T)
    dWx = np.dot(x.T, dforward)
    dWh = np.dot(prev_h.T, dforward)
    db = np.sum(dforward, axis=0)

    return dx, dprev_h, dWx, dWh, db

def sech(x):
    return 1./np.cosh(x)

def rnn_forward(x, h0, Wx, Wh, b):
    h, cache = None, None
    N, T, D = x.shape
    N, H = h0.shape

    h = np.zeros((T, N, H))
    x = x.transpose(1, 0, 2)
    cache = []

    for t in range(T):
        if t == 0:
            prev_h = h0
        else:
            prev_h = h[t - 1]
        next_h, n_cache = step_forward(x[t], Wx, prev_h, Wh, b)
        cache.append(n_cache)
        h[t] = next_h

    h = h.transpose(1, 0, 2)
    return h,cache

def rnn_backward(dh,cache):

    N, T, H = dh.shape
    D = cache[0][0].shape[1]

    # Initialize dx,dh0,dWx,dWh,db
    dx = np.zeros((T, N, D))
    dh0 = np.zeros((N, H))
    db = np.zeros((H))
    dWh = np.zeros((H, H))
    dWx = np.zeros((D, H))

    # On transpose dh
    dh = dh.transpose(1, 0, 2)
    dh_prev = np.zeros((N, H))

    for t in reversed(range(T)):
        dh_current = dh[t] + dh_prev
        dx_t, dh_prev, dWx_t, dWh_t, db_t = step_backward(dh_current, cache[t])
        dx[t] += dx_t
        dh0 = dh_prev
        dWx += dWx_t
        dWh += dWh_t
        db += db_t

    dx = dx.transpose(1, 0, 2)

    return dx, dh0, dWx, dWh, db

def word_embedding_forward(x, W):
    out = W[x, :]
    cache = x, W

    return out, cache

def word_embedding_backward(dout, cache):
    x, W = cache
    dW = np.zeros_like(W)
    np.add.at(dW, x, dout)

    return dW

def temporal_affine_forward(x, w, b):
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out

    return out, cache

def temporal_affine_backward(dout, cache):

    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db

def temporal_softmax_loss(x, y, mask, verbose=False):

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
