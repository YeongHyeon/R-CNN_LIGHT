# coding: utf-8
import numpy as np

def smooth_curve(x):
    """ Used to smooth the graph of the loss function

    reference:http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


def shuffle_dataset(x, t):
    """
    Parameters
    ----------
    x : Image
    t : Label

    Returns
    -------
    x, t : Image and Label
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t

def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size + 2*pad - filter_size) / stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : 4d array (N, C, H, W)
    filter_h : filter height
    filter_w : filter width
    stride
    pad

    Returns
    -------
    col : 2d array (N, Data)
    """

    N, C, H, W = input_data.shape
    out_h = conv_out_size(H, filter_h, pad, stride)
    out_w = conv_out_size(W, filter_w, pad, stride)
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    # y: filter height index
    # x: filter width index
    for y in range(filter_h): # 5
        y_max = y + stride*out_h
        for x in range(filter_w): # 5
            x_max = x + stride*out_w
            # get y to y_max (stride(interval) = 1)
            # get x to x_max (stride(interval) = 1)
            # 0 to 24 and (0 to 24, 1 to 25, 2 to 26, 3 to 27, 4 to 28)
            # 1 to 25 and (0 to 24, 1 to 25, 2 to 26, 3 to 27, 4 to 28)
            # 2 to 26 and (0 to 24, 1 to 25, 2 to 26, 3 to 27, 4 to 28)
            # 3 to 27 and (0 to 24, 1 to 25, 2 to 26, 3 to 27, 4 to 28)
            # 4 to 28 and (0 to 24, 1 to 25, 2 to 26, 3 to 27, 4 to 28)
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # col.transpose(0, 4, 5, 1, 2, 3):
    # (N, C, filter_h, filter_w, out_h, out_w) -> (N, out_h, out_w, C, filter_h, filter_w)
    # ------------------------------------------------------------------------------------
    # reshape(N*out_h*out_w, -1):
    # (N, out_h, out_w, C, filter_h, filter_w) -> (N * out_h * out_w, C * filter_h * filter_w)
    # -1 means auto shaping (others)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)

    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : input data (example: (10, 1, 28, 28))
    filter_h : filter height
    filter_w : filter width
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

def conv_out_size(input_size, filter_size, filter_pad, filter_stride):
    #output_height = ((input_size + 2*filter_pad - filter_size) / filter_stride) + 1
    # deep learning from scratch p.234
    return ((input_size - filter_size + 2*filter_pad) / filter_stride) + 1

def pool_out_size(conv_output_size, filter_pad, pool_size):
    return (conv_output_size-2*filter_pad)/pool_size
