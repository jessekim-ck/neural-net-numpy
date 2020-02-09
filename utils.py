import numpy as np

"""
Referenced to "밑바닥부터 시작하는 딥러닝" by O'Reilly

Github repository:
https://github.com/WegraLee/deep-learning-from-scratch
"""

def im2col(image, filter_h=3, filter_w=3, stride=1, padding=0):
    
    # Calculate col shape
    num_data, num_channels, img_h, img_w = image.shape
    window_h = (img_h + 2*padding - filter_h)//stride + 1
    window_w = (img_w + 2*padding - filter_w)//stride + 1

    # Add padding
    image = np.pad(image, [(0, 0), (0, 0), (padding, padding), (padding, padding)], 'constant')

    # Initialize col.
    col = np.zeros((num_data, num_channels, filter_h, filter_w, window_h, window_w))

    # Let's fill each box!
    for y in range(filter_h):
        y_max = y + stride*window_h
        for x in range(filter_w):
            x_max = x + stride*window_w
            # Pick (y, x) elements of every window.
            col[:, :, y, x, :, :] = image[:, :, y:y_max:stride, x:x_max:stride]
    
    # (num_data, window_h, window_w, num_channels, filter_h, filter_w)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(num_data*window_h*window_w, -1)

    return col


def col2im(col, img_shape, filter_h=3, filter_w=3, stride=1, padding=0):

    num_data, num_channels, img_h, img_w = img_shape
    window_h = (img_h + 2*padding - filter_h)//stride + 1
    window_w = (img_w + 2*padding - filter_w)//stride + 1

    col = col.reshape(num_data, window_h, window_w, num_channels, filter_h, filter_w)

    # (num_data, num_channels, filter_h, filter_w, window_h, window_w)
    col = col.transpose(0, 3, 4, 5, 1, 2)

    # Initialize padded image
    img = np.zeros((num_data, num_channels, img_h + 2*padding, img_w + 2*padding))

    for y in range(filter_h):
        y_max = y + stride*window_h
        for x in range(filter_w):
            x_max = x + stride*window_w
            img[:, :, y:y_max:stride, x:x_max:stride] = col[:, :, y, x, :, :]

    img = img[:, :, padding:img_h + padding, padding:img_w + padding]
    
    return img
