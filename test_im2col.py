from utils import *

shape = (10, 3, 64, 64)

a = np.random.randn(*shape)

col = im2col(a, stride=4, padding=1)
img = col2im(col, shape, stride=4, padding=1)

print(a == img)
