'''
A baseline for ARF and ORN computation.
'''

import numpy as np

def print_mat(mat):
    h, w = mat.shape[:2]
    for j in range(h):
        for i in range(w):
            print mat[j][i],
        print

std_coord = np.array(
    [
        [[-1, 1], [0, 1], [1, 1]],
        [[-1, 0], [0, 0], [1, 0]],
        [[-1,-1], [0,-1], [1,-1]]
    ]
)
print std_coord, std_coord.shape

h, w, _ = std_coord.shape
for j in range(h):
    for i in range(w):
        print (j, i),
    print

print_mat(std_coord)

def init_coord(w=3):
    pass

def orig_coord(dst_coord, k, N):
    '''
    For a dst coord, compute the origin coord that generate it by rotating (2 * pi * k / N)
    '''
    src_coord = dst_coord.copy()

    tmp = tr_mat(k, N)

    for i in src_coord:
        for j in i:
            print np.matmul(j, tmp),
            j[...] = np.matmul(j, tmp)
        print
    return src_coord

def interpolation(src_coord, F):
    '''
    Given source coordinates and Filter matrix compute corresponding values for source coordinates
    '''
    F_ = F.copy() 
    h, w, _ = src_coord.shape
    for j in range(h):
        for i in range(w):
            q_, p_ = src_coord[j][i]
            if p_ == np.floor(p_) and q_ == np.floor(q_):
                print (1-p_, 1+q_),
                F_[j][i] = F[1-p_][1+q_]

        print

    print_mat(F)
    print_mat(F_)

    return F_

def tr_mat(k, N):
    '''
    k-th rotate matrix for N directions
    '''
    # cosine/sine for 8 
    cos_8 = np.array([1., np.cos(np.pi/4), 0., np.cos(3*np.pi/4), -1., np.cos(5*np.pi/4), 0., np.cos(7*np.pi/4)])
    sin_8 = np.array([0., np.sin(np.pi/4), 1., np.sin(3*np.pi/4), 0., np.sin(5*np.pi/4), -1., np.sin(7*np.pi/4)])

    assert(N==4 or N==8)

    k = 8 / N * k
    print k

    return np.array([[cos_8[k], sin_8[k]], [-sin_8[k], cos_8[k]]])

# An Active Rotating Filter is a weight matrix in shape of (W x W x N)
# Like traditional Convolution Filter, W is the width of the filter.
# N is the number of rotations. Typical 4 or 8.
def ARF(object):
    def __init__(self, w=3, n=4):
        '''
        Input:
            w(int): width of the filter
            n(int): number of rotating directions
        '''
        self.w = w
        self.n = n

    def coord_rot(self):
        pass


if __name__ == '__main__':
    print tr_mat(3, 4).shape, tr_mat(1,4)
    print np.matmul(np.array([0,1]).reshape((1,2)), tr_mat(1,4))
    print interpolation(orig_coord(std_coord, 2, 4), np.arange(9).reshape((3,3)))
