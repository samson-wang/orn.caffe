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
    ], dtype=np.float32
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
#    print 'In Orig'
    src_coord = dst_coord.copy()

    tmp = tr_mat(k, N)

    for i in src_coord:
        for j in i:
#            print np.matmul(j, tmp),
            j[...] = np.matmul(j, tmp)
#        print
    print 'Out Orig',
    print_mat(src_coord)
#    print_mat(dst_coord)
    return src_coord

def interpolation(src_coord, F):
    '''
    Given source coordinates and Filter matrix compute corresponding values for source coordinates
    '''
    F_ = F.copy() 
    h_, w_, _ = src_coord.shape
    print 'h_, w', h_, w_
    for j in range(h_):
        for i in range(w_):
            q_, p_ = src_coord[j][i]
#            print 'p,q', i, p_, np.floor(p_), p_ == np.floor(p_)
            u = np.floor(p_)
            v = np.floor(q_)
            y = p_ - u
            w = q_ - v
#            print ((u, v), (u, v+1), (u+1, v), (u+1, v+1)), ((1-y)*(1-w), (1-y)*w, y*(1-w), y*w)
            coord = [(u, v), (u, v+1), (u+1, v), (u+1, v+1)]
            weight = [(1-y)*(1-w), (1-y)*w, y*(1-w), y*w]
            tmp = 0.
            for k, (x0, y0) in enumerate(coord):
                x0 = 1 - int(x0)
                y0 = 1 + int(y0)
                if x0 > -1 and x0 < w_ and y0 > -1 and y0 < h_:
                    print (x0, y0), weight[k],
                    tmp += F[x0][y0] * weight[k]
            print '||||\t',
#            print
#            if p_ == np.floor(p_) and q_ == np.floor(q_):
#                print (1.-p_, 1.+q_), (p_, q_), ()
            F_[j][i] = tmp      #F[1-p_][1+q_]

        print '\\r', j, i

    print_mat(F)
    print_mat(F_)

    return F_

def spin():
    '''
    Just shift the channels
    '''
    pass

def tr_mat(k, N):
    '''
    k-th rotate matrix for N directions
    '''
    # cosine/sine for 8 
    cos_8 = np.array([1., np.cos(np.pi/4), 0., np.cos(3*np.pi/4), -1., np.cos(5*np.pi/4), 0., np.cos(7*np.pi/4)])
    sin_8 = np.array([0., np.sin(np.pi/4), 1., np.sin(3*np.pi/4), 0., np.sin(5*np.pi/4), -1., np.sin(7*np.pi/4)])

    assert(N==4 or N==8)

    k = 8 / N * k
    print 'Rotated: ', -k

    return np.array([[cos_8[k], sin_8[k]], [-sin_8[k], cos_8[k]]])

def f_orig(F, K, N):
    '''
    Fast computing for interpolation
    '''

    # For  interpolation in four surrounding points
    # Weight for near point
    W_N = 0.5
    # Weight for middle range point
    W_M = (1 - np.sin(np.pi/4)) * np.sin(np.pi/4)
    # Weight for far point
    W_F = (1 - np.sin(np.pi/4)) * (1 - np.sin(np.pi/4))
    # Weight for outside point
    W_O = 2. * (1 - np.sin(np.pi/4))
    print W_N, W_M, W_F, W_O

    # For first rotation on 2 * pi / 8 When N = 8
    # Iteration coord
    ITER_COORD = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
    # coord for row major store
    ROT_COORD = [[(1, 0)], [(1, 0), (1, 1), (0, 0), (0, 1)], [(0, 1)], [(1, 1), (1, 2), (0, 1), (0, 2)], [(1, 2)], [(2, 1), (2, 2), (1, 1), (1, 2)], [(2, 1)], [(2, 0), (2, 1), (1, 0), (1, 1)]]
    # weight for row major store
    WEIGHT = [[W_O], [W_M, W_F, W_N, W_M], [W_O], [W_F, W_M, W_M, W_N], [W_O], [W_M, W_N, W_F, W_M], [W_O], [W_N, W_M, W_M, W_F]]
    assert(len(ITER_COORD) == len(ROT_COORD))
    assert(len(ITER_COORD) == len(WEIGHT))

    F_ = F.copy()
    if N == 4:
        N = 8
        K = K * 2

    if K % 2 == 0:
        for k, i in enumerate(ITER_COORD):
            y, x = i
            Y, X = ITER_COORD[(k-K)%N]
            F_[y][x] = F[Y][X]
    else:
        for k, i in enumerate(ITER_COORD):
            y, x = i
            tmp = 0
            for m, j in enumerate(ROT_COORD[(k-K+1) % N]):
                tmp += F[j[0]][j[1]] * WEIGHT[(k-K+1) % N][m]

            F_[y][x] = tmp

    return F_

def f_orig_pi_2(F, K, N):
    '''
    Fast computing for pi / 2 rotation
    '''
    w_ = 3
    ITER_COORD = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]

    F_ = F.copy()

    for k, v in enumerate(ITER_COORD):
        y, x = v
        print (k - 2*K) % N
        Y, X = ITER_COORD[(k - 2*K) % N]

        F_[y][x] = F[Y][X]
    
    return F_

def test_rot_spin():
    N = 8
    F = np.arange(N*9).reshape(N, 3, 3).astype(np.float32)
    print F
    F_ = F.copy()
    for n in range(N):
      for k in range(N):
        F_[k][...] = f_orig(F[(k-n) % N], 4, N)
      print n, 'Rotate', F_
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
    for i in range(1,3):
        interpolation(orig_coord(std_coord, i, 8), np.arange(9).reshape((3,3)).astype(np.float32))

    print 'interpolation'
    interpolation(orig_coord(std_coord, -2, 8), np.arange(9).reshape((3,3)).astype(np.float32))
    print "f_orig"
    print_mat(f_orig(np.arange(9).reshape((3,3)).astype(np.float32), -2, 8))
    print "orig_2"
    print_mat(f_orig_pi_2(np.arange(9).reshape((3,3)).astype(np.float32), -2/2, 8))

    test_rot_spin()
