#ifndef _CAFFE_UTIL_ROT_HPP_
#define _CAFFE_UTIL_ROT_HPP_

namespace caffe {

#define D_N 8
#define K_W 3

#define W_N 0.5
#define W_M ((1 - sin(M_PI/4)) * (sin(M_PI/4)))
#define W_F ((1 - sin(M_PI/4)) * (1 - sin(M_PI/4)))
#define W_O (2 * (1 - sin(M_PI/4)))

template <typename Dtype>
void rot_weights_gpu(Dtype* F_, const Dtype* F, const int k, const int N, const bool acc, 
    const int channel_out, const int channel_in, const int k_w);

} // namespace caffe

#endif  // CAFFE_UTIL_ROT_HPP_
