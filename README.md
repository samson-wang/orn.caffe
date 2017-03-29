# orn.caffe
A test implementation of Oriented Response Networks on caffe

A try to implement [Oriented Response Networks](https://arxiv.org/abs/1701.01833). The official release will come out soon @ [ORN](https://zhouyanzhao.github.io/ORN/)

The `test_orn.py` tries to simulate roatation of filter. For 3x3-size and 8-orientation kernel, there is a fast computation.

Based on the `test_orn.py`, a cpu version of OrientedConvolution layer was implemented. A test on the MNIST dataset showed simliar result to the paper.

 type | paper | this 
--- | --- | ---
 Base line | 0.9927 | 0.9911 
 ORN-8 None | 0.9921 | 0.9903 
 
 
I will add the ORAlign and ORPooling, as well as a gpu version, later. Hope no serious mistake here.
