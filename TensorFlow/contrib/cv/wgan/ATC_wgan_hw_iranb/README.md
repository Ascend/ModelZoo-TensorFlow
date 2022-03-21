1. prepare data  
    - download lsun bedroom and reshpe to 64, see /root/pr/new_modelzoo/modelzoo/contrib/TensorFlow/Research/cv/wgan/README.md 
    - use test_om.sh get output feature 
2. test 
    - convert ckpt to pb , see ckpt2pb.py 
    - convert pb file to om file ,see pb2om.sh 
    - test your data , see test.py

note: the ckpt data , pb data, and om, input and output data can be found in obs://ma-iranb/pr/atc_wgan/

MEAN SS: **2.9 * e-5**