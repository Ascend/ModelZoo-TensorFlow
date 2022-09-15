# Pixel Recursive Super Resolution

TensorFlow implementation of [Pixel Recursive Super Resolution](https://arxiv.org/abs/1702.00783).

## Requirements

- Python 3.7
- [Skimage](http://scikit-image.org/)
- [TensorFlow](https://www.tensorflow.org/) 1.15


## Usage

First, download data [celebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

    $ mkdir data
	$ cd data
	$ ln -s $celebA_path celebA

Then, create image_list file:

	$ python tools/create_img_lists.py --dataset=data/celebA --outfile=data/train.txt


To train model on npu:

	$ python train.py

# offline inference
## 1. inference ckpt
	$ python3 inference.py

## 2. inference pb
### a. convert ckpt to pb
	$ python3 ckpt2pb.py
### b. inference pb model
	$ python3 pb_inference.py

## 3. inference om
### a. convert pb to om
	$ /usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=./pb_model/prsr_conditioning.pb --framework=3 --output=./pb_model/prsr_base --soc_version=Ascend910 --input_shape="input1:1,32,32,3;input2:1,8,8,3" --log=info --out_nodes="p_logits:0;c_logits:0"

get the om model ./pb_model/prsr_base.om
### b. generate input bin file
	$ python3 ./tools/img2bin/img2bin.py -i ./input_img/ -w 32 -h 32 -a NHWC -t float32 -o ./input1/
    $ python3 ./tools/img2bin/img2bin.py -i ./input_img/ -w 8 -h 8 -a NHWC -t float32 -o ./input2/

### c. inference om model
	$ export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
    $ export NPU_HOST_LIB=/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64/stub

    $ ./tools/msame/out/msame --model ./pb_model/prsr_base.om --input ./input1/,./input2/ --output ./om_output/
now we get the output of om in ./om_output/xxxxx/xxx.bin, then we calculate PSNR between output and input

    $ python3 bin2np.py
    $ python3 psnr.py

# Comparison of inference PSNR
ckpt: 17.4\
pb: 16.4\
om: 14.4