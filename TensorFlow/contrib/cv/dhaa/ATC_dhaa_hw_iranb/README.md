1. Get dataset  
See modelzoo/contrib/TensorFlow/Research/cv/dhaa/dhaa_tf_hw80211537/README.md

2. Convert dataset to bin file   
    - ``cd image2bin`` 
    - Change the out_path in image2bin.py
    - Run ``python3.7.5 image2bin.py``
3. Download the pretrained ckpt file for dhaa  
See modelzoo/contrib/TensorFlow/Research/cv/dhaa/dhaa_tf_hw80211537/README.md

4. Convert ckpt file to om file  
    - First, see **ckpt2pb.py** , change the **ckpt_path** and  **output_graph**
    - Second, see **pb2om.sh**, convert the pb file to om file
5. Test  
for test we only support batch size 1 ,this will loss some accuracy 
    - Change **test_dataset_path** to original test label file
    - Change **test_bin_path** to your bin data file dir 
    - run ``python 3.7.5 test.py``


Also, you can download the converted om file and bin file from : obs://ma-iranb/pr/atc_dhaa
you can download the ckpt file at obs://ma-iranb/pr/arc_dhaa_all/LAJ_DHAA/Jobs_morph/models/2021-5-25
the dataset files include morph_pts_80-20_test.txt test label file at obs://ma-iranb/pr/arc_dhaa_all/LAJ_DHAA/morph/txt/pts_txt/morph_pts_80-20_test.txt

the 
```
====================
MAE RESULT 33.26789
====================
```