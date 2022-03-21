1. Get dataset and model ckpt files  
See modelzoo/contrib/TensorFlow/Research/cv/jlpls-paa/jlpls-paa_tf_hw80211537/README.md 
2. Get converted test bin file   
Dwonload from obs://ma-iranb/pr/atc_jspjaa/LAJ_PED/data_bin/test
3. Convert ckpt to om file 
    - First run **ckpt2pb.py**
    - Second run **pb2om_jspjaa.sh**

4. Run test
    - run ``bash Inference.sh ``

You can also get the om file and bin file on obs://ma-iranb/pr/atc_jspjaa/LAJ_PED

results:   
accuracy_mean: 50.00    
instance_accuracy: 40.61        
instance_recall: 48.08  
instance_precision: 67.16       
instance_F1: 56.04


note:  
For convert your own data to bin, youp can refer **test_PadAttr_bare_pb.py**, uncomet the code

```python

'''
                # =============================
                inner_path = filename_batch[0].decode(encoding='UTF-8').split('/')[-3:]
                inner_path[-1] = inner_path[-1].split('.')[0]
                out_path = os.path.join('./data_bin/{}/'.format(args.phase), '_'.join(inner_path) + ".bin")
                bin_path = os.path.split(out_path)[0]
                if not os.path.exists(bin_path):
                    os.makedirs(bin_path)
                try:
                    image_batch.tofile(out_path)
                except Exception as err:
                    print(out_path)
                    print(type(image_batch))
                    print('Error: '+str(err))
                    input()
                # print(inner_path)
                # input()
                # =============================
'''
```