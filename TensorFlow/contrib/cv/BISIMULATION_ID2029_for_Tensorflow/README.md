### BISIMULATION

##### Requirements
 - python 3.7.5
 - Tensorflow (tested with v1.15.0)
 - Huawei Ascend
 
 Run
```
pip install -r pip-requirements.txt
```

### Download datasets
 - configs
 ```
  obs://gaten/dataset/
 ```

### Run
```
python3.7.5 -m compute_metric \
  --base_dir=/tmp/grid_world \
  --grid_file=configs/mirrored_rooms.grid \
  --gin_files=configs/mirrored_rooms.gin
```

### 精度


### 性能