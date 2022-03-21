#!/bin/bash

# Download original ShapeNetPart dataset (around 1GB)
wget -P ../part_seg/ https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_v0.zip
unzip ../part_seg/shapenetcore_partanno_v0.zip -d ../part_seg
rm ../part_seg/shapenetcore_partanno_v0.zip

# Download HDF5 for ShapeNet Part segmentation (around 346MB)
wget -P ../part_seg/ https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip
unzip ../part_seg/shapenet_part_seg_hdf5_data.zip -d ../part_seg
rm ../part_seg/shapenet_part_seg_hdf5_data.zip
