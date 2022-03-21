Author: Heiko J Schick ([heiko.schick@huawei.com]())

# Changelog

## [0.6.0] - 2021-05-12
### Added
- Optimized inference execution time (improvement of 46%)

### Changed
- Refactored model conversion and inference to be aligned with the training

### Fixed
- Fixed bug in inference when using "sparse" model

## [0.4.0] - 2021-04-06
### Added
- Inference pipeline for Ascend 310 using "cora-full" dataset.
- NPU compatible sparse implemention of GCN training.
- Document structure for final report.


## [0.3.0] - 2021-03-22
### Added
- Training with “core-full” dataset on CPU and GPU.
- Inference pipeline for Ascend 310 using for "cora dataset.


## [0.2.0] - 2021-03-04
### Added
- Usage of `tf.sparse` API which reduces training time up to 4x.

### Changed
- Reduced number of used operators by 14.3%.
- Reduced cycles of most used operators by 5.9%.  


## [0.1.0] - 2021-02-19
### Added
- Network training on CPU, GPU and NPU (Ascend 910).
- Output trained model in checkpoint and protobuf format.
- Required Python libaries are listed in requirements.txt.

### Changed
- Configuration parameters are parsed via command line.

### Fixed
- Source code is compliant with *pylint*.


## [0.0.1] - 2021-02-01
### Added
- Initial GCN release using TensorFlow 1.15.