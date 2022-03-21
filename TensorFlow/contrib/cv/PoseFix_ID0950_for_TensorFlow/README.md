Deliverable results for GPU training: [README_GPU](1_gpu_training/README_GPU.md) and supporting document in folder [1_gpu_training](1_gpu_training)

Deliverable results for NPU training: [README_NPU](2_npu_training/README_NPU.md) and supporting document in folder [2_npu_training](2_npu_training) 

Deliverable results for NPU infernce: [README_INFERENCE](3_npu_310_inference//README_INFERENCE.md) and supporting document in folder [3_npu_310_inference](3_npu_310_inference) 

### Code changes after using conversion tool:  
| Issue | Code change|
|-------|------------|
|There seems to be conflicts if you use pywrap_tensorflow.NewCheckpointReader to load pretrained model and create a tf variable in your own model that has the same name with the one in the pretrained model but with different shapes.  | Assign the variable created with a different name from the tensor loaded from the pretrained modoel | 
|slicing tensor. Not support pythonic way of slicing tensor. Happens when running sess.run(tf.assign(v[:,:,:3,:],original_v)) where v is the layer in the model with shape (7,7,20,64) and original_v is the layer in the resnet50 with shape (7,7,3,64)| fix this issue by changing sess.run(tf.assign(v[:,:,:3,:],original_v) to sess.run(tf.assign(v[0:7,0:7,0:3,0:64],original_v). 
|tf.scatter_nd does not check indices as it does when using gpu training.| Remove input which exceeds the indice range. |
