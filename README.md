# Tensorflow Pretrained Dual Path Networks (DPN)

This repository contains an implementation of Dual-Path-Networks ((https://arxiv.org/abs/1707.01629) with cypw's pretrained weights, converted to tensorflow format.

# Details

* All models require 224x224 image size. 
* The input images are preprocessed by substracting the RGB mean = [ 124, 117, 104 ], and then multiplying by 0.0167.
* Axis = 3 is considered the channel axis.

# How to load weights example

```
model_type = 'dpn92'
model = dpn_model(input_shape=(224, 224, 3), model_type=model_type)
model.load_weights('{}.h5'.format(model_type))
```

