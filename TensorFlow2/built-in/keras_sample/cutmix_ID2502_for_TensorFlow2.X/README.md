# CutMix data augmentation for image classification

**Author:** [Sayan Nath](https://twitter.com/sayannath2350)<br>
**Date created:** 2021/06/08<br>
**Last modified:** 2021/06/08<br>
**Description:** Data augmentation with CutMix for image classification on CIFAR-10.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team\keras-io\blob\master\examples\vision/ipynb/cutmix.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team\keras-io\blob\master\examples\vision/cutmix.py)



---
## Introduction

_CutMix_ is a data augmentation technique that addresses the issue of information loss
and inefficiency present in regional dropout strategies.
Instead of removing pixels and filling them with black or grey pixels or Gaussian noise,
you replace the removed regions with a patch from another image,
while the ground truth labels are mixed proportionally to the number of pixels of combined images.
CutMix was proposed in
[CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/pdf/1905.04899.pdf)
(Yun et al., 2019)

It's implemented via the following formulas:

<img src="https://i.imgur.com/cGvd13V.png" width="200"/>

where `M` is the binary mask which indicates the cutout and the fill-in
regions from the two randomly drawn images and `λ` (in `[0, 1]`) is drawn from a
[`Beta(α, α)` distribution](https://en.wikipedia.org/wiki/Beta_distribution)

The coordinates of bounding boxes are:

<img src="https://i.imgur.com/eNisep4.png" width="150"/>

which indicates the cutout and fill-in regions in case of the images.
The bounding box sampling is represented by:

<img src="https://i.imgur.com/Snph9aj.png" width="200"/>

where `rx, ry` are randomly drawn from a uniform distribution with upper bound.

---
## Setup


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

np.random.seed(42)
tf.random.set_seed(42)
```

---
## Load the CIFAR-10 dataset

In this example, we will use the
[CIFAR-10 image classification dataset](https://www.cs.toronto.edu/~kriz/cifar.html).


```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]
```

<div class="k-default-codeblock">
```
(50000, 32, 32, 3)
(50000, 10)
(10000, 32, 32, 3)
(10000, 10)
```
</div>
    

---
## Define hyperparameters


```python
AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 32
IMG_SIZE = 32
```

---
## Define the image preprocessing function


```python

def preprocess_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.image.convert_image_dtype(image, tf.float32) / 255.0
    return image, label

```

---
## Convert the data into TensorFlow `Dataset` objects


```python
train_ds_one = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(1024)
    .map(preprocess_image, num_parallel_calls=AUTO)
)
train_ds_two = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(1024)
    .map(preprocess_image, num_parallel_calls=AUTO)
)

train_ds_simple = tf.data.Dataset.from_tensor_slices((x_train, y_train))

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_ds_simple = (
    train_ds_simple.map(preprocess_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

# Combine two shuffled datasets from the same training data.
train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

test_ds = (
    test_ds.map(preprocess_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)
```

---
## Define the CutMix data augmentation function

The CutMix function takes two `image` and `label` pairs to perform the augmentation. It samples `λ(l)` from the [Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution) and returns a bounding box from `get_box` function. We then crop the second image (`image2`) and pad this image in the final padded image at the same location.


```python

def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


@tf.function
def get_box(lambda_value):
    cut_rat = tf.math.sqrt(1.0 - lambda_value)

    cut_w = IMG_SIZE * cut_rat  # rw
    cut_w = tf.cast(cut_w, tf.int32)

    cut_h = IMG_SIZE * cut_rat  # rh
    cut_h = tf.cast(cut_h, tf.int32)

    cut_x = tf.random.uniform((1,), minval=0, maxval=IMG_SIZE, dtype=tf.int32)  # rx
    cut_y = tf.random.uniform((1,), minval=0, maxval=IMG_SIZE, dtype=tf.int32)  # ry

    boundaryx1 = tf.clip_by_value(cut_x[0] - cut_w // 2, 0, IMG_SIZE)
    boundaryy1 = tf.clip_by_value(cut_y[0] - cut_h // 2, 0, IMG_SIZE)
    bbx2 = tf.clip_by_value(cut_x[0] + cut_w // 2, 0, IMG_SIZE)
    bby2 = tf.clip_by_value(cut_y[0] + cut_h // 2, 0, IMG_SIZE)

    target_h = bby2 - boundaryy1
    if target_h == 0:
        target_h += 1

    target_w = bbx2 - boundaryx1
    if target_w == 0:
        target_w += 1

    return boundaryx1, boundaryy1, target_h, target_w


@tf.function
def cutmix(train_ds_one, train_ds_two):
    (image1, label1), (image2, label2) = train_ds_one, train_ds_two

    alpha = [0.25]
    beta = [0.25]

    # Get a sample from the Beta distribution
    lambda_value = sample_beta_distribution(1, alpha, beta)

    # Define Lambda
    lambda_value = lambda_value[0][0]

    # Get the bounding box offsets, heights and widths
    boundaryx1, boundaryy1, target_h, target_w = get_box(lambda_value)

    # Get a patch from the second image (`image2`)
    crop2 = tf.image.crop_to_bounding_box(
        image2, boundaryy1, boundaryx1, target_h, target_w
    )
    # Pad the `image2` patch (`crop2`) with the same offset
    image2 = tf.image.pad_to_bounding_box(
        crop2, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE
    )
    # Get a patch from the first image (`image1`)
    crop1 = tf.image.crop_to_bounding_box(
        image1, boundaryy1, boundaryx1, target_h, target_w
    )
    # Pad the `image1` patch (`crop1`) with the same offset
    img1 = tf.image.pad_to_bounding_box(
        crop1, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE
    )

    # Modify the first image by subtracting the patch from `image1`
    # (before applying the `image2` patch)
    image1 = image1 - img1
    # Add the modified `image1` and `image2`  together to get the CutMix image
    image = image1 + image2

    # Adjust Lambda in accordance to the pixel ration
    lambda_value = 1 - (target_w * target_h) / (IMG_SIZE * IMG_SIZE)
    lambda_value = tf.cast(lambda_value, tf.float32)

    # Combine the labels of both images
    label = lambda_value * label1 + (1 - lambda_value) * label2
    return image, label

```

**Note**: we are combining two images to create a single one.

---
## Visualize the new dataset after applying the CutMix augmentation


```python
# Create the new dataset using our `cutmix` utility
train_ds_cmu = (
    train_ds.shuffle(1024)
    .map(cutmix, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

# Let's preview 9 samples from the dataset
image_batch, label_batch = next(iter(train_ds_cmu))
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.title(class_names[np.argmax(label_batch[i])])
    plt.imshow(image_batch[i])
    plt.axis("off")
```


    
![png](/img/examples/vision/cutmix/cutmix_16_0.png)
    


---
## Define a ResNet-20 model


```python

def resnet_layer(
    inputs,
    num_filters=16,
    kernel_size=3,
    strides=1,
    activation="relu",
    batch_normalization=True,
    conv_first=True,
):
    conv = keras.layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = keras.layers.BatchNormalization()(x)
        if activation is not None:
            x = keras.layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = keras.layers.BatchNormalization()(x)
        if activation is not None:
            x = keras.layers.Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v20(input_shape, depth, num_classes=10):
    if (depth - 2) % 6 != 0:
        raise ValueError("depth should be 6n+2 (eg 20, 32, 44 in [a])")
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = keras.layers.Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(
                    inputs=x,
                    num_filters=num_filters,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False,
                )
            x = keras.layers.add([x, y])
            x = keras.layers.Activation("relu")(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = keras.layers.AveragePooling2D(pool_size=8)(x)
    y = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(
        num_classes, activation="softmax", kernel_initializer="he_normal"
    )(y)

    # Instantiate model.
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    return model


def training_model():
    return resnet_v20((32, 32, 3), 20)


initial_model = training_model()
initial_model.save_weights("initial_weights.h5")
```

---
## Train the model with the dataset augmented by CutMix


```python
model = training_model()
model.load_weights("initial_weights.h5")

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_ds_cmu, validation_data=test_ds, epochs=15)

test_loss, test_accuracy = model.evaluate(test_ds)
print("Test accuracy: {:.2f}%".format(test_accuracy * 100))
```

<div class="k-default-codeblock">
```
Epoch 1/15
1563/1563 [==============================] - 62s 24ms/step - loss: 1.9216 - accuracy: 0.4090 - val_loss: 1.9737 - val_accuracy: 0.4061
Epoch 2/15
1563/1563 [==============================] - 37s 24ms/step - loss: 1.6549 - accuracy: 0.5325 - val_loss: 1.5033 - val_accuracy: 0.5061
Epoch 3/15
1563/1563 [==============================] - 38s 24ms/step - loss: 1.5536 - accuracy: 0.5840 - val_loss: 1.2913 - val_accuracy: 0.6112
Epoch 4/15
1563/1563 [==============================] - 38s 24ms/step - loss: 1.4988 - accuracy: 0.6097 - val_loss: 1.0587 - val_accuracy: 0.7033
Epoch 5/15
1563/1563 [==============================] - 38s 24ms/step - loss: 1.4531 - accuracy: 0.6291 - val_loss: 1.0681 - val_accuracy: 0.6841
Epoch 6/15
1563/1563 [==============================] - 37s 24ms/step - loss: 1.4173 - accuracy: 0.6464 - val_loss: 1.0265 - val_accuracy: 0.7085
Epoch 7/15
1563/1563 [==============================] - 37s 24ms/step - loss: 1.3932 - accuracy: 0.6572 - val_loss: 0.9540 - val_accuracy: 0.7331
Epoch 8/15
1563/1563 [==============================] - 37s 24ms/step - loss: 1.3736 - accuracy: 0.6680 - val_loss: 0.9877 - val_accuracy: 0.7240
Epoch 9/15
1563/1563 [==============================] - 38s 24ms/step - loss: 1.3575 - accuracy: 0.6782 - val_loss: 0.8944 - val_accuracy: 0.7570
Epoch 10/15
1563/1563 [==============================] - 38s 24ms/step - loss: 1.3398 - accuracy: 0.6886 - val_loss: 0.8598 - val_accuracy: 0.7649
Epoch 11/15
1563/1563 [==============================] - 38s 24ms/step - loss: 1.3277 - accuracy: 0.6939 - val_loss: 0.9032 - val_accuracy: 0.7603
Epoch 12/15
1563/1563 [==============================] - 38s 24ms/step - loss: 1.3131 - accuracy: 0.6964 - val_loss: 0.7934 - val_accuracy: 0.7926
Epoch 13/15
1563/1563 [==============================] - 37s 24ms/step - loss: 1.3050 - accuracy: 0.7029 - val_loss: 0.8737 - val_accuracy: 0.7552
Epoch 14/15
1563/1563 [==============================] - 37s 24ms/step - loss: 1.2987 - accuracy: 0.7099 - val_loss: 0.8409 - val_accuracy: 0.7766
Epoch 15/15
1563/1563 [==============================] - 37s 24ms/step - loss: 1.2953 - accuracy: 0.7099 - val_loss: 0.7850 - val_accuracy: 0.8014
313/313 [==============================] - 3s 9ms/step - loss: 0.7850 - accuracy: 0.8014
Test accuracy: 80.14%
```
</div>
    

---
## Train the model using the original non-augmented dataset


```python
model = training_model()
model.load_weights("initial_weights.h5")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_ds_simple, validation_data=test_ds, epochs=15)

test_loss, test_accuracy = model.evaluate(test_ds)
print("Test accuracy: {:.2f}%".format(test_accuracy * 100))
```

<div class="k-default-codeblock">
```
Epoch 1/15
1563/1563 [==============================] - 38s 23ms/step - loss: 1.4864 - accuracy: 0.5173 - val_loss: 1.3694 - val_accuracy: 0.5708
Epoch 2/15
1563/1563 [==============================] - 36s 23ms/step - loss: 1.0682 - accuracy: 0.6779 - val_loss: 1.1424 - val_accuracy: 0.6686
Epoch 3/15
1563/1563 [==============================] - 36s 23ms/step - loss: 0.8955 - accuracy: 0.7449 - val_loss: 1.0555 - val_accuracy: 0.7007
Epoch 4/15
1563/1563 [==============================] - 36s 23ms/step - loss: 0.7890 - accuracy: 0.7878 - val_loss: 1.0575 - val_accuracy: 0.7079
Epoch 5/15
1563/1563 [==============================] - 36s 23ms/step - loss: 0.7107 - accuracy: 0.8175 - val_loss: 1.1395 - val_accuracy: 0.7062
Epoch 6/15
1563/1563 [==============================] - 36s 23ms/step - loss: 0.6524 - accuracy: 0.8397 - val_loss: 1.1716 - val_accuracy: 0.7042
Epoch 7/15
1563/1563 [==============================] - 36s 23ms/step - loss: 0.6098 - accuracy: 0.8594 - val_loss: 1.4120 - val_accuracy: 0.6786
Epoch 8/15
1563/1563 [==============================] - 36s 23ms/step - loss: 0.5715 - accuracy: 0.8765 - val_loss: 1.3159 - val_accuracy: 0.7011
Epoch 9/15
1563/1563 [==============================] - 36s 23ms/step - loss: 0.5477 - accuracy: 0.8872 - val_loss: 1.2873 - val_accuracy: 0.7182
Epoch 10/15
1563/1563 [==============================] - 36s 23ms/step - loss: 0.5233 - accuracy: 0.8988 - val_loss: 1.4118 - val_accuracy: 0.6964
Epoch 11/15
1563/1563 [==============================] - 36s 23ms/step - loss: 0.5165 - accuracy: 0.9045 - val_loss: 1.3741 - val_accuracy: 0.7230
Epoch 12/15
1563/1563 [==============================] - 36s 23ms/step - loss: 0.5008 - accuracy: 0.9124 - val_loss: 1.3984 - val_accuracy: 0.7181
Epoch 13/15
1563/1563 [==============================] - 36s 23ms/step - loss: 0.4896 - accuracy: 0.9190 - val_loss: 1.3642 - val_accuracy: 0.7209
Epoch 14/15
1563/1563 [==============================] - 36s 23ms/step - loss: 0.4845 - accuracy: 0.9231 - val_loss: 1.5469 - val_accuracy: 0.6992
Epoch 15/15
1563/1563 [==============================] - 36s 23ms/step - loss: 0.4749 - accuracy: 0.9294 - val_loss: 1.4034 - val_accuracy: 0.7362
313/313 [==============================] - 3s 9ms/step - loss: 1.4034 - accuracy: 0.7362
Test accuracy: 73.62%
```
</div>
    

---
## Notes

In this example, we trained our model for 15 epochs.
In our experiment, the model with CutMix achieves a better accuracy on the CIFAR-10 dataset
(80.36% in our experiment) compared to the model that doesn't use the augmentation (72.70%).
You may notice it takes less time to train the model with the CutMix augmentation.

You can experiment further with the CutMix technique by following the
[original paper](https://arxiv.org/abs/1905.04899).