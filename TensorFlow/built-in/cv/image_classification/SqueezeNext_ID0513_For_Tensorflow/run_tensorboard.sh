#!/usr/bin/env bash

MODELS="/usr/local/share/models/"

docker run -it -p 6006:6006  \
-v $(pwd):/usr/local/src/ \
-v $MODELS:$MODELS \
-v $DATA_DIR:$DATA_DIR \
tensorflow/tensorflow python -m tensorboard.main --logdir=$MODELS
