import os.path as osp
import sys
import warnings

import tensorflow as tf
from rich.console import Console

CONSOLE = Console()


sys.path.append("../")
warnings.filterwarnings("ignore")


def test_model(dataset: str, model_config, test_x):
    if osp.exists(osp.join(model_config['dirs']['saved_models'], dataset)):
        model = tf.keras.models.load_model(
            osp.join(model_config['dirs']['saved_models'], dataset))
    else:
        CONSOLE.print('Model could not be found @"saved_model" directory',
                      style='red')
        return

    # CONSOLE.log(model_config[dataset]['batch_size'])
    pred = model.predict(test_x, batch_size=model_config[dataset]['batch_size'], verbose=1)

    return pred
