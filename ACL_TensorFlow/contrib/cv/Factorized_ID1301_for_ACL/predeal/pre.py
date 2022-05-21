from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
__all__ = [tf]
import glob, os, random, collections, argparse
from utils.warp import image_warping2
import numpy as np
MODE = "train"
INPUT_DIR = 'C:/Users/13329/PycharmProjects/pythonProject2'  #读取数据集路径
OUTPUT_DIR = './output' #输出路径
LANDMARK_N = 8
CHECKPOINT = None

SAVE_FREQ = 500
SUMMARY_FREQ = 20
BATCH_SIZE = 1
DOWNSAMPLE_M = 4
DIVERSITY = 500.
ALIGN = 1.
LEARNING_RATE = 1.e-4
MOMENTUM = 0.5
RANDOM_SEED = 1234
WEIGHT_DECAY = 0.0005
SCALE_SIZE = 146
CROP_SIZE = 146
MAX_EPOCH = 200
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Factorized Spatial Embeddings")
    #parser.add_argument("--mode", default=MODE, choices=["train", "test"])
    parser.add_argument("--mode", default=MODE)
    parser.add_argument("--input_dir", default=INPUT_DIR,help="Path to the directory containing the training or testing images.")
    parser.add_argument("--K", type=int, default=LANDMARK_N,help="Number of landmarks.")
    parser.add_argument("--output_dir", default=OUTPUT_DIR,help="Where to put output files")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,help="Number of images sent to the network in one step.")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,help="Learning rate for adam.")
    parser.add_argument("--beta1", type=float, default=MOMENTUM, help="Momentum component of the optimiser.")
    parser.add_argument("--M", type=int, default=DOWNSAMPLE_M,help="Downsampling value of the diversity loss.")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY,help="Regularisation parameter for L2-loss.")
    parser.add_argument("--random_seed", type=int, default=RANDOM_SEED,help="Random seed to have reproducible results.")
    parser.add_argument("--diversity_weight", type=float, default=DIVERSITY,help="Weight on diversity loss.")
    parser.add_argument("--align_weight", type=float, default=ALIGN,help="Weight on align loss.")
    parser.add_argument("--scale_size", type=int, default=SCALE_SIZE,help="Scale images to this size before cropping to CROP_SIZE")
    parser.add_argument("--crop_size", type=int, default=CROP_SIZE,help="CROP images to this size")
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCH,help="Number of training epochs")
    parser.add_argument("--checkpoint", default=CHECKPOINT,help="Directory with checkpoint to resume training from or use for testing")
    parser.add_argument("--summary_freq", type=int, default=SUMMARY_FREQ,help="Update summaries every summary_freq steps")
    parser.add_argument("--save_freq", type=int, default=SAVE_FREQ, help="Save model every save_freq steps")
    parser.add_argument("--num_gpus", default=1)
    return parser.parse_args()
# Collections definition
Examples = collections.namedtuple("Examples","paths, images, images_deformed, deformation, count, steps_per_epoch, shape")
Model = collections.namedtuple("Model", "pos_loss, neg_loss, distance")
def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1
def load_examples(args):
    """Load all images in the input_dir.

    Returns:
      Examples.paths : batch of path of images,
      Examples.images : batch of images,
      Examples.images_deformed : batch of deformed images,
      Examples.deformation : batch of deformation parameters,
    """
    #if args.input_dir is None or not os.path.exists(args.input_dir):
    if args.input_dir is None :
        raise Exception("input_dir does not exist")
    # load distorted pairs address
    input_paths = glob.glob(os.path.join(args.input_dir, "*.jpg"))
    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    with tf.name_scope("load_images"):
        contents = tf.read_file(input_paths[0])
        input = tf.image.decode_jpeg(contents)
        input = tf.image.convert_image_dtype(input, dtype=tf.float32)
        assertion = tf.assert_equal(tf.shape(input)[2], 3, message="image does not have required channels")
        with tf.control_dependencies([assertion]):
            input = tf.identity(input)
        input.set_shape([None, None, 3])
        images = preprocess(input)
    seed = random.randint(0, 2 ** 31 - 1)
    # scale and crop input image to match 256x256 size
    def transform(image):
        r = image
        r = tf.image.resize_images(r, [args.scale_size, args.scale_size], method=tf.image.ResizeMethod.AREA)
        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, args.scale_size - args.crop_size + 1, seed=seed)), dtype=tf.int32)
        if args.scale_size > args.crop_size:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], args.crop_size, args.crop_size)
        elif args.scale_size < args.crop_size:
            raise Exception("scale size cannot be less than crop size")
        return r
    with tf.name_scope("images"):
        input_images = transform(images)
        if args.mode=="train":
            input_images, _ = image_warping2(input_images, w=0.0)
        deformed_images, deformation = image_warping2(input_images, w=0.1)
        deformation = tf.squeeze(deformation)
        # crop after warping
        input_images = tf.image.crop_to_bounding_box(input_images, 5, 5, 128, 128)
        deformed_images = tf.image.crop_to_bounding_box(deformed_images, 5, 5, 128, 128)
        # clip image values
        input_images = tf.clip_by_value(input_images, clip_value_min=-1., clip_value_max=1.)
        deformed_images = tf.clip_by_value(deformed_images, clip_value_min=-1., clip_value_max=1.)
        deformation = tf.expand_dims(deformation, 0)
        deformation.set_shape(args.batch_size+ deformation.shape[1:])
        np1 = tf.Session().run(input_images)
        np2 = deformed_images.eval(session=tf.Session())
        deformationlog = tf.Session().run(deformation)

        np1.tofile("./1.bin")
        np2.tofile("./2.bin")
        np.save("deformationlog.npy",deformationlog)


def main():
    """Create the model and start the training."""
    args = get_arguments()
    tf.set_random_seed(args.random_seed)
    load_examples(args)
if __name__ == '__main__':
    main()
