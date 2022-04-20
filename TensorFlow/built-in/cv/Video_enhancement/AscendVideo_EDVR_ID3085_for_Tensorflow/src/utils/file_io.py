# Copyright 2022 Huawei Technologies Co., Ltd
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import multiprocessing as mp
import os
import queue
import random
import shutil
import threading
import time
from functools import partial
from subprocess import Popen, PIPE

import cv2
import imageio
import numpy as np
from PIL import Image
from src.utils.klass import Singleton
from src.utils.logger import logger
from src.utils.constant import VALID_COLORSPACE, IO_BACKEND
from src.utils.utils import convert_to_dict


def imread(x, target_color_space='rgb'):
    """Wrapped image read function.

    Support normal SDR png image, as well as HDR exr image.

    Args:
        x: str, image file name.
        target_color_space: str, what color space should the output image is in.
    
    Returns:
        ndarray, an image of the target_color_space.
    """
    target_color_space = target_color_space.lower()
    assert target_color_space in VALID_COLORSPACE

    if x.endswith('.exr'):
        # read hdr
        im = cv2.imread(x, cv2.IMREAD_UNCHANGED)
    else:
        im = cv2.imread(x)

    # convert to grayscale if required.
    if target_color_space == 'gray3d':
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    elif target_color_space == 'gray':
        im = im[:,:,0:1]

    # data_format convert
    if target_color_space in ['bgr', 'gray', 'gray3d']:
        out = im
    elif target_color_space == 'rgb':
        out = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    elif target_color_space == 'lab':
        out = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    elif target_color_space == 'ycrcb':
        out = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    elif target_color_space == 'yuv':
        out = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    elif target_color_space == 'y':
        out = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
        out = out[:,:,0:1]
    else:
        raise ValueError("Unknown data_format as {}, or maybe just mismatched!".format(target_color_space))

    return out


def imwrite(name, x, source_color_space='rgb', benormalized=True):
    """Wrapped image write function.

    Support normal SDR png image, as well as HDR exr image.

    Args:
        name: str, output image file name.
        x: ndarray, with shape [H, W, C].
        source_color_space: str, in what color space the source image is.
        benormalized: boolean, whether the image is normalized.
    """
    source_color_space = source_color_space.lower()
    assert source_color_space in VALID_COLORSPACE

    hdr = name.endswith('.exr')
    out = image_deprocess(x, source_color_space, benormalized, hdr)
    if hdr:
        hdr_image_write(name, out)
    else:
        sdr_image_write(name, out)


def image_deprocess(x, source_color_space='rgb', benormalized=True, hdr=False):
    """Image deprocess function.
    
    Converts the normalized ndarray to a writable image by opencv.
    """
    if hdr:
        return hdr_image_deprocess(x, source_color_space)
    else:
        return sdr_image_deprocess(x, source_color_space, benormalized)


def hdr_image_deprocess(x, source_color_space='rgb'):
    """Image deprocess function of HDR image.
    
    HDR image is always normalized. The only thing to do is to convert to another
    color space.
    """
    if source_color_space == 'rgb':
        x = x[..., ::-1]
    elif source_color_space == 'bgr':
        pass
    else:
        raise NotImplementedError(f'HDR output does not support color-spaces other than RGB and BGR.')
    return x


def sdr_image_deprocess(x, source_color_space='rgb', benormalized=True):
    """Image deprocess function of SDR image.
    
    Converts the color space to 'bgr' for opencv to write out; denormalizes the
    data to uint8.
    """
    source_color_space = source_color_space.lower()
    assert source_color_space in VALID_COLORSPACE

    if benormalized and source_color_space not in ['ycrcb', 'yuv', 'y']:
        x[...] = x[...] * 255
    x = np.clip(x, 0., 255.)

    if source_color_space in ['bgr', 'gray']:
        out = x
    elif source_color_space == 'rgb':
        out = cv2.cvtColor(x, cv2.COLOR_RGB2BGR, cv2.CV_32F)
    elif source_color_space in ['lab', 'gray3d']:
        x[:, :, 0:1] = x[:, :, 0:1] / 2.55
        x[:, :, 1:3] = x[:, :, 1:3] - 128.
        out = cv2.cvtColor(x, cv2.COLOR_LAB2BGR, cv2.CV_32F)
        out[...] = out[...] * 255.
    elif source_color_space == 'ycrcb':
        out = cv2.cvtColor(x, cv2.COLOR_YCrCb2BGR, cv2.CV_32F)
        if benormalized:
            out = np.clip(out * 255., 0., 255.)
    elif source_color_space == 'yuv':
        out = cv2.cvtColor(x, cv2.COLOR_YUV2BGR, cv2.CV_32F)
        out = np.clip(out * 255, 0., 255.)
    elif source_color_space == 'y':
        out = np.clip(x * 255, 0., 255.)
    else:
        raise ValueError

    out = out.astype(np.uint8)
    return out


def sdr_image_write(name, out):
    # Just a wrapper.
    cv2.imwrite(name, out)


def hdr_image_write(name, out):
    # Save as half precision to for smaller file.
    out = np.maximum(out, 0.)
    cv2.imwrite(name, out, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])

class HardDiskImageWriter:
    """An image writer which saves the image data in a file on the hard disk.

    We use a queue and multi-thread to write the images to hard disk in the 
    background.

    Args:
        max_num_threads: int, maximum number of the threads to use.
        max_queue_size: int, maximum size of the queue to save the data in memory.
    """
    def __init__(self, max_num_threads=1, max_queue_size=64):
        self.queue = queue.Queue(max_queue_size)
        self.threads_pool = []
        self.sentinel = (None, None)
        self.max_num_threads = max_num_threads
        self.notified = False

    def worker(self):
        # Thread work to write out the images.
        while True:
            try:
                elem = self.queue.get(True)
                if id(elem) == id(self.sentinel):
                    self.end()
                    break
                target_path, im_data = elem
                hdr = target_path.endswith('.exr')
                if hdr:
                    hdr_image_write(target_path, im_data)
                else:
                    sdr_image_write(target_path, im_data)
            except Exception as e:
                if not self.notified:
                    self.notified = True
                    logger.error(f'Error when writing out images, {e}.')
                pass

    def __del__(self):
        # Wait until all the threads to join.
        for t in self.threads_pool:
            try:
                t.join()
            except:
                pass
        logger.info('Processing remaining elements')

        # Post check whether there are un-written.
        while True:
            try:
                elem = self.queue.get(False)
                assert id(elem) == id(self.sentinel), '[Warning] Remain elements in writing queue'
            except queue.Empty:
                break

    def put_to_queue(self, target_path, im_data):
        # Put the target file name and the image data in the queue.
        self.queue.put((target_path, im_data))
        if len(self.threads_pool) <= self.max_num_threads:
            t = threading.Thread(target=self.worker, args=())
            t.start()
            self.threads_pool.append(t)

    def end(self):
        # Put sentinel in the queue to call exit.
        self.queue.put(self.sentinel)


# https://github.com/imageio/imageio-ffmpeg/blob/f27b6cb31d4ed3fd436f3a22871b2b63d2384c80/imageio_ffmpeg/_utils.py#L55
def _popen_kwargs(prevent_sigint=False):
    startupinfo = None
    preexec_fn = None
    creationflags = 0
    if prevent_sigint:
        # Prevent propagation of sigint (see #4)
        # https://stackoverflow.com/questions/5045771
        preexec_fn = os.setpgrp  # the _pre_exec does not seem to work

    falsy = ("", "0", "false", "no")
    if os.getenv("IMAGEIO_FFMPEG_NO_PREVENT_SIGINT", "").lower() not in falsy:
        # Unset preexec_fn to work around a strange hang on fork() (see #58)
        preexec_fn = None

    return {
        "startupinfo": startupinfo,
        "creationflags": creationflags,
        "preexec_fn": preexec_fn,
    }


class FFMPEGStreamWriter:
    """An image writer which saves the image data through ffmpeg stream to a video
    file.

    Args:
        video_filename: str, output target video file.
        fps: str, vidoe fps for encoding.
        output_param_file: str, codec config file for encoding.
        output_resolution: list[int], the resolution [H, W] of the output video.
        source_pix_fmt: str, the pixel format of the input to ffmpeg. When encoding
            SDR vidoe, it should be `bgr24`. Or it should be 'gbrpf32le' for HDR.
        ffmpeg_timeout: int, time limitation to prevent ffmpeg dies.
    """
    def __init__(self, video_filename, fps='25',
                 output_param_file='./configs/codecs/default_x264.json',
                 output_resolution=None,
                 source_pix_fmt='bgr24',
                 ffmpeg_timeout=60):

        if output_resolution is None:
            raise ValueError('Expect the output resolution, but got None.')
        else:
            assert len(output_resolution) == 2
        # W x H
        s = f"{output_resolution[1]}x{output_resolution[0]}"

        with open(output_param_file, 'r') as fid:
            output_params_dict = json.load(fid)
        
        # Input information.
        vinput_opts = [
                       '-r', str(fps),
                       '-f', 'rawvideo',
                       '-s', s,
                       '-pix_fmt', source_pix_fmt,
                       '-analyzeduration', str(2147483647),
                       '-probesize', str(2147483647),
        ]
        vinput_src = ['-i', '-']

        # Output encoding information.
        output_params = []
        vcodec = None
        bitrate = None
        pix_fmt = "yuv420p"
        for k, v in output_params_dict["codec"].items():
            if k in ["-c:v", "-vcodec"]:
                vcodec = v
            elif k == '-bitrate':
                bitrate = v
            elif k == '-pix_fmt':
                pix_fmt = v
            else:
                output_params += [k, v]
        ext = output_params_dict.get("format", 'mp4')

        if bitrate is not None:
            output_params += ['-bitrate', bitrate]
        if vcodec is not None:
            output_params += ['-c:v', vcodec]
        output_params += ['-pix_fmt', pix_fmt]

        if not video_filename.endswith(ext):
            video_filename = f'{video_filename}.{ext}'

        self.ffmpeg_timeout = ffmpeg_timeout

        self._basic_cmd = ['ffmpeg', '-y',
                           *vinput_opts,
                           *vinput_src,
                           *output_params,
                           video_filename,
                           ]

    def initialize(self):
        # Use a generator to accept the image data without blocking the main 
        # processing.
        self.write_gen = self._initialize_gen()
        assert self.write_gen is not None
        self.write_gen.send(None)
        logger.info("Codec command:")
        logger.info(self._basic_cmd)

    def _initialize_gen(self):
        # Borrowed from imageio-ffmpeg
        # https://github.com/imageio/imageio-ffmpeg/blob/master/imageio_ffmpeg/_io.py#L478
        stop_policy = 'timeout'
        p = Popen(
            self._basic_cmd,
            stdin=PIPE,
            stdout=None,
            stderr=None,
            **_popen_kwargs(prevent_sigint=True)
        )

        try:
            while True:
                frame = yield
                try:
                    p.stdin.write(frame)
                except Exception as err:
                    msg = (
                        "{0:}\n\nFFMPEG COMMAND:\n{1:}\n\nFFMPEG STDERR "
                        "OUTPUT:\n".format(err, self._basic_cmd)
                    )
                    stop_policy = "kill"
                    raise IOError(msg)
        except GeneratorExit:
            # Note that GeneratorExit does not inherit from Exception but BaseException
            # Detect premature closing
            raise
        except Exception:
            # Normal exceptions fall through
            raise
        except BaseException:
            # Detect KeyboardInterrupt / SystemExit: don't wait for ffmpeg to quit
            stop_policy = "kill"
            raise
        finally:
            if p.poll() is None:
                try:
                    p.stdin.close()
                except Exception as err:  # pragma: no cover
                    logger.warning("Error while attempting stop ffmpeg (w): " + str(err))

                if stop_policy == "timeout":
                    # Wait until timeout, produce a warning and kill if it still exists
                    try:
                        etime = time.time() + self.ffmpeg_timeout
                        while (time.time() < etime) and p.poll() is None:
                            time.sleep(0.01)
                    finally:
                        if p.poll() is None:  # pragma: no cover
                            logger.warn(
                                "We had to kill ffmpeg to stop it. "
                                + "Consider increasing ffmpeg_timeout, "
                                + "or setting it to zero (no timeout)."
                            )
                            p.kill()

                elif stop_policy == "wait":
                    # Wait forever, kill if it if we're interrupted
                    try:
                        while p.poll() is None:
                            time.sleep(0.01)
                    finally:  # the above can raise e.g. by ctrl-c or systemexit
                        if p.poll() is None:  # pragma: no cover
                            p.kill()

                else:  #  stop_policy == "kill":
                    # Just kill it
                    p.kill()
            # Just to be safe, wrap in try/except
            try:
                p.stdout.close()
            except Exception:
                pass

    def put_to_queue(self, target_path, im_data):
        # target_path won't matter here
        if im_data.dtype == np.float32:
            # Notice that after the deprocess, everything is in bgr color space.
            # HDR data, will use gbrpf32le pix_fmt.
            # Make it [C, H, W] data format, and shift channels to [g, b, r].
            im_data = np.transpose(im_data[..., [1,0,2]], [2,0,1])
        # else is normal uint8 data. Use bgr24le pix_fmt and don't do anything
        img_str = im_data.tobytes()
        self.write_gen.send(img_str)

    def end(self):
        self.write_gen.close()


class ImageWriter:
    """A top class to handle the image writing.

    Multi-imagewriter is supported when writing out. This class contains all the
    concrete writing instances, and deprocess the results passed by the engine and
    feed to the writers. Multi-imagewriter can be configured using the 
    cfg.inference.io_backend where the backends are concatenated with ':'. For 
    example, setting:
    
    cfg.inference.io_backend = 'disk:ffmpeg'

    will use two image writer instance, one HardDiskImageWriter and the other
    FFMPEGStreamWriter.    

    Args:
        output_dir: str, output top folder.
        cfg: yacs node, global configuration.
        source_color_space: str, in what color space the source image is.
        benormalized: boolean, whether the image is normalized.
        output_resolution: list[int], the resolution [H, W] of the output video.
        pix_fmt: str, the pixel format of the input to ffmpeg. When encoding
            SDR vidoe, it should be `bgr24`. Or it should be 'gbrpf32le' for HDR.
    """
    def __init__(self, output_dir, cfg, benormalized=True, 
                 source_color_space='bgr', output_resolution=None,
                 pix_fmt='bgr24'
                 ):
        io_backends = cfg.inference.io_backend.split(':')
        for ib in io_backends:
            IO_BACKEND.CHECK_VALID(ib)

        self.io_backend = io_backends
        self.cfg = cfg

        self.image_deprocess = partial(image_deprocess,
                                       source_color_space=source_color_space,
                                       benormalized=benormalized)
        self.pix_fmt = pix_fmt
        self.output_resolution = output_resolution
        self.root_output_dir = output_dir
        self.writers = []

        for ib in io_backends:
            self.add_writers(ib, self.root_output_dir)

    def add_writers(self, io_backend, root_dir=None):
        # Add specific writers.
        if root_dir is None:
            root_dir = self.root_output_dir

        if io_backend == IO_BACKEND.DISK:
            writer = HardDiskImageWriter(max_num_threads=self.cfg.inference.writer_num_threads,
                                         max_queue_size=self.cfg.inference.writer_queue_size)
            output_folder = root_dir
        elif io_backend == IO_BACKEND.FFMPEG:
            video_filename = os.path.join(f'{root_dir}_videos', self.cfg.inference.ffmpeg.video_filename)
            writer = FFMPEGStreamWriter(video_filename=video_filename,
                                        fps=self.cfg.inference.ffmpeg.fps,
                                        output_param_file=self.cfg.inference.ffmpeg.codec_file,
                                        source_pix_fmt=self.pix_fmt,
                                        output_resolution=self.output_resolution,
                                        )
            output_folder = f'{root_dir}_videos'
        else:
            raise NotImplementedError(f'{io_backend}')

        # Record both the writer instance and the output folder.
        self.writers.append([writer, output_folder])

    def initialize(self):
        # Initialization and create folders if necessary.
        logger.info(f'Using {self.io_backend} as the io backend.')
        for writer_id, ib in enumerate(self.io_backend):
            if ib in [IO_BACKEND.DISK, IO_BACKEND.FFMPEG]:
                output_folder = self.writers[writer_id][1]
                logger.info(f'For {ib} backend, the results will be written to {output_folder}')
                os.makedirs(output_folder, exist_ok=True)

            if ib == IO_BACKEND.FFMPEG:
                self.writers[writer_id][0].initialize()

    def finalize(self):
        # Close the writers.
        for writer, output_folder in self.writers:
            writer.end()

    def write_out(self, output_data_dict):
        output_data_dict = dict(sorted(output_data_dict.items(), key=lambda item: item[0]))
        # Append image date to the writers after inference.
        for target_file, data in output_data_dict.items():
            # without file copy: {output_file_name: ndarray}
            # with file copy: {output_file_name: [source_file_name, ndarray]}
            for backend_id, ib in enumerate(self.io_backend):
                writer, output_folder = self.writers[backend_id]
                target_file = os.path.join(output_folder, target_file)
                if isinstance(data, np.ndarray):
                    # This scenario is only for multi-in single-out model, not include vfi
                    writer.put_to_queue(target_file, data)
                elif isinstance(data, (list, tuple)):
                    # Mainly used in vfi scenario, or pipeline scenario.
                    if ib == IO_BACKEND.DISK:
                        # In the single vfi processing, we use shutil to copy the
                        # the original data instead of writing out from memory to disk.
                        # Yet we have not tested the performance of `writing out` strategy.
                        assert isinstance(data[0], str)
                        shutil.copy(data[0], target_file)
                    elif ib == IO_BACKEND.FFMPEG:
                        # This is used in ffmpeg stream, the first the target file to output
                        # while the second the output data.
                        assert isinstance(data[1], np.ndarray)
                        writer.put_to_queue(target_file, data[1])
                    else:
                        raise NotImplementedError
                else:
                    raise TypeError(f'Expect value `data` to be np.ndarray, or a list of [str, np.ndarray].'
                                    f'But given {type(data)}')
