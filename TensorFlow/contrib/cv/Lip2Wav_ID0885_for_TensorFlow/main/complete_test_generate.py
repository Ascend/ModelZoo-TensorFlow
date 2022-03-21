# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================
# Copyright 2021 Huawei Technologies Co., Ltd
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


#from main.synthesizer.rahul_train import time_string
from npu_bridge.npu_init import *
import synthesizer
from synthesizer import inference as sif
import numpy as np
import sys, cv2, os, pickle, argparse, subprocess
from tqdm import tqdm
from shutil import copy
from glob import glob

class Generator(object):
	def __init__(self):
		super(Generator, self).__init__()

		self.synthesizer = sif.Synthesizer(verbose=False)

	def read_window(self, window_fnames):
		window = []
		for fname in window_fnames:
			img = cv2.imread(fname)
			if img is None:
				raise FileNotFoundError('Frames maybe missing in {}.' 
						' Delete the video to stop this exception!'.format(sample['folder']))

			img = cv2.resize(img, (sif.hparams.img_size, sif.hparams.img_size))
			window.append(img)

		images = np.asarray(window) / 255. # T x H x W x 3
		return images

	def vc(self, sample, outfile):
		hp = sif.hparams
		images = sample['images']
		all_windows = []
		i = 0
		while i + hp.T <= len(images):
			all_windows.append(images[i : i + hp.T])
			i += hp.T - hp.overlap

		for window_idx, window_fnames in enumerate(all_windows):
			images = self.read_window(window_fnames)

			s = self.synthesizer.synthesize_spectrograms(images)[0]
			print ("returned s ", s.shape, window_idx)
			if window_idx == 0:
				mel = np.squeeze(s, axis = 2)
				#mel = s
			else:
				new_s = np.squeeze(s, axis = 2) #s[:, hp.mel_overlap:, :]
				print ('new_s shape : ', new_s.shape)
				mel = np.concatenate((mel, new_s[:, hp.mel_overlap:]), axis=1)

				#mel = np.concatenate((mel, s[:, hp.mel_overlap:]), axis=1)

		print ("outisde loop mel shape : ", mel.shape)			
		wav = self.synthesizer.griffin_lim(mel)
		sif.audio.save_wav(wav, outfile, sr=hp.sample_rate)


def get_image_list(split, data_root):
	filelist = []
	with open(os.path.join(data_root, '{}.txt'.format(split))) as vidlist:
		for vid_id in vidlist:
			vid_id = vid_id.strip()
			filelist.extend(list(glob(os.path.join(data_root, 'preprocessed', vid_id, '*/*.jpg'))))
	return filelist


def get_testlist(data_root):
	print (data_root)
	test_images = get_image_list('npu_test', data_root)
	print ("length : ", len(test_images))
	print('{} hours is available for testing'.format(len(test_images) / (sif.hparams.fps * 3600.)))
	test_vids = {}
	for x in test_images:
		x = x[:x.rfind('/')]
		test_vids[x] = True
	return list(test_vids.keys())

def to_sec(idx):
	frame_id = idx + 1
	sec = frame_id / float(sif.hparams.fps)
	return sec

def contiguous_window_generator(vidpath):
	frames = glob(os.path.join(vidpath, '*.jpg'))
	if len(frames) < sif.hparams.T: return

	ids = [int(os.path.splitext(os.path.basename(f))[0]) for f in frames]
	sortedids = sorted(ids)
	end_idx = 0
	start = sortedids[end_idx]

	while end_idx < len(sortedids):
		while end_idx < len(sortedids):
			if end_idx == len(sortedids) - 1:
				if sortedids[end_idx] + 1 - start >= sif.hparams.T: 
					yield ((to_sec(start), to_sec(sortedids[end_idx])), 
					[os.path.join(vidpath, '{}.jpg'.format(x)) for x in range(start, sortedids[end_idx] + 1)])
				return
			else:
				if sortedids[end_idx] + 1 == sortedids[end_idx + 1]:
					end_idx += 1
				else:
					if sortedids[end_idx] + 1 - start >= sif.hparams.T: 
						yield ((to_sec(start), to_sec(sortedids[end_idx])), 
						[os.path.join(vidpath, '{}.jpg'.format(x)) for x in range(start, sortedids[end_idx] + 1)])
					break
		
		end_idx += 1
		start = sortedids[end_idx]

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', "--data_root", help="Speaker folder path", required=True)
	parser.add_argument('-r', "--results_root", help="Speaker folder path", required=True)
	parser.add_argument('--checkpoint', help="Path to trained checkpoint", required=True)
	parser.add_argument("--preset", help="Speaker-specific hyper-params", type=str, required=True)
	args = parser.parse_args()


	## add speaker-specific parameters
	with open(args.preset) as f:
		sif.hparams.parse_json(f.read())

	sif.hparams.set_hparam('eval_ckpt', args.checkpoint)
	
	videos = get_testlist(args.data_root)

	if not os.path.isdir(args.results_root):
		os.mkdir(args.results_root)

	GTS_ROOT = os.path.join(args.results_root, 'gts/')
	WAVS_ROOT = os.path.join(args.results_root, 'wavs/')
	files_to_delete = []
	if not os.path.isdir(GTS_ROOT):
		os.mkdir(GTS_ROOT)
	else:
		files_to_delete = list(glob(GTS_ROOT + '*'))
	if not os.path.isdir(WAVS_ROOT):
		os.mkdir(WAVS_ROOT)
	else:
		files_to_delete.extend(list(glob(WAVS_ROOT + '*')))
	for f in files_to_delete: os.remove(f)

	g = Generator()
	template = 'ffmpeg -y -loglevel panic -ss {} -i {} -to {} -strict -2 {}'
	for vid in tqdm(videos):
		vidpath = vid + '/'
		for (ss, es), images in tqdm(contiguous_window_generator(vidpath)):
			sample = {}
			sample['images'] = images

			vidname = vid.split('/')[-2] + '_' + vid.split('/')[-1]
			outfile = '{}{}_{}:{}.wav'.format(WAVS_ROOT, vidname, ss, es)
			try:
				g.vc(sample, outfile)
			except KeyboardInterrupt:
				exit(0)
			except Exception as e:
				print(e)
				continue

			command = template.format(ss, vidpath + 'audio.wav', es, 
									'{}{}_{}:{}.wav'.format(GTS_ROOT, vidname, ss, es))

			subprocess.call(command, shell=True)

