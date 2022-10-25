import cv2
import numpy as np
import os
import sys
import time

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path, "."))
sys.path.append(os.path.join(path, "./acllite/"))
from constants import IMG_EXT
from acllite_model import AclLiteModel
from acllite_image import AclLiteImage
from acllite_resource import AclLiteResource
from fast_rcnn.config import cfg
from fast_rcnn.nms_wrapper import nms
from rpn_msr.proposal_layer_tf import proposal_layer as proposal_layer_py
from text_proposal_connector import TextProposalConnector
from utils.blob import im_list_to_blob
MODEL_WIDTH = 900
MODEL_HEIGHT = 900
INPUT_DIR = '../data/'
OUTPUT_DIR = '../out/'
model_path = '../model/VGGnet_fast_rcnn.om'


def _get_blobs(im, rois):
    blobs = {'data': None, 'rois': None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    return blobs, im_scale_factors

def _get_image_blob(im):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(
            im_orig,
            None,
            None,
            fx=im_scale,
            fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def connect_proposal(text_proposals, scores, im_size):
    cp = TextProposalConnector()
    line = cp.get_text_lines(text_proposals, scores, im_size)
    return line

def save_results(image_name, im, line, width_rate, height_rate, thresh):
    inds = np.where(line[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = line[i, :4]
        score = line[i, -1]
        cv2.rectangle(
            im, (int(bbox[0]*width_rate), int(bbox[1]*height_rate)), (int(bbox[2]*width_rate), int(bbox[3]*height_rate)),
            color=(0, 0, 255),
            thickness=1)
    image_name = image_name.split('/')[-1]
    cv2.imwrite(os.path.join("../out/", image_name), im)
    
def test_ctpn(pic, im, boxes=None):
    anchor_scales = cfg.ANCHOR_SCALES
    _feat_stride = [16, ]
    blobs, im_scales = _get_blobs(im, boxes)
    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            dtype=np.float32)

    model = AclLiteModel(model_path)
    result_list = model.execute(blobs['data'])
    rpn_bbox_pred = result_list[0]
    rpn_cls_prob_reshape = result_list[1]
    blob, bbox_deltas = \
        proposal_layer_py(rpn_cls_prob_reshape, rpn_bbox_pred, blobs['im_info'],\
                          'TEST', _feat_stride, anchor_scales)
    rois = blob
    scores = rois[:, 0]
    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        boxes = rois[:, 1:5] / im_scales[0]
    return scores, boxes


def inference(image_name):#del net sess
    # preprocess
    img = cv2.imread(image_name)
    im_size = img.shape
    width_rate = im_size[1] / MODEL_WIDTH
    height_rate = im_size[0] / MODEL_HEIGHT
    img = cv2.resize(img, (MODEL_WIDTH, MODEL_HEIGHT)).astype(np.float32)
    im = img
    cfg.TEST.HAS_RPN = True
    # inference
    scores, boxes = test_ctpn(image_name,im)
    # postprocess
    CONF_THRESH = 0.9
    NMS_THRESH = 0.3
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]

    keep = np.where(dets[:, 4] >= 0.7)[0]
    dets = dets[keep, :]

    line = connect_proposal(dets[:, 0:4], dets[:, 4], im.shape)
    im = cv2.resize(img, (int(MODEL_WIDTH*width_rate), int(MODEL_HEIGHT*height_rate))).astype(np.float32)
    save_results(image_name, im, line, width_rate, height_rate, thresh=0.9)


def main():
    """
    acl resource initialization
    """
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    #ACL resource initialization    
    acl_resource = AclLiteResource()
    acl_resource.init()
    
    model = AclLiteModel(model_path)
    images_list = [os.path.join(INPUT_DIR, img)
                   for img in os.listdir(INPUT_DIR)
                   if os.path.splitext(img)[1] in IMG_EXT]

    for pic in images_list:
        inference(pic)
    print("Execute end")

if __name__ == '__main__':
    main()
