import os
import torchvision
from googleapiclient.discovery import build
from apiclient.http import MediaIoBaseDownload
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import json
from shapely.geometry import shape, Point


def filter_fn(row, aoi):
    point = Point(row.longitude, row.latitude)

    # check each polygon to see if it contains the point
    for feature in aoi['features']:
        polygon = shape(feature['geometry'])
        if polygon.contains(point):
            return True
    return False


def filter_fn_neg(row, aoi):
    point = Point(row.longitude, row.latitude)

    # check each polygon to see if it contains the point
    for feature in aoi['features']:
        polygon = shape(feature['geometry'])
        if polygon.contains(point):
            return False
    return True


def plot_bbox(prediction, img, thresh=0.5):
    bbox_list = get_bboxes(prediction[0], thresh)
    if len(bbox_list) < 1:
        return 

    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(10,10))

    # Display the image
    ax.imshow(img)


    for _, bbox in get_bboxes(prediction[0], thresh):
        x0, y0, x1, y1 = bbox
        
        center_x = x0 + ((x1 - x0) / 2)
        center_y = y0 + ((y1 - y0) / 2)
        # Create a Rectangle patch
        rect = patches.Rectangle((x0, y0), x1-x0, y1-y0,linewidth=1,edgecolor='r',facecolor='none')
        
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.scatter(center_x, center_y,color='r')

    plt.show()
    fig = None
    ax = None
    

def download_img(dl_path, id_num):
    gcs_service = build('storage', 'v1')
    if not os.path.exists(os.path.dirname(dl_path)):
        try:
            os.makedirs(os.path.dirname(dl_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(dl_path, 'wb') as f:
        # Download the file from the Google Cloud Storage bucket.
        request = gcs_service.objects().get_media(bucket=BUCKET_NAME,
                                                  object=dl_path)
        media = MediaIoBaseDownload(f, request)
        print('Downloading item ', id_num + 1, '...')
        print('Download Progress: ')
        done = False
        while not done:
            prog, done = media.next_chunk()
            print(prog.progress())

    print('Image ', id_num + 1, ' downloaded.')
    return dl_path


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model
