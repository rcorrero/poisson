import os
import errno
import json
import gdal

from googleapiclient.http import MediaFileUpload

from google.cloud import storage

# Create the service client
from googleapiclient.discovery import build
from apiclient.http import MediaIoBaseDownload


GOOGLE_APPLICATION_CREDENTIALS = os.getenv('APPLICATION_CREDENTIALS')
BUCKET_NAME = os.getenv('BUCKET_NAME')
GEO_FILTER_PATH = os.getenv('GEO_FILTER_PATH')
PATH_PREFIX = os.getenv('PATH_PREFIX')
ORDER_ID = os.getenv('ORDER_ID')
ITEM_TYPE = os.getenv('ITEM_TYPE')
ITEM_ID_PATH = os.getenv('ITEM_ID_PATH')
DL_IMAGE_PATH = os.getenv('DL_IMAGE_PATH')
BAND_ID = os.getenv('BAND_ID')


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
      print('Downloading image ', id_num, '...')
      print('Download Progress: ')
      done = False
      while not done:
          prog, done = media.next_chunk()
          print(prog.progress())

    print('Image ', id_num, ' downloaded.')
    return dl_path


def clip_img(img, id_num):
  img_cropped = img[:-4] + '_cropped.tif'
  if not os.path.exists(os.path.dirname(img_cropped)):
      try:
          os.makedirs(os.path.dirname(img_cropped))
      except OSError as exc: # Guard against race condition
          if exc.errno != errno.EEXIST:
              raise
  print('Clipping image ', id_num, '...')
  cmd = 'gdalwarp -of GTiff -cutline ' + GEO_FILTER_PATH + ' -crop_to_cutline '\
        + DL_IMAGE_PATH + img + ' ' + DL_IMAGE_PATH + img_cropped
  response = os.system(cmd)
  if response != 0:
      raise RuntimeError('Clip command exited with nonzero status. Status: ' \
                         + str(response))
  return img_cropped


def upload_img(img_clipped, item_id, ul_path, BUCKET_NAME):
    media = MediaFileUpload(img_clipped, 
                            mimetype='image/tif',
                            resumable=True)

    request = gcs_service.objects().insert(bucket=BUCKET_NAME, 
                                           name=ul_path,
                                           media_body=media)

    print('Uploading image ', id_num, '...')
    response = None
    while response is None:
        # _ is a placeholder for a progress object that we ignore.
        # (Our file is small, so we skip reporting progress.)
        _, response = request.next_chunk()
        print('Upload complete')
    return response


if __name__ == '__main__':
    inpath = r'' + PATH_PREFIX +  ORDER_ID + '/' + ITEM_TYPE + '/'
    with open(ITEM_ID_PATH) as f:
        item_ids = f.read().splitlines()
    for id_num, item_id in enumerate(item_ids):
        dl_path = r'' + inpath + item_id + BAND_ID + '.tif'
        ul_path = r'' + PATH_PREFIX +  ORDER_ID + '/clipped/' \
            + ITEM_TYPE + '/' + item_id + BAND_ID + '.tif'
        img = download_img(dl_path, id_num)
        img_clipped = clip_img(img, id_num)
        response = upload_img(img_clipped, item_id, ul_path, BUCKET_NAME)
        #print(response)
    print('Done.')
