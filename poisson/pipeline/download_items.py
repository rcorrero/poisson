import json
import os
import pathlib
import time
import requests

from requests.auth import HTTPBasicAuth


PLANET_API_KEY = os.getenv('PL_API_KEY')
PLANET_USER = os.getenv('PL_USER')
PLANET_PASSWORD = os.getenv('PL_PASSWORD')
ORDER_NAME = os.getenv('ORDER_NAME')
ITEM_ID_PATH = os.getenv('ITEM_ID_PATH')
PATH_PREFIX = os.getenv('PATH_PREFIX')
ITEM_TYPE = os.getenv('ITEM_TYPE')
PRODUCT_BUNDLE = os.getenv('PRODUCT_BUNDLE')
GOOGLE_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
BUCKET_NAME = os.getenv('BUCKET_NAME')

orders_url = 'https://api.planet.com/compute/ops/orders/v2'
auth = HTTPBasicAuth(PLANET_API_KEY, '')
headers = {'content-type': 'application/json'}
user = PLANET_USER
password = PLANET_PASSWORD
name = ORDER_NAME
subscription_id = 0
item_type = ITEM_TYPE
product_bundle = PRODUCT_BUNDLE
single_archive = False
archive_filename = ARCHIVE_FILENAME
bucket = BUCKET_NAME
path_prefix = PATH_PREFIX
email = True
max_num_samples = 500


def create_request(user, password, name, subscription_id, item_ids, item_type, 
                   product_bundle, single_archive, archive_filename, 
                   bucket, credentials, path_prefix, email):
  request = {
      "name": name,
      "subscription_id": subscription_id,
      "products": [
        {
          "item_ids": item_ids,
          "item_type": item_type,
          "product_bundle": product_bundle
        }
      ],
      "delivery": {
        "single_archive": single_archive,
        #"archive_filename": archive_filename,
        "google_cloud_storage": {
          "bucket": bucket,
          "credentials": credentials,
          "path_prefix": path_prefix
        }
      },
      "notifications": {
        "email": email
      },
      "order_type": "full"
    }
  return request


def place_order(request, auth):
    response = requests.post(orders_url, data=json.dumps(request), auth=auth, headers=headers)
    print("Response ok? ", response.ok)
    if not response.ok:
      print(response.json())
    order_id = response.json()['id']
    print("Order id: ", order_id)
    order_url = orders_url + '/' + order_id
    return order_url


def poll_for_success(order_url, auth, num_loops=250):
    print('Order status: ')
    count = 0
    while(count < num_loops):
        count += 1
        r = requests.get(order_url, auth=auth)
        response = r.json()
        state = response['state']
        print(state)
        end_states = ['success', 'failed', 'partial']
        if state in end_states:
            break
        time.sleep(25)


if __name__ == '__main__':
  with open(ITEM_ID_PATH) as f:
      item_ids = f.read().splitlines()
      if len(item_ids) > max_num_samples:
        item_ids = item_ids[:max_num_samples]
      
  with open(GOOGLE_CREDENTIALS) as f:
      credentials = f.read()
      request = create_request(user, password, name, subscription_id, 
                               item_ids, item_type, product_bundle, single_archive, 
                               archive_filename, bucket, credentials, 
                               path_prefix, email)
      order_url = place_order(request, auth)
      poll_for_success(order_url, auth)
