import pandas as pd

from datetime import datetime
from osgeo import gdal, osr, ogr


def make_df(rows=None):
    col_names = ['target_id', 'item_id', 'datetime', 'probability', 'latitude', 'longitude',
                'bounding_box', 'area', 'major_length', 'minor_length']
    df = pd.DataFrame(columns=col_names)
    #df.set_index('target_id')
    if rows is not None:
          df = df.append(rows)
    return df


def get_datetime(item_id):
    datetime_str = item_id[:15]
    datetime_obj = datetime.strptime(datetime_str, '%Y%m%d_%H%M%S')
    return datetime_obj


def get_bboxes(prediction, thresh=0.5):
    '''Selects only those bounding boxes associated with 
    probabilities greater than threshold.
    '''
    scores =  prediction['scores'].cpu().numpy()
    filtered_idx = np.where(scores > thresh)
    bboxes = prediction['boxes'].cpu().numpy()
    bboxes = bboxes[filtered_idx]
    return list(zip(scores[filtered_idx], bboxes))


def convert_px_to_lat_long(results, coords, transform, mx, my, xadj=330, yadj=-330):
    xoff, a, b, yoff, d, e = coords
    lat_long_list = []
    for _, bbox in results:
        x0, y0, x1, y1 = bbox
        center_x = x0 + ((x1 - x0) / 2)
        center_y = y0 + ((y1 - y0) / 2)
        
        # Get global coordinates from pixel x, y coords
        #projected_x = a * center_x + a * mx + xoff + b * center_y 
        #projected_y = d * center_y + d * my + yoff + e * center_y
        projected_x = a * (center_x + mx) + b * (center_y + my) + xoff + xadj
        projected_y = d * (center_x + mx) + e * (center_y + my) + yoff + yadj

        # Transform from projected x, y to geographic lat, lng
        lng, lat, _ = transform.TransformPoint(projected_x, projected_y)
        lat_long_list.append((lat, lng))
    return lat_long_list


def format_rows(rows, datetime, item_id, subsample_id, m_per_pix=3.0):
    row_list = []
    for idx, target in enumerate(rows):
        target_id = item_id + '_'  + subsample_id + '_' + str(idx)
        probability = target[0][0]
        latitude = target[1][0]
        longitude = target[1][1]

        bounding_box = target[0][1]
        x0, y0, x1, y1 = bounding_box
        diff_y = (y1 - y0) * m_per_pix
        diff_x = (x1 - x0) * m_per_pix
        area = diff_y * diff_x
        major_length = max(diff_y, diff_x)
        minor_length = min(diff_y, diff_x)

        row_list.append({'target_id':target_id, 'item_id':item_id, 'datetime':datetime, 
                   'probability':probability, 'latitude':latitude, 'longitude':longitude, 
                   'bounding_box':bounding_box, 'area':area, 'major_length':major_length, 
                   'minor_length':minor_length}) #ignore_index=True)
    return row_list
        
        
def process_outputs(prediction, item_id, subsample_id, coords, transform, mx, my,
                    thresh=0.5, m_per_pix=3.0):
    if prediction[0]['scores'].cpu().numpy().shape[0] < 1:
        return
    datetime = get_datetime(item_id)    
    results = get_bboxes(prediction[0], thresh)
    rows = list(zip(results, convert_px_to_lat_long(results, 
                                                    coords, 
                                                    transform, mx, my)))
    rows = format_rows(rows, datetime, item_id, subsample_id, m_per_pix=m_per_pix)
    return rows
