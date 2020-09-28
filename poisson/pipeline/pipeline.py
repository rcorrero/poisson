from datetime import datetime
from utils import *
from preprocess import *
from postprocess import *


# GOOGLE CREDS is different for notebook – change for .py files
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
GOOGLE_CREDENTIALS = os.getenv('GOOGLE_CREDENTIALS')
BUCKET_NAME = os.getenv('BUCKET_NAME')
DICT_FP_NAME = os.getenv('DICT_FP_NAME')
GEO_FILTER_PATH = os.getenv('GEO_FILTER_PATH')
PATH_PREFIX = os.getenv('PATH_PREFIX')
ITEM_TYPE = os.getenv('ITEM_TYPE')
ITEM_ID_PATH = os.getenv('ITEM_ID_PATH')
DL_IMAGE_PATH = os.getenv('DL_IMAGE_PATH')
BAND_ID = os.getenv('BAND_ID')
PL_API_KEY = os.getenv('PL_API_KEY')
PL_USER = os.getenv('PL_USER')
PL_PASSWORD = os.getenv('PL_PASSWORD')
ORDER_NAME = os.getenv('ORDER_NAME')


if __name__ == '__main__':
    dict_dl_path = DICT_FP_NAME
    state_dict = download_img(dict_dl_path, 0)
    
    model = get_instance_segmentation_model(2)
    model.load_state_dict(torch.load(state_dict))
    device = torch.device('cuda')
    model.to(device)
    
    thresh=0.50
    dirpath = input('Path to image directory: ')
    aoi = None
    imgs = prep_imgs(dirpath, BUCKET_NAME, (768, 768, 3), aoi=aoi, id_start_idx=65, id_end_idx=-18,
                     max_num_imgs=np.float('inf'), skip_first=0, plot_image=False, print_clip=False,
                     delete_img_after=True)
    rows = []
    model.eval()
    idx = 0
    for img, item_id, subsample_id, (xoff, a, b, yoff, d, e), transform, mx, my in imgs:
        # Label the image
        image = F.to_tensor(img)
        with torch.no_grad():
            try:
                prediction = model([(image).to(device)])
            except RuntimeError:
                print('RuntimeError occured. Skipping image.')
                continue
            # Add data to df list
            new_rows = process_outputs(prediction, item_id, subsample_id, 
                                       (xoff, a, b, yoff, d, e), transform, mx, my,
                                       thresh=thresh)
            if new_rows is not None:
                rows += new_rows
                if idx % 1000 == 0:
                    print('Total number of targets identified so far: %s.' % len(rows))
                    if idx % 10000 == 0:
                        plot_bbox(prediction, img, thresh=thresh)
            idx += 1

    df = make_df(rows)
    
    # load GeoJSON file containing sectors
    with open(GEO_FILTER_PATH) as f:
        aoi = json.load(f)

    select_rows = df.apply(filter_fn, axis=1, aoi=aoi)
    df_filtered = df[select_rows]

    if make_chart:
        # Make chart
        datetimes_less_30 = df_filtered[df_filtered.major_length < 30]['datetime']
        datetimes = df_filtered['datetime']
        fig, ax = plt.subplots(1,1, figsize=(16, 9))
        ax.hist(datetimes_less_30, bins=len(np.unique(df_filtered.datetime)), alpha=0.5, label='Less than 30 meters', color='blue')
        ax.hist(datetimes, bins=len(np.unique(df_filtered.datetime)), alpha=0.5, label='All sizes', color='grey')
        ax.set_title('Vessel Counts')
        ax.set_xlabel('Date')
        ax.set_ylabel('Counts')
        plt.legend(loc='upper left')
        plt.show()
        fig.savefig()

    df.to_csv(UNFILTERED_PATH)
    df_filtered.to_csv(FILTERED_PATH)
    print('Done.')
