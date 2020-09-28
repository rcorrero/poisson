import json
import os
import os.path
from osgeo import ogr, osr, gdal
import matplotlib.pyplot as plt
import geojson as gj


def plot_img(img_array, figsize=(20, 20)):
    fig,ax = plt.subplots(1, figsize=figsize)
    # Display the image
    ax.imshow(img_array)

# Enable GDAL/OGR exceptions
gdal.UseExceptions()


# GDAL & OGR memory drivers
GDAL_MEMORY_DRIVER = gdal.GetDriverByName('MEM')
OGR_MEMORY_DRIVER = ogr.GetDriverByName('Memory')


def get_geo_info(ds):
    xoff, a, b, yoff, d, e = ds.GetGeoTransform()

    # Get projection information from source image
    ds_proj = ds.GetProjectionRef()
    ds_srs = osr.SpatialReference(ds_proj)

    # Get the source image's geographic coordinate system (the 'GEOGCS' node of ds_srs)
    geogcs = ds_srs.CloneGeogCS()

    # Set up a transformation between projected coordinates (x, y) & geographic coordinates (lat, lon)
    transform = osr.CoordinateTransformation(ds_srs, geogcs)
    return (xoff, a, b, yoff, d, e), transform


def load_img(img_path):
  '''Returns both the relevant geographic data, and other metadata such as
   cloudcover etc. and the img itself as a numpy array.
  '''
  ds = gdal.Translate('',
                img_path,
                options='-ot Byte -scale minval maxval -of MEM')
  (xoff, a, b, yoff, d, e), transform = get_geo_info(ds)
  return ds, (xoff, a, b, yoff, d, e), transform


def cut_by_geojson(input_file, output_file, shape_geojson):
    # Get coords for bounding box
    reader = ogr.Open(shape_geojson)
    layer = reader.GetLayer()
    feature = layer.GetFeature(0)
    geoms = json.loads(feature.ExportToJson())['geometry']
    x, y = zip(*gj.utils.coords(geoms))

    min_x, max_x, min_y, max_y = min(x), max(x), min(y), max(y)

    # Open original data as read only
    dataset = gdal.Open(input_file, gdal.GA_ReadOnly)

    bands = dataset.RasterCount

    # Getting georeference info
    transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]

    # Getting spatial reference of input raster
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)

    # WGS84 projection reference
    OSR_WGS84_REF = osr.SpatialReference()
    OSR_WGS84_REF.ImportFromEPSG(4326)

    # OSR transformation
    wgs84_to_image_trasformation = osr.CoordinateTransformation(OSR_WGS84_REF,
                                                                srs)
    XYmin = wgs84_to_image_trasformation.TransformPoint(min_x, max_y)
    XYmax = wgs84_to_image_trasformation.TransformPoint(max_x, min_y)

    # Computing Point1(i1,j1), Point2(i2,j2)
    i1 = int((XYmin[0] - xOrigin) / pixelWidth)
    j1 = int((yOrigin - XYmin[1]) / pixelHeight)
    i2 = int((XYmax[0] - xOrigin) / pixelWidth)
    j2 = int((yOrigin - XYmax[1]) / pixelHeight)
    new_cols = i2 - i1 + 1
    new_rows = j2 - j1 + 1

    # New upper-left X,Y values
    new_x = xOrigin + i1 * pixelWidth
    new_y = yOrigin - j1 * pixelHeight
    new_transform = (new_x, transform[1], transform[2], new_y, transform[4],
                     transform[5])

    wkt_geom = ogr.CreateGeometryFromJson(str(geoms))
    wkt_geom.Transform(wgs84_to_image_trasformation)

    target_ds = GDAL_MEMORY_DRIVER.Create('', new_cols, new_rows, 1,
                                          gdal.GDT_Byte)
    target_ds.SetGeoTransform(new_transform)
    target_ds.SetProjection(projection)

    # Create a memory layer to rasterize from.
    ogr_dataset = OGR_MEMORY_DRIVER.CreateDataSource("shapefile")
    ogr_layer = ogr_dataset.CreateLayer("shapefile", srs=srs)
    ogr_feature = ogr.Feature(ogr_layer.GetLayerDefn())
    ogr_feature.SetGeometryDirectly(ogr.Geometry(wkt=wkt_geom.ExportToWkt()))
    ogr_layer.CreateFeature(ogr_feature)

    gdal.RasterizeLayer(target_ds, [1], ogr_layer, burn_values=[1],
                        options=["ALL_TOUCHED=TRUE"])

    # Create output file
    driver = gdal.GetDriverByName('GTiff')
    outds = driver.Create(output_file, new_cols, new_rows, bands,
                          gdal.GDT_Float32)

    # Read in bands and store all the data in bandList
    mask_array = target_ds.GetRasterBand(1).ReadAsArray()
    band_list = []

    for i in range(bands):
        band_list.append(dataset.GetRasterBand(i + 1).ReadAsArray(i1, j1,
                         new_cols, new_rows))

    for j in range(bands):
        data = np.where(mask_array == 1, band_list[j], mask_array)
        outds.GetRasterBand(j + 1).SetNoDataValue(0)
        outds.GetRasterBand(j + 1).WriteArray(data)

    outds.SetProjection(projection)
    outds.SetGeoTransform(new_transform)

    target_ds = None
    dataset = None
    outds = None
    ogr_dataset = None

    return output_file


def clip_img(ds, img_path, aoi, coords, transform, plot_image=False,
             figsize=(20, 20), verbose=False):
  (xoff, a, b, yoff, d, e) = coords
  if aoi is not None:
      outpath = r'/clipped/' + img_path
      if not os.path.exists(os.path.dirname(outpath)):
          try:
              os.makedirs(os.path.dirname(outpath))
          except OSError as exc: # Guard against race condition
              if exc.errno != errno.EEXIST:
                  raise
      # Make gdal options
      if verbose:
        print('Clipping image...')
      new_img_path = cut_by_geojson(img_path, outpath, aoi)
      if verbose:
        print('Image clipped.')
      ds = gdal.Translate('',
                new_img_path,
                options='-ot Byte -scale minval maxval -of MEM')
      (xoff, a, b, yoff, d, e), transform = get_geo_info(ds)

  b1, b2, b3, b4 = (ds.GetRasterBand(1).ReadAsArray(), 
                  ds.GetRasterBand(2).ReadAsArray(), 
                  ds.GetRasterBand(3).ReadAsArray(), 
                  ds.GetRasterBand(4).ReadAsArray())
  ds = None
  img_array = np.array((b3,b2,b1))
  img_array = np.moveaxis(img_array, 0, 2)

  if plot_image:
    plot_img(img_array, figsize)
  
  return img_array, (xoff, a, b, yoff, d, e), transform


def split_img(img_array, patch_shape=None, xoffset=-110, yoffset=-110):
  if patch_shape is None:
    return img_array
  img_shape = img_array.shape
  width, height = img_shape[:2]
  patch_width, patch_height = patch_shape[:2]
  width_remainder = width % patch_width
  height_remainder = height % patch_height

  pad_w = patch_width - width_remainder
  pad_h = patch_height - height_remainder

  half_pad_w = pad_w / 2
  half_pad_w_1 = int(math.ceil(half_pad_w))
  half_pad_w_2 = int(math.floor(half_pad_w))

  half_pad_h = pad_h / 2
  half_pad_h_1 = int(math.ceil(half_pad_h))
  half_pad_h_2 = int(math.floor(half_pad_h))
    
  img_padded = np.pad(img_array, 
                    [[half_pad_w_1, half_pad_w_2], 
                     [half_pad_h_1, half_pad_h_2], 
                     [0, 0]],
                    'constant', constant_values=(0))
  patches = view_as_blocks(img_padded, patch_shape)

  n_horz_patches = patches.shape[1] # Number of patches along x-axis
  n_ver_patches = patches.shape[0] # Number of patches along y-axis

  for i in range(n_ver_patches):
    my = i * patch_height + yoffset - half_pad_w_1
    for j in range(n_horz_patches):
      mx = j * patch_width + xoffset - half_pad_h_1
      yield (i, j), patches[i, j, :, :, :, :], mx, my
  patches = None


def get_img_paths(dirpath, bucket_name, dl=True, max_num_imgs=None, skip_first=0):
  client = storage.Client()
  id = 0
  for blob in client.list_blobs(bucket_name, prefix=dirpath):
    if max_num_imgs is not None:
      if id > max_num_imgs - 1:
        break
    filepath = blob.name
    if filepath[-14:] == 'AnalyticMS.tif':
      if id < skip_first:
        id += 1
        continue
      if dl == True:
        if not os.path.isfile(filepath):
            download_img(filepath, id)
      id += 1
      yield filepath, id
  

def delete_img(img_path, idx):
    if os.path.isfile(img_path):
        os.remove(img_path)
        print('Image %s deleted.' % idx)
    else:
      print('Error: %s file not found.' % img_path)



def prep_imgs(dirpath, bucketname, patch_shape=None, aoi=None, id_start_idx=55, 
              id_end_idx=-18, max_num_imgs=25, skip_first=0, plot_image=False, 
              figsize=(20,20), print_clip=False, delete_img_after=False):
    img_paths = get_img_paths(dirpath, bucketname, max_num_imgs=max_num_imgs + skip_first, skip_first=skip_first)
    for img_path, idx in img_paths:
        ds = None
        img_clipped = None
        imgs_split = None
        item_id = img_path[id_start_idx:id_end_idx]
        try:
            ds, (xoff, a, b, yoff, d, e), transform = load_img(img_path)
        except:
            print('There was a problem loading image %s. Skipping.' % idx)
            continue
        img_clipped, (xoff, a, b, yoff, d, e), transform = clip_img(ds, img_path, 
                                                                    aoi, 
                                                                    (xoff, a, b, yoff, d, e), 
                                                                    transform,
                                                                    plot_image,
                                                                    figsize=figsize,
                                                                    verbose=print_clip)
        if patch_shape is None:
            return img_clipped, item_id, None, (xoff, a, b, yoff, d, e), transform
        if delete_img_after:
            delete_img(img_path, idx)
        imgs_split = split_img(img_clipped, patch_shape)
        for (i, j), split, mx, my in imgs_split:
            subsample_id = str(i) + '_' + str(j)
            yield split[0,:,:,:], item_id, subsample_id, (xoff, a, b,
                                                          yoff, d, e), \
                                                          transform, mx, my
    ds = None
