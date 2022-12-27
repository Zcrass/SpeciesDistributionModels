import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask


def Random_Points_in_polygon(polygon, number, crs):   
    '''https://www.matecdev.com/posts/random-points-in-polygon.html'''
    df = pd.DataFrame()
    df["longitude"] = np.random.uniform( polygon.bounds["minx"], polygon.bounds["maxx"], number )
    df["latitude"] = np.random.uniform( polygon.bounds["miny"], polygon.bounds["maxy"], number )
    df["point"] = 0 ### absence
    gdf_points = gpd.GeoDataFrame(df, crs=crs, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]))
    res_points = gpd.overlay(gdf_points, polygon, how="intersection")
    return res_points

def mask_raster(raster_path, mask_poly, crs, out_raster):
    raster = rasterio.open(raster_path)
    out_img, out_transform  = mask(raster, mask_poly.geometry, invert=False, crop=True)
    new_raster = rasterio.open(out_raster, 'w', driver='GTiff',
                            height = out_img.shape[1], width = out_img.shape[2],
                            count=1, dtype=str(out_img.dtype),
                            crs=crs,
                            transform=out_transform)
    new_raster.write(out_img)
    new_raster.close()
    new_raster = rasterio.open(out_raster)
    return new_raster

def resulting_raster(base_raster, new_vals):
    raster= rasterio.open(base_raster)
    results_raster = raster.read(1)
    results_transform = raster.transform
    results_raster = new_vals.reshape(results_raster.shape)
    return results_raster, results_transform