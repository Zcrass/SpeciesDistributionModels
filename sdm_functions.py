import numpy as np
import geopandas as gpd
import os
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon


class sdm_functions:
    def list_files(folder):   
        '''
        Creates a dictionary with the name and path of the files foun in a certain folder.
        
        Args:
            folder: Folder to look for files.
            
        Returns:
            A dictionay with the base names and paths of the files found in the folder.
        ''' 
        variables_files = os.listdir(folder)
        variables = {}
        for key in variables_files:
                variables[key] = folder + key
        return variables


    def Random_Points_in_polygon(polygon, number, crs):   
        '''
        Generate random points whitin an extent.
        
        Modified from:https://www.matecdev.com/posts/random-points-in-polygon.html
        
        Args:
            polygon: extension where the points will be generated.
            number: number of point to generate.
            crs: CRS of the resulting points.
        
        Returns:
            A geopandas dataframe with points geometries.
        '''
        df = pd.DataFrame()
        df["decimalLongitude"] = np.random.uniform( polygon.bounds["minx"], polygon.bounds["maxx"], number )
        df["decimalLatitude"] = np.random.uniform( polygon.bounds["miny"], polygon.bounds["maxy"], number )
        df["presence"] = False ### absence
        df = gpd.GeoDataFrame(df, crs=crs, geometry=gpd.points_from_xy(df["decimalLongitude"], df["decimalLatitude"]))
        df = gpd.overlay(df, polygon, how="intersection")
        return df


    def mask_raster(raster_path, mask_poly, crs, out_raster):
        '''
        Mask raster according to polygon.
        
        Args:
            raster_path: path to input raster file.
            mask_poly: polygon to use as mask.
            crs: CRS of the result raster.
            out_raster: path to the resulting raster file.
        
        Returns:
            An masked raster.
        '''
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
        '''
        Create a new raste using another raster as base and a numpy array of data.
        
        Args:
            base_raster: Raster file used as base for the new raster.
            new_vals: A numpy array of values for the new raster.
            
        Returns:
            A new raster with the attributes of the base raster and the values of the numpy array.
        '''
        base_raster = rasterio.open(base_raster)
        results_raster = base_raster.read(1)
        results_transform = base_raster.transform
        results_raster = new_vals.reshape(results_raster.shape)
        return results_raster, results_transform
    

    def extent_poly(df, margin_size, crs):
        '''
        Creates a geopandas dataframe based on the extenssion of an input dataframe and a defined margin.
        Args:
            df: Aandas dataframe with decimalLongitude and decimalLatitude columns.
            margin_size: Margin drawn around the points. Values is based on the CRS used.
            crs: CRS to use in the geopandas dataframe.
            
        Returns:
            A new raster with the attributes of the base raster and the values of the numpy array.
        '''
        extent = [[(df.decimalLongitude.min() - float(margin_size)),
                        (df.decimalLatitude.min()  - float(margin_size))],
                  [(df.decimalLongitude.min() - float(margin_size)),
                        (df.decimalLatitude.max()  + float(margin_size))],
                  [(df.decimalLongitude.max() + float(margin_size)),
                        (df.decimalLatitude.max()  + float(margin_size))],
                  [(df.decimalLongitude.max() + float(margin_size)),
                        (df.decimalLatitude.min()  - float(margin_size))]
                 ]
        extent = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[Polygon(extent)])
        return extent


    def cor_test(df, thr):
        '''
        Computes the correlation matrix and extract the name of the correlated variables.
        Args:
            df: dataframe to compute correlation.
            thr: threshold to use as cut off to define the high correlation.
            
        Returns:
            A list of the correlated variables.
        '''    
        cor_mat = df.corr().abs()
        cor_mat = cor_mat.where(np.triu(np.ones(cor_mat.shape),k=1).astype(bool))
        return [column for column in cor_mat.columns if any(cor_mat[column] > thr)]
