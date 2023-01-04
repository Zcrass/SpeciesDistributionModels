#!/usr/bin/env python

import argparse
import geopandas as gpd  
import logging as lg
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random 
from rasterio.plot import show
from shapely.geometry import Polygon
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import sys

from sdm_functions import Random_Points_in_polygon, mask_raster, resulting_raster

if __name__ == "__main__":
        ### define logger
        lg.basicConfig(filename='sdm.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
        logger = lg.getLogger('sdm')
        logger.setLevel(lg.INFO)
        
        stdout_handler = lg.StreamHandler(sys.stdout)
        stdout_handler.setLevel(lg.INFO)
        logger.addHandler(stdout_handler)
        
        ### define arguments 
        parser = argparse.ArgumentParser(prog = 'RandomForestSDM', description = 'Program to perform an SDM using Random Forest')
        parser.add_argument('-l', '--locs_file') ### csv file with columns ["longitude" "latitude"]
        parser.add_argument('-m', '--margin_size')
        parser.add_argument('-b', '--buffer_size')
        parser.add_argument('-v', '--vars_folder')
        args = parser.parse_args()
        crs = "epsg:4326"
        
        ##################################################
        ##### START 
        ##################################################
        ### extract species coord
        logger.info(f'Reading localities...')
        sp = pd.read_csv(args.locs_file).dropna(axis=0)    
        sp["point"] = 1 ### presence
        pres_points = gpd.GeoDataFrame(sp, crs=crs, geometry=gpd.points_from_xy(sp.longitude, sp.latitude))
        logger.info(f'Loaded {sp.shape[0]} valid localities')

        ##################################################
        ### list rasters layers
        logger.info(f'Reading variable files...')
        variables_files = os.listdir(args.vars_folder)
        variables = {}
        for key in variables_files:
                variables[key] = args.vars_folder + key
        logger.info(f'Found {len(variables_files)} in variables folder')
                
        ##################################################
        ##### create area limits 
        logger.info(f'Defining area of interest...')
        limits = [(sp.longitude.min() - float(args.margin_size)),
                  (sp.latitude.min()  - float(args.margin_size)),
                  (sp.longitude.max() + float(args.margin_size)),
                  (sp.latitude.max()  + float(args.margin_size))]

        ### create pseudo absences
        logger.info(f'Creating (pseudo)absences...')
        buf = pres_points.geometry.buffer(float(args.buffer_size))
        buf = gpd.GeoDataFrame(crs=crs, geometry=buf.to_crs(crs))
        
        extent = [[limits[0], limits[1]], 
                [limits[0], limits[3]],
                [limits[2], limits[3]],
                [limits[2], limits[1]]]
        extent_poly = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[Polygon(extent)])
        absences_poly = gpd.overlay(extent_poly, buf, how="difference")
        ### random point whitin extent
        absences_points = Random_Points_in_polygon(absences_poly, 1000, crs)
        ### concat all points
        all_points = pd.concat([pres_points[["point", "longitude", "latitude", "geometry"]],
                                absences_points[["point", "longitude", "latitude", "geometry"]]])
        coord_list = [(x,y) for x,y in zip(all_points["geometry"].x , all_points["geometry"].y)]
        ##################################################
        ### load each raster and extract values by each point and from all the layer
        logger.info(f'Loading rasters...')
        var_names = list(variables.keys())
        raster_df = pd.DataFrame() 
        
        for var in var_names:
                logger.info(f'Loading {var} file...')
                masked = mask_raster(variables[var], extent_poly, crs, "tmp.tif")
                all_points[var] = [x[0] for x in masked.sample(coord_list)]
                var_df = pd.DataFrame(np.array(masked.read()).reshape([1,-1]).T)
                raster_df = pd.concat([raster_df, var_df], axis=1)
                
        ### split dataset into train and test
        logger.info(f'Spliting train and test datasets...')
        split_points = random.sample(range(len(all_points.index)), round(len(all_points.index)*0.25))
        train_x = all_points.iloc[split_points, :].loc[:, var_names]
        train_y = all_points.iloc[split_points, :].loc[:, "point"].to_list()
        # test_points = all_points.loc[~all_points.index.isin(split_points), :] 
        test_x = all_points.loc[~all_points.index.isin(split_points), var_names] 
        test_y = all_points.loc[~all_points.index.isin(split_points), "point"].to_list() 

        ### train model
        logger.info(f'Training model...')
        rf_model = RandomForestRegressor(n_estimators = 1000, criterion = "absolute_error", max_depth = None,
                                        oob_score = False, n_jobs = -1, bootstrap=True, random_state = 123)
        rf_model.fit(train_x, train_y)

        ### predict test and compute rmse
        logger.info(f'Testing model...')
        test_prediction = rf_model.predict(X = test_x)
        rmse = mean_squared_error(y_true = test_y, y_pred = test_prediction, squared = False)
        logger.info(f'Model RMSE = {rmse}')
        
        
        ### extrapolate prediction to complete rasters
        logger.info(f'Expanding model to complete area of interest...')
        raster_df.columns = var_names
        raster_prediction = rf_model.predict(X = raster_df)
        
        ### proyect results
        results_raster, results_transform = resulting_raster("tmp.tif", raster_prediction)

        ##################################################
        ##### FIGURE
        ##################################################
        logger.info(f'Ploting...')
        fig, ax = plt.subplots()
        plt.xlim([limits[0], limits[2]])
        plt.ylim([limits[1], limits[3]])
        ax.set_aspect("equal")
        show(results_raster, ax = ax, transform=results_transform, cmap="gray") 
        pres_points.plot(ax = ax, color="green", edgecolor = "black")   
        plt.show()