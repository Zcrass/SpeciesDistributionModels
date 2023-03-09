#!/usr/bin/env python

import argparse
import geopandas as gpd
import logging as lg
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import decomposition
from sklearn.cluster import AgglomerativeClustering, KMeans
import sys


from sdm_functions import sdm_functions as fun

if __name__ == "__main__":
        ### define logger
        lg.basicConfig(filename='filter_points.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
        logger = lg.getLogger('filter_points')
        logger.setLevel(lg.INFO)
        
        stdout_handler = lg.StreamHandler(sys.stdout)
        stdout_handler.setLevel(lg.INFO)
        logger.addHandler(stdout_handler)
        
        ### define arguments 
        parser = argparse.ArgumentParser(prog = 'filter_points.py', description = 'Program to filter poinst based on variables data')
        parser.add_argument('-i', '--data_file') ### csv file with decimalLatitude and decimalLongitude columns
        parser.add_argument('-o', '--output_file')
        parser.add_argument('-v', '--vars_folder')
        parser.add_argument('-m', '--method')
        args = parser.parse_args()
        
        ##################################################
        ##### START 
        ##################################################   
        ### extract species coord
        logger.info(f'Reading data file...')
        sp = pd.read_csv(args.data_file, sep=',').dropna().drop_duplicates()
        logger.info(f'Keeping {sp.shape[0]} valid localities')

        ### listing raster variables
        logger.info(f'Reading variable files...')
        variables = fun.list_rasters(args.vars_folder)
        logger.info(f'Found {len(variables)} in variables folder')
        var_names = list(variables.keys())

        ### compute PCA
        # data = sp[var_names]
        logger.info(f'computing PCA...')
        pca = decomposition.PCA()
        pca.fit(sp[var_names])
        pca_val = pd.DataFrame(pca.transform(sp[var_names]))
        
        ### compute k means
        if args.method == 'kmeans':
                logger.info(f'computing k means clustering...')
                model = KMeans(n_clusters=2, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
        if args.method == 'hclust':        
                model = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='average')
        sp['cluster'] = model.fit_predict(sp[var_names])
        logger.info(f'Saving clusters to file {args.output_file}...')
        sp.to_csv(args.output_file, index=False)
        

        ### MAP
        logger.info(f'Ploting...')
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        limits = [(sp.decimalLongitude.min() - (sp.decimalLongitude.max() - sp.decimalLongitude.min())*0.1),
                  (sp.decimalLatitude.min()  - (sp.decimalLatitude.max()  - sp.decimalLatitude.min())*0.1),
                  (sp.decimalLongitude.max() + (sp.decimalLongitude.max() - sp.decimalLongitude.min())*0.1),
                  (sp.decimalLatitude.max()  + (sp.decimalLatitude.max()  - sp.decimalLatitude.min())*0.1)]

        fig, ax = plt.subplots()
        plt.xlim([limits[0], limits[2]])
        plt.ylim([limits[1], limits[3]])
        ax.set_aspect("equal")
        world.boundary.plot(ax = ax, color='gray')
        scatter = ax.scatter(sp.decimalLongitude, sp.decimalLatitude, c=sp.cluster)
        ax.legend(*scatter.legend_elements())
        plt.show()


