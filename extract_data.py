#!/usr/bin/env python

import argparse
import logging as lg
import pandas as pd
import sys


from sdm_functions import sdm_functions as fun

if __name__ == "__main__":
        ### define logger
        lg.basicConfig(filename='cleaning.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
        logger = lg.getLogger('cleaning')
        logger.setLevel(lg.INFO)
        
        stdout_handler = lg.StreamHandler(sys.stdout)
        stdout_handler.setLevel(lg.INFO)
        logger.addHandler(stdout_handler)
        
        ### define arguments 
        parser = argparse.ArgumentParser(prog = 'cleaning.py', description = 'Program to extract localities from GBIF file and values from raster variables')
        parser.add_argument('-i', '--gbif_file') ### csv file downloades from GBIF
        parser.add_argument('-o', '--output_file')
        parser.add_argument('-v', '--vars_folder')
        args = parser.parse_args()
        
        ##################################################
        ##### START 
        ##################################################
        ### extract species coord
        logger.info(f'Reading GBIF file...')
        sp = pd.read_csv(args.gbif_file, sep='\t', usecols=['scientificName', 'decimalLatitude', 'decimalLongitude']).dropna().drop_duplicates()
        sp['presence'] = True
        logger.info(f'Keeping {sp.shape[0]} valid localities')

        ##################################################
        logger.info(f'Reading variable files...')
        variables = fun.list_rasters(args.vars_folder)
        logger.info(f'Found {len(variables)} in variables folder')

        ### load each raster and extract values by each point and from all the layer
        logger.info(f'Loading rasters...')
        var_names = list(variables.keys())
        raster_df = pd.DataFrame() 

        logger.info(f'Extracting values...')
        coords = [(x,y) for x, y in zip(sp.decimalLongitude, sp.decimalLatitude)]
        for var in var_names:
            raster = rasterio.open(variables[var])
            sp[var] = [x[0] for x in raster.sample(coords)]
            print(raster.sample(coords))
        
        logger.info(f'Saving to file {args.output_file}...')
        sp.to_csv(args.output_file, index=False)

        logger.info(f'Done!')
        