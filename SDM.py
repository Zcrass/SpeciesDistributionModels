from rasterio.plot import show
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import geopandas as gpd  
import random 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import rioxarray
from sdm_functions import Random_Points_in_polygon, mask_raster, resulting_raster
import numpy as np

if __name__ == "__main__":
    ### define parameters
    locs = pd.read_csv("localities.csv")
    species_name = "species_02"
    margin = 1/111*50 ### margin around the points (units according to crs)
    buffer_size = 1/111*10 ### buffer around the points to avoid seudo absenses (units according to crs)
    crs = "epsg:4326"
    
    variables = {"elevation":"worldclim_2.1_30s_elev/wc2.1_30s_elev.tif",
                 "mean_temp":"worldclim_2.1_30s_bio/wc2.1_30s_bio_1.tif",
                 "annu_prec":"worldclim_2.1_30s_bio/wc2.1_30s_bio_12.tif",
                }
    
    # var_names = ["mean_temp", "annu_prec"]
    # var_names = ["elevation"]

    ##################################################
    ##### START 
    ##################################################
    ### extract species coord
    sp = locs.loc[locs["species"] == species_name, :].dropna(axis=0)
    sp["point"] = 1 ### presence
    pres_points = gpd.GeoDataFrame(sp, crs=crs, geometry=gpd.points_from_xy(sp.longitude, sp.latitude))

    ##### create area limits 
    limits = [(sp.longitude.min() - margin),
            (sp.latitude.min()  - margin),
            (sp.longitude.max() + margin),
            (sp.latitude.max()  + margin)]

    ### create pseudo absences
    buf = pres_points.buffer(buffer_size)
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
    var_names = list(variables.keys())
    raster_df = pd.DataFrame()
    
    for var in var_names:
        masked = mask_raster(variables[var], extent_poly, crs, "tmp.tif")
        all_points[var] = [x[0] for x in masked.sample(coord_list)]
        var_df = pd.DataFrame(np.array(masked.read()).reshape([1,-1]).T)
        raster_df = pd.concat([raster_df, var_df], axis=1)
    ### load raster 
#     elev = mask_raster("worldclim_2.1_30s_elev/wc2.1_30s_elev.tif", extent_poly, crs, "worldclim_2.1_30s_elev/wc2.1_30s_elev_tmp.tif")
#     temp = mask_raster("worldclim_2.1_30s_bio/wc2.1_30s_bio_1.tif", extent_poly, crs, "worldclim_2.1_30s_bio/wc2.1_30s_bio_1_tmp.tif")
#     prec = mask_raster("worldclim_2.1_30s_bio/wc2.1_30s_bio_12.tif", extent_poly, crs, "worldclim_2.1_30s_bio/wc2.1_30s_bio_12_tmp.tif")

    ### extract data from raster
    
#     coord_list = [(x,y) for x,y in zip(all_points["geometry"].x , all_points["geometry"].y)]

#     all_points[var_names[0]] = [x[0] for x in elev.sample(coord_list)]
#     all_points[var_names[1]] = [x[0] for x in temp.sample(coord_list)]
#     all_points[var_names[2]] = [x[0] for x in prec.sample(coord_list)]

    ### split dataset
    split_points = random.sample(range(len(all_points.index)), round(len(all_points.index)*0.25))
    train_x = all_points.iloc[split_points, :].loc[:, var_names]
    train_y = all_points.iloc[split_points, :].loc[:, "point"].to_list()
    test_points = all_points.loc[~all_points.index.isin(split_points), :] 
    test_x = all_points.loc[~all_points.index.isin(split_points), var_names] 
    test_y = all_points.loc[~all_points.index.isin(split_points), "point"].to_list() 

    ### train model
    rf_model = RandomForestRegressor(n_estimators = 1000, criterion = "absolute_error", max_depth = None,
                                    oob_score = False, n_jobs = -1, bootstrap=True, random_state = 123)
    rf_model.fit(train_x, train_y)

    ### predict test and compute 
    test_prediction = rf_model.predict(X = test_x)
    rmse = mean_squared_error(y_true = test_y, y_pred = test_prediction, squared = False)
    print(rmse)
    
    ### predict to complete rasters
#     raster_df = pd.DataFrame()
#     for var in var_names:
        # var_df = pd.DataFrame(np.array(variables[var].read()).reshape([1,-1]).T)
        # raster_df = pd.concat([raster_df, var_df], axis=1)
            
#     raster_df = pd.concat([raster_df, pd.DataFrame(np.array(elev.read()).reshape([1,-1]).T)], axis=1)
#     raster_df = pd.concat([raster_df, pd.DataFrame(np.array(temp.read()).reshape([1,-1]).T)], axis=1)
#     raster_df = pd.concat([raster_df, pd.DataFrame(np.array(prec.read()).reshape([1,-1]).T)], axis=1)
    raster_df.columns = var_names
    raster_prediction = rf_model.predict(X = raster_df)
    
    ### proyect results
#     print(variables[var_names[1]])
#     results_raster = rasterio.open(variables[var_names[1]]).read(1)
#     results_raster = variables[var_names[1]].read(1)
#     results_transform = results_raster.transform
#     results_raster = raster_prediction.reshape(results_raster.shape)
#     print(results_transform)
    results_raster, results_transform = resulting_raster("tmp.tif", raster_prediction)
    

    ##################################################
    ##### FIGURE
    ##################################################
    fig, ax = plt.subplots()
    plt.xlim([limits[0], limits[2]])
    plt.ylim([limits[1], limits[3]])
    ax.set_aspect("equal")
    # show(elev.read(1), ax = ax, transform=elev.transform, cmap="gray") 
    show(results_raster, ax = ax, transform=results_transform, cmap="gray") 
    pres_points.plot(ax = ax, color="green", edgecolor = "black")
    # all_points.plot(ax = ax, c = all_points.point, edgecolor = "black")
    # test_points.plot(ax = ax, c = test_prediction, edgecolor = "black")
    
    plt.show()