import rasterio
from rasterio.mask import mask
# import rasterio.plot
from rasterio.plot import show
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from descartes import PolygonPatch
from shapely.geometry import Polygon
import geopandas as gpd  
import random 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, confusion_matrix
import rioxarray

locs = pd.read_csv("localities.csv")
edos_shp = "dest20gw.zip"
margin = 50 ### margin around the points in km
buffer_size = 10 ### buffer around the points in km to avoid seudo absenses
crs = "epsg:4326"
var_names = ["elevation", "mean_temp", "annu_prec"]

### extract species coord
sp = locs.loc[locs["species"] == "species_01", :].dropna(axis=0)
sp["point"] = 1 ### presence
pres_points = gpd.GeoDataFrame(sp, crs=crs, geometry=gpd.points_from_xy(sp.longitude, sp.latitude))

### import shapefiles
edos_poly = gpd.read_file(edos_shp)

##### create area limits 
limits = [(sp.longitude.min() - 1/111*margin),
          (sp.latitude.min()  - 1/111*margin),
          (sp.longitude.max() + 1/111*margin),
          (sp.latitude.max()  + 1/111*margin)]

### create pseudo absences
# buf = points.to_crs("epsg:32633").buffer(buffer_size*1000)
buf = pres_points.buffer(1/111*buffer_size)
# print(type(buf))
buf = gpd.GeoDataFrame(crs=crs, geometry=buf.to_crs(crs))

extent = [[limits[0], limits[1]], 
          [limits[0], limits[3]],
          [limits[2], limits[3]],
          [limits[2], limits[1]]]
extent_poly = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[Polygon(extent)])

absences_poly = gpd.overlay(extent_poly, buf, how="difference")
### random point whitin extent
def Random_Points_in_polygon(polygon, number, crs):   
    '''https://www.matecdev.com/posts/random-points-in-polygon.html'''
    df = pd.DataFrame()
    df["longitude"] = np.random.uniform( polygon.bounds["minx"], polygon.bounds["maxx"], number )
    df["latitude"] = np.random.uniform( polygon.bounds["miny"], polygon.bounds["maxy"], number )
    df["point"] = 0 ### absence
    gdf_points = gpd.GeoDataFrame(df, crs=crs, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]))
    res_points = gpd.overlay(gdf_points, polygon, how="intersection")
    return res_points

absences_points = Random_Points_in_polygon(absences_poly, 1000, crs)

### concat all points
all_points = pd.concat([pres_points[["point", "longitude", "latitude", "geometry"]],
                        absences_points[["point", "longitude", "latitude", "geometry"]]])

### load raster 
elev = rasterio.open('worldclim_2.1_30s_elev/wc2.1_30s_elev.tif')
# print(elev.bounds)
# elev, elev_transform = mask(elev, extent_poly.geometry, crop=True)
# elev_crop = rasterio.open('elev_crop.tif', 'w', driver='GTiff',
#                           height=elev.shape[0], width=elev.shape[1], count=1,
#                           dtype=elev.dtype, crs=crs,transform=elev_transform,

# )
# elev = rasterio.open("elev_crop.tif")

# print(elev.bounds)
temp = rasterio.open("worldclim_2.1_30s_bio/wc2.1_30s_bio_1.tif")
prec = rasterio.open("worldclim_2.1_30s_bio/wc2.1_30s_bio_12.tif")

### extract data from raster
coord_list = [(x,y) for x,y in zip(all_points['geometry'].x , all_points['geometry'].y)]

### iterar en más variables
all_points['elevation'] = [x[0] for x in elev.sample(coord_list)]
all_points['mean_temp'] = [x[0] for x in temp.sample(coord_list)]
all_points['annu_prec'] = [x[0] for x in prec.sample(coord_list)]
# print(all_points)

### split dataset
split_points = random.sample(range(len(all_points.index)), round(len(all_points.index)*0.25))
train_x = all_points.iloc[split_points, :].loc[:, var_names]
train_y = all_points.iloc[split_points, :].loc[:, "point"].to_list()
test_x = all_points.loc[~all_points.index.isin(split_points), var_names] 
test_y = all_points.loc[~all_points.index.isin(split_points), "point"].to_list() 

### train model
rf_model = RandomForestRegressor(n_estimators = 10, criterion = 'friedman_mse', max_depth = None,
                                 oob_score = False, n_jobs = -1, random_state = 123)
rf_model.fit(train_x, train_y)

### predict test and compute 
prediction = rf_model.predict(X = test_x)
rmse = mean_squared_error(y_true = test_y, y_pred = prediction, squared = False)
# con_matrx = confusion_matrix(y_true = test_y, y_pred = prediction)

# print(rmse)
### predict for the complete image
# rds = rioxarray.open_rasterio("worldclim_2.1_30s_elev/wc2.1_30s_elev.tif")
# # rds.name = "data"
# # df = rds.squeeze().to_dataframe().reset_index()
# # geometry = gpd.points_from_xy(df.x, df.y)
# # gdf = gpd.GeoDataFrame(df, crs=rds.rio.crs, geometry=geometry)

# print(rds)


fig, ax = plt.subplots()
plt.xlim([limits[0], limits[2]])
plt.ylim([limits[1], limits[3]])
ax.set_aspect('equal')
show(elev.read(1), transform=elev.transform, cmap='gray') 
# edos_poly.plot(ax = ax, color = 'gray', edgecolor = 'black')
# buf.plot(ax = ax, color = 'cyan', edgecolor = 'black')
# extent_poly.plot(ax = ax, color = 'cyan', edgecolor = 'black')
# absences_poly.plot(ax = ax, color = 'cyan', edgecolor = 'black')
# pres_points.plot(ax = ax, color = 'red', edgecolor = 'black')
# absences_points.plot(ax = ax, color = 'blue', edgecolor = 'black')
all_points.plot(ax = ax, c = all_points.point, edgecolor = 'black')
# ax.scatter(sp.longitude, sp.latitude, color = "red")
plt.show()