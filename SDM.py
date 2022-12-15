import rasterio
import pandas as pd
import matplotlib.pyplot as plt
from descartes import PolygonPatch
from shapely.geometry import Polygon
import geopandas as gpd  

locs = pd.read_csv("localities.csv")
edos_shp = "dest20gw.zip"
margin = 50 ### margin around the points in km
buffer_size = 10 ### buffer around the points in km to avoid seudo absenses
crs = "epsg:4326"


### extract species coord
sp = locs.loc[locs["species"] == "species_01", :].dropna(axis=0)
points = gpd.GeoDataFrame(sp, crs=crs, geometry=gpd.points_from_xy(sp.longitude, sp.latitude))
# coords = [(x,y) for x, y in zip(sp.longitude, sp.latitude)]

### import shapefiles
edos_poly = gpd.read_file(edos_shp)

##### create area limits 
limits = [(sp.longitude.min() - 1/111*margin),
          (sp.latitude.min()  - 1/111*margin),
          (sp.longitude.max() + 1/111*margin),
          (sp.latitude.max()  + 1/111*margin)]

### create pseudo absences
# buf = points.to_crs("epsg:32633").buffer(buffer_size*1000)
buf = points.buffer(1/111*buffer_size)
# print(type(buf))
buf = gpd.GeoDataFrame(crs=crs, geometry=buf.to_crs(crs))

extent = [[limits[0], limits[1]], 
          [limits[0], limits[3]],
          [limits[2], limits[3]],
          [limits[2], limits[1]]]
extent_poly = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[Polygon(extent)])

absences_poly = gpd.overlay(extent_poly, buf, how="difference")
### cut extent with 
### intersectin with extent and cut buffer
### then random point whitin extent




### extract data from raster
# src = rasterio.open('worldclim_2.1_30s_elev/wc2.1_30s_elev.tif')
# sp['elevation'] = [x[0] for x in src.sample(coords)]
### iterar en más variables
### modelos


fig, ax = plt.subplots()
plt.xlim([limits[0], limits[2]])
plt.ylim([limits[1], limits[3]])
ax.set_aspect('equal')
edos_poly.plot(ax = ax, color = 'gray', edgecolor = 'black')
# buf.plot(ax = ax, color = 'cyan', edgecolor = 'black')
# extent_poly.plot(ax = ax, color = 'cyan', edgecolor = 'black')
absences_poly.plot(ax = ax, color = 'cyan', edgecolor = 'black')
points.plot(ax = ax, color = 'red', edgecolor = 'black')

# ax.scatter(sp.longitude, sp.latitude, color = "red")
plt.show()