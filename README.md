# Simple Species Distribution Model in Python
## Introduction
Species distribution model are...

We are going to produce an species distribution model using python and GBIF data.

### Species of interest
We selected a *Magnolia* species called *Magnolia iltisiana*. This is a Tree that inhabits in the western part of Mexico. 


## Methods
### Data download
For this guide we will use data downloaded from the Global Biodiversity Information Facility ([GBIF](https://www.gbif.org/)). The GBIF database contains a huge collection of information about all kinds of organisms. Our main interest in this database is the information about the distribution of the species.
For this guide we will search and download the information about our species Magnolia iltisiana.

![Main GBIF](https://raw.githubusercontent.com/Zcrass/SpeciesDistributionModels/main/assets/images/01_gbif_main.png "GBIF main site")

We have to select the correct species from the suggested results.

![GBIF search](https://raw.githubusercontent.com/Zcrass/SpeciesDistributionModels/main/assets/images/02_gbif_search.png "GBIF search results")

And here we can see all the data available for this species in the GBIF database. In our case our main interest are the ocurrences which link appear in right above the images.

![GBIF species data](https://raw.githubusercontent.com/Zcrass/SpeciesDistributionModels/main/assets/images/03_gbif_M_iltisiana.png "GBIF species data")

In the occurrences site we found a table with all kind of information about the species. In particular we can find the localities were the species has been found. This data is gatered from different sources. 

![GBIF species occurrences](https://raw.githubusercontent.com/Zcrass/SpeciesDistributionModels/main/assets/images/04_gbif_occurrences.png "GBIF species occurrences")

If we review this data we will find that many occurrences are not georreferencced and other indicate localities very distant from the known distribution of the species. We goin to deal with this later. For the moment we going to download the database using the link at the top of the website. This will generate an unique download link with their corresponding reference. It is recommended to save this information for future reference.

As variables for our model we will use the 19 bioclimatic variables of [world clim](https://www.worldclim.org/data/worldclim21.html). We also going to include the elevation layer also from world clim. All layer were downloaded with a resolution of 30s and saved in a folder called **variables**.

![Worldclim variables](https://github.com/Zcrass/SpeciesDistributionModels/blob/main/assets/images/05_worldclim.png?raw=true "Worldclim variables")

We also downloaded a [shapefile](http://www.conabio.gob.mx/informacion/gis/maps/geo/dest20gw.zip) with the Mexican states to use as mask for other layers and as background in the figures.

### Data preprocesing
We will mainly use the pandas and geopandas library for the data management and matplotlib.pyplot for the figures. For the clustering and predictive models we will use the modules of scikit learn. Aditionally we will use a couple of custom functions from the file sdm_functions.py

```python
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import rasterio
from rasterio.plot import show
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import sys
from yellowbrick.cluster import KElbowVisualizer

from sdm_functions import sdm_functions as fun

```

We start by reading the data downloaded from GBIF. This is done using the pandas library:


```python
sp = pd.read_csv('magnolia_iltisiana.csv', sep='\t')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gbifID</th>
      <th>datasetKey</th>
      <th>occurrenceID</th>
      <th>kingdom</th>
      <th>phylum</th>
      <th>class</th>
      <th>order</th>
      <th>family</th>
      <th>genus</th>
      <th>species</th>
      <th>...</th>
      <th>identifiedBy</th>
      <th>dateIdentified</th>
      <th>license</th>
      <th>rightsHolder</th>
      <th>recordedBy</th>
      <th>typeStatus</th>
      <th>establishmentMeans</th>
      <th>lastInterpreted</th>
      <th>mediaType</th>
      <th>issue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4032005328</td>
      <td>7bd65a7a-f762-11e1-a439-00145eb45e9a</td>
      <td>urn:catalog:MO:Tropicos:103115759</td>
      <td>Plantae</td>
      <td>Tracheophyta</td>
      <td>Magnoliopsida</td>
      <td>Magnoliales</td>
      <td>Magnoliaceae</td>
      <td>Magnolia</td>
      <td>Magnolia iltisiana</td>
      <td>...</td>
      <td>José Antonio Vázquez García</td>
      <td>2016-01-01T00:00:00</td>
      <td>CC_BY_4_0</td>
      <td>Missouri Botanical Garden</td>
      <td>Guillermo Ibarra Manríquez</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2023-02-17T22:10:50.597Z</td>
      <td>NaN</td>
      <td>COORDINATE_ROUNDED;GEODETIC_DATUM_ASSUMED_WGS8...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4031977117</td>
      <td>7bd65a7a-f762-11e1-a439-00145eb45e9a</td>
      <td>urn:catalog:MO:Tropicos:101810366</td>
      <td>Plantae</td>
      <td>Tracheophyta</td>
      <td>Magnoliopsida</td>
      <td>Magnoliales</td>
      <td>Magnoliaceae</td>
      <td>Magnolia</td>
      <td>Magnolia iltisiana</td>
      <td>...</td>
      <td>A. Vazquez</td>
      <td>1989-01-01T00:00:00</td>
      <td>CC_BY_4_0</td>
      <td>Missouri Botanical Garden</td>
      <td>Theodore S. Cochrane;Mark A. Wetter;Francisco ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2023-02-17T22:09:10.816Z</td>
      <td>StillImage</td>
      <td>GEODETIC_DATUM_ASSUMED_WGS84;TYPE_STATUS_INVAL...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4031769395</td>
      <td>7bd65a7a-f762-11e1-a439-00145eb45e9a</td>
      <td>urn:catalog:MO:Tropicos:102798081</td>
      <td>Plantae</td>
      <td>Tracheophyta</td>
      <td>Magnoliopsida</td>
      <td>Magnoliales</td>
      <td>Magnoliaceae</td>
      <td>Magnolia</td>
      <td>Magnolia iltisiana</td>
      <td>...</td>
      <td>J.A. Vazquez-García</td>
      <td>2011-01-01T00:00:00</td>
      <td>CC_BY_4_0</td>
      <td>Missouri Botanical Garden</td>
      <td>Esteban M. Martínez S.;Fred R. Barrie</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2023-02-17T22:09:20.822Z</td>
      <td>NaN</td>
      <td>COORDINATE_ROUNDED;GEODETIC_DATUM_ASSUMED_WGS8...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4031769394</td>
      <td>7bd65a7a-f762-11e1-a439-00145eb45e9a</td>
      <td>urn:catalog:MO:Tropicos:102798078</td>
      <td>Plantae</td>
      <td>Tracheophyta</td>
      <td>Magnoliopsida</td>
      <td>Magnoliales</td>
      <td>Magnoliaceae</td>
      <td>Magnolia</td>
      <td>Magnolia iltisiana</td>
      <td>...</td>
      <td>J.A. Vazquez-García</td>
      <td>2011-01-01T00:00:00</td>
      <td>CC_BY_4_0</td>
      <td>Missouri Botanical Garden</td>
      <td>Esteban M. Martínez S.;T.P. Ramamoorthy</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2023-02-17T22:10:42.613Z</td>
      <td>NaN</td>
      <td>COORDINATE_ROUNDED;GEODETIC_DATUM_ASSUMED_WGS8...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4031691547</td>
      <td>7bd65a7a-f762-11e1-a439-00145eb45e9a</td>
      <td>urn:catalog:MO:Tropicos:102650175</td>
      <td>Plantae</td>
      <td>Tracheophyta</td>
      <td>Magnoliopsida</td>
      <td>Magnoliales</td>
      <td>Magnoliaceae</td>avoid
      <td>Magnolia</td>
      <td>Magnolia iltisiana</td>
      <td>...</td>
      <td>K. Velasco</td>
      <td>2008-01-01T00:00:00</td>
      <td>CC_BY_4_0</td>
      <td>Missouri Botanical Garden</td>
      <td>Arturo Nava Zafra;K. Velasco G.;Nahúm García G.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2023-02-17T22:09:26.660Z</td>
      <td>NaN</td>
      <td>COORDINATE_ROUNDED;GEODETIC_DATUM_ASSUMED_WGS8...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>118</th>
      <td>1261471196</td>
      <td>7bd65a7a-f762-11e1-a439-00145eb45e9a</td>
      <td>urn:catalog:MO:Tropicos:749898</td>
      <td>Plantae</td>
      <td>Tracheophyta</td>
      <td>Magnoliopsida</td>
      <td>Magnoliales</td>
      <td>Magnoliaceae</td>
      <td>Magnolia</td>
      <td>Magnolia iltisiana</td>
      <td>...</td>
      <td>Law</td>
      <td>1995-01-01T00:00:00</td>
      <td>CC_BY_4_0</td>
      <td>Missouri Botanical Garden</td>
      <td>Alwyn H. Gentry;Enirique Jardel</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2023-02-17T22:09:39.012Z</td>
      <td>NaN</td>
      <td>GEODETIC_DATUM_ASSUMED_WGS84;OCCURRENCE_STATUS...</td>
    </tr>
    <tr>
      <th>119</th>
      <td>1259230036</td>
      <td>7bd65a7a-f762-11e1-a439-00145eb45e9a</td>
      <td>urn:catalog:MO:Tropicos:1964362</td>
      <td>Plantae</td>
      <td>Tracheophyta</td>
      <td>Magnoliopsida</td>
      <td>Magnoliales</td>
      <td>Magnoliaceae</td>
      <td>Magnolia</td>
      <td>Magnolia iltisiana</td>
      <td>...</td>
      <td>A. Vázquez-G.</td>
      <td>1989-01-01T00:00:00</td>
      <td>CC_BY_4_0</td>
      <td>Missouri Botanical Garden</td>
      <td>G. Arsène</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2023-02-17T22:08:02.442Z</td>
      <td>NaN</td>
      <td>COORDINATE_ROUNDED;GEODETIC_DATUM_ASSUMED_WGS8...</td>
    </tr>
    <tr>
      <th>120</th>
      <td>1228367888</td>
      <td>90c853e6-56bd-480b-8e8f-6285c3f8d42b</td>
      <td>6bf036de-ad07-4535-a785-d04ffbc9ce90</td>
      <td>Plantae</td>
      <td>Tracheophyta</td>
      <td>Magnoliopsida</td>
      <td>Magnoliales</td>
      <td>Magnoliaceae</td>
      <td>Magnolia</td>
      <td>Magnolia iltisiana</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>CC0_1_0</td>
      <td>The Field Museum of Natural History</td>
      <td>T. S. Cochrane et al.</td>
      <td>ISOTYPE</td>
      <td>NaN</td>
      <td>2023-01-24T22:49:34.313Z</td>
      <td>StillImage</td>
      <td>GEODETIC_DATUM_ASSUMED_WGS84;AMBIGUOUS_COLLECT...</td>
    </tr>
    <tr>
      <th>121</th>
      <td>1056494240</td>
      <td>7e380070-f762-11e1-a439-00145eb45e9a</td>
      <td>59e40859-7b3e-4191-80ad-dc57067f1339</td>
      <td>Plantae</td>
      <td>Tracheophyta</td>
      <td>Magnoliopsida</td>
      <td>Magnoliales</td>
      <td>Magnoliaceae</td>
      <td>Magnolia</td>
      <td>Magnolia iltisiana</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>CC0_1_0</td>
      <td>NaN</td>
      <td>T.S. Cochrane, et al</td>
      <td>ISOTYPE</td>
      <td>NaN</td>
      <td>2023-02-05T21:32:13.740Z</td>
      <td>StillImage</td>
      <td>GEODETIC_DATUM_ASSUMED_WGS84;INSTITUTION_MATCH...</td>
    </tr>
    <tr>
      <th>122</th>
      <td>1056486604</td>
      <td>7e380070-f762-11e1-a439-00145eb45e9a</td>
      <td>819a3c85-cc60-4d45-bb3e-91b3edc32bd6</td>
      <td>Plantae</td>
      <td>Tracheophyta</td>
      <td>Magnoliopsida</td>
      <td>Magnoliales</td>
      <td>Magnoliaceae</td>
      <td>Magnolia</td>
      <td>Magnolia iltisiana</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>CC0_1_0</td>
      <td>NaN</td>
      <td>T.S. Cochrane, et al</td>
      <td>ISOTYPE</td>
      <td>NaN</td>
      <td>2023-02-05T21:31:47.278Z</td>
      <td>StillImage</td>
      <td>GEODETIC_DATUM_ASSUMED_WGS84;INSTITUTION_MATCH...</td>
    </tr>
  </tbody>
</table>
<p>123 rows × 50 columns</p>
</div>

As we can see the original file includes 50 columns and 123 occurrences. For our analysis we only need the georeference data so we only keep the columns of scientificName, decimalLatitude and decimalLongitude and aditionally we discard the localities without coordinates. Optionally we can save this data as a new file.


```python
sp = sp[['scientificName', 'decimalLatitude', 'decimalLongitude']].dropna().drop_duplicates()
sp['presence'] = True
sp = sp.reset_index()
# sp.to_csv('magnolia_iltisiana_locs.csv', index=False) ### optional: saves data to a new csv file
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>scientificName</th>
      <th>decimalLatitude</th>
      <th>decimalLongitude</th>
      <th>presence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Magnolia iltisiana Vazquez</td>
      <td>19.892222</td>
      <td>-104.546389</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Magnolia iltisiana Vazquez</td>
      <td>19.580000</td>
      <td>-104.280000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Magnolia iltisiana Vazquez</td>
      <td>17.390000</td>
      <td>-100.197778</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Magnolia iltisiana Vazquez</td>
      <td>17.395000</td>
      <td>-100.198889</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Magnolia iltisiana Vazquez</td>
      <td>16.677000</td>
      <td>-97.794583</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>60</th>
      <td>117</td>
      <td>Magnolia iltisiana Vazquez</td>
      <td>19.730000</td>
      <td>-104.250000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>61</th>
      <td>118</td>
      <td>Magnolia iltisiana Vazquez</td>
      <td>19.410000</td>
      <td>-104.100000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>62</th>
      <td>119</td>
      <td>Magnolia iltisiana Vazquez</td>
      <td>19.650833</td>
      <td>-101.115556</td>
      <td>True</td>
    </tr>
    <tr>
      <th>63</th>
      <td>120</td>
      <td>Magnolia iltisiana Vazquez</td>
      <td>23.000000</td>
      <td>-102.000000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>64</th>
      <td>121</td>
      <td>Magnolia iltisiana Vazquez</td>
      <td>23.950465</td>
      <td>-102.532886</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>65 rows × 5 columns</p>
</div>

Now we have a dataframe with only 65 ocurrences and three columns. Next we will convert this pandas dataframe into a geopandas dataframe. This includes a geometry column that store the geographycal information. Then we will extract the extent of the points to define the study area. We use a distance of 1 degree around the points to define a margin. Also used the Mexican states to clip the region and remove the area corresponding to the oceans. 


```python
### turn pandas df into a geopandas df
sp = gpd.GeoDataFrame(sp, crs='epsg:4326', geometry=gpd.points_from_xy(sp.decimalLongitude, sp.decimalLatitude))    

### define area of interest
dest = gpd.read_file("shapefiles/dest20gw.zip")
extent = fun.extent_poly(df=sp, margin_size=1, crs="epsg:4326")
extent = gpd.overlay(extent, dest, how="intersection").dissolve()
```

Now we can visualize our localities in a map. 

```python
fig, ax = plt.subplots()
ax.set_aspect("equal")
plt.xlim([extent.bounds['minx'][0], extent.bounds['maxx'][0]])
plt.ylim([extent.bounds['miny'][0], extent.bounds['maxy'][0]])
extent.plot(ax=ax, color='lightgrey')
dest.plot(ax=ax, color='none', edgecolor='darkgrey')
sp.plot(ax = ax, color="green", edgecolor = "black") 
plt.show()
```


    
![Original occurrences](https://raw.githubusercontent.com/Zcrass/SpeciesDistributionModels/main/assets/images/README_11_0.png)
    


### Data reduction
As we can see in the map most of the localities apear closed together but some others are far from the rest. Has been reconigzed that GBIF data often include misidentifications or wrong cordinates. We cannot be sure if those remote localities are indeed far dispersed individuals or were wronly identified as *M. schiedeana*. If we include wrongly identified individuals in our analysis the results will be most likely wrong so we need to check this localities carefully. By the other hand, we have 20 variables available, use all these in our model could cause a bias in the results, due the high correlation that could exist within these. All this makes it necessary to reduce our data both discarding the occurrences that does not correspond to our species of interest and reducint the variables to those that show no correlation.

To discard wrong occurrences we goin to implement a clustering method. This will cluster together those localities with similar bioclimatic conditions and then we have to select the cluster that best represent the known distribution of the species. For this the first step is to extract the bioclimatic variables of each locality.


```python
##### list rasters layers
variables = fun.list_files('variables/')
var_names = list(variables.keys())

### load each raster and extract values by each point
coords = [(x,y) for x, y in zip(sp.decimalLongitude, sp.decimalLatitude)]
for var in var_names:
    raster = rasterio.open(variables[var])
    sp[var] = [x[0] for x in raster.sample(coords)]

# sp.to_csv('magnolia_iltisiana_vars.csv', index=False) ### optional: saves data to a new csv file
```


Now we will discard variables if those are correlated. For this we perform a correlation test and we going to keep only the variables with a correlation bellow 0.75. This leave us with only 6 bioclimatic variables.


```python

var_names = list(set(var_names) - set(fun.cor_test(df=sp[var_names], thr=0.75)))
var_names
```




    ['wc2.1_30s_bio_1.tif',
     'wc2.1_30s_bio_15.tif',
     'wc2.1_30s_bio_12.tif',
     'wc2.1_30s_bio_2.tif',
     'wc2.1_30s_bio_18.tif',
     'wc2.1_30s_bio_14.tif']



As we dont know if all the localities correspond only to *M. iltisiana* or includes more than one miss identified species we need to know how many groups we have in the data. For this we going to use the elbow method to identify the most likely number of clusters:



```python
visualizer = KElbowVisualizer(KMeans(), k=(1,10), timings= True)
visualizer.fit(sp[var_names])        
visualizer.show()   
```

    
![Elbow graph](https://raw.githubusercontent.com/Zcrass/SpeciesDistributionModels/main/assets/images/README_18_1.png)

For other datasets we recomend to thest other models (such agglomerative clustering methods) to enssure a correct division of the data that is coherent to the knowledge of the distribution of the species. Also, althoug the Elbow method can be useful to determine the number of clusters to use, we suggest to take the final decition based on the knowleg of the organism of interest.

After some test we found that for this dataset the best algorithm is the k-means clustering method and, as suggested by the elbow method, the number of cluster found in this data is four. We going to split our datapoints using this method in fourt different groups and visualize all in a map:


```python
# model = AgglomerativeClustering(n_clusters=4, metric='manhattan', linkage='average') 
model = KMeans(n_clusters=4, init='random', n_init=10, max_iter=300, tol=1e-04) ### sp['cluster'] = model.fit_predict(sp[var_names])
# sp.to_csv('magnolia_iltisiana_clusters.csv', index=False) ### optionally save to a csv file
```
```python
fig, ax = plt.subplots()
plt.xlim([extent.bounds['minx'][0], extent.bounds['maxx'][0]])
plt.ylim([extent.bounds['miny'][0], extent.bounds['maxy'][0]])
ax.set_aspect("equal")
dest.plot(ax=ax, color='lightgrey', edgecolor='darkgrey')
scatter = ax.scatter(sp.decimalLongitude, sp.decimalLatitude, c=sp.cluster)
ax.legend(*scatter.legend_elements())
plt.show()
```


    
![Data clustering](https://raw.githubusercontent.com/Zcrass/SpeciesDistributionModels/main/assets/images/README_20_0.png)
    


Based on our previous knowledge of the species, we see that cluster 0 is the most similar to the known distribution of *M. iltisiana*. So we will use this in the following analysis. It is important to note that the cluster selected could vary depending on the clustering method and the dataset. The decision of wich cluster to use should be based on an important revision of the existing information of the species.


```python
sp = sp.loc[sp.cluster == 0]
sp = sp.reset_index()

```

### Generating pseudo-absences data

Most machine learning models used for SDM required absence data to train and test the model. However this absences data is often difficult to obtain so we need to generate pseudo-absences. We do this based on the study area and a buffer around the precense localities. Finally we concatenate all the prescence and absences in one dataframe and visualize it in a map.



```python
### create buffer and pseudo absences
buf = sp.geometry.buffer(0.2)
buf = gpd.GeoDataFrame(crs='epsg:4326', geometry=buf.to_crs('epsg:4326'))
absences_points = fun.Random_Points_in_polygon(gpd.overlay(extent, buf, how="difference"), 1000, 'epsg:4326')

### concat all points
sp = pd.concat([sp[["presence", "decimalLongitude", "decimalLatitude", "geometry"]],
                absences_points[["presence", "decimalLongitude", "decimalLatitude", "geometry"]]])
        
```

```python
fig, ax = plt.subplots()
plt.xlim([extent.bounds['minx'][0], extent.bounds['maxx'][0]])
plt.ylim([extent.bounds['miny'][0], extent.bounds['maxy'][0]])
ax.set_aspect("equal")
dest.plot(ax=ax, color='lightgrey', edgecolor='darkgrey')
scatter = ax.scatter(sp.decimalLongitude, sp.decimalLatitude, c=sp.presence)

ax.legend(*scatter.legend_elements())
plt.show()
```


    
![Presence and pseudo-absences](https://raw.githubusercontent.com/Zcrass/SpeciesDistributionModels/main/assets/images/README_25_0.png)
    


### Training and testing model

The next step is train and test the model. For this we need to extract the biclimatic data for each point. Aditionally we extract data from the complete rasters to use later when we extrapolate our model to all the area of interest.


```python
### load each raster and extract values by each point and from all the layer
raster_df = pd.DataFrame() 

for var in var_names:
    masked = fun.mask_raster(variables[var], extent, 'epsg:4326', "tmp.tif")
    sp[var] = [x[0] for x in masked.sample([(x,y) for x, y in zip(sp.decimalLongitude, sp.decimalLatitude)])]
    raster_df[var] = pd.DataFrame(np.array(masked.read()).reshape([1,-1]).T)

```

Now we will split our dataset in the train and test subsets and then we train the model.


```python
### split dataset into train and test
split_points = random.sample(range(len(sp.index)), round(len(sp.index)*0.25))

### train model
model = RandomForestRegressor(n_estimators = 1000, criterion = "absolute_error",
                                max_depth = None, oob_score = False, 
                                n_jobs = -1, bootstrap=True, random_state = 123)
model.fit(sp.iloc[split_points, :].loc[:, var_names], 
                sp.iloc[split_points, :].loc[:, "presence"].to_list())
```

Next we use our test dataset to test the resulting model. We going to calculate the root-mean-square error (RMSE) to estimate the error of our model...


```python
### predict test and compute rmse
test_prediction = model.predict(X = sp.loc[~sp.index.isin(split_points), var_names] )
rmse = mean_squared_error(y_true = sp.loc[~sp.index.isin(split_points), "presence"].to_list(),
                                y_pred = test_prediction, squared = False)
rmse
```




    0.10348157848666938



### Extrapolating results
Finally we extrapolate our model to the complete study area and visualize our results. 


```python
### extrapolate prediction to complete rasters
raster_df.columns = var_names
raster_prediction = model.predict(X = raster_df)

### proyect results
results_raster, results_transform = fun.resulting_raster("tmp.tif", raster_prediction)


```

```python
##### FIGURE
fig, ax = plt.subplots()
plt.xlim([extent.bounds['minx'][0], extent.bounds['maxx'][0]])
plt.ylim([extent.bounds['miny'][0], extent.bounds['maxy'][0]])
ax.set_aspect("equal")
show(results_raster, ax = ax, transform=results_transform, cmap="gray") 
dest.plot(ax=ax, color='none', edgecolor='darkgrey')
sp[sp.presence == True].plot(ax = ax, color="green", edgecolor = "black") 
# sp.plot(ax = ax, color="green", edgecolor = "black") 
# absences_points.plot(ax = ax, color="red", edgecolor = "black") 
plt.show()
        
```


    
![png](https://raw.githubusercontent.com/Zcrass/SpeciesDistributionModels/main/assets/images/README_37_0.png)
    
Finally we can save the resulting raster in a geotiff file.


```python
### Save raster 


```python
with rasterio.open('test1.tif', 'w', driver='GTiff',
                            height = results_raster.shape[0], width = results_raster.shape[1],
                            count=1, dtype=str(results_raster.dtype),
                            crs='epsg:4326',
                            transform=results_transform) as rast:
        rast.write(results_raster, 1)
```




## References

GBIF.org (09 March 2023) GBIF Occurrence Download https://doi.org/10.15468/dl.aarrqr

