# Simple Species Distribution Model in Python
## Introduction
Species distribution model are...

### Species Distribution Models 
Species Distribution Models (SDM)
We are going to produce an species distribution model using python and GBIF data.

### Species of interest
We selected a Magnolia species called Magnolia iltisiana. This is a Tree that inhabits in the western part of Mexico. 


## Methods
### Data download
For this guide we will use data downloaded from the Global Biodiversity Information Facility ([GBIF](https://www.gbif.org/)). the GBIF database contains a huge collection of information about all kinds of organisms. Our main interest in this database is the information about the distribution of the species.
For this guide we will search and download the information about our species Magnolia iltisiana.

![Main GBIF](/assets/images/assets/images/01_gbif_main.png "GBIF main site")

We have to select the correct species from the suggested results.

![GBIF search](/assets/images/assets/images/02_gbif_search.png "GBIF search results")

Here we can see all the data available for this species in the GBIF database. In our case our main interest are the ocurrences link above the images.

![GBIF species data](/assets/images/assets/images/03_gbif_M_iltisiana.png "GBIF species data")

In the occurrences site we found a table with all kind of information about the species. In particular we can find the localities were the species has been found. This data is gatered from different sources. We can review this in the Basis of record and Dataset columns.

![GBIF species occurrences](/assets/images/assets/images/04_gbif_occurrences.png "GBIF species occurrences")

Once we reviewed the data of our species we can download the database using the link at the top of the website. This will generate an unique download link with their corresponding reference. It is recommended to save this information for future reference.

As variables for our model we will use the 19 bioclimatic variables of [world clim](https://www.worldclim.org/data/worldclim21.html). We also going to include the elevation layer also from world clim. All layer were downloaded with a resolution of 30s and saved in a folder called *variables*.

![Worldclim variables](/assets/images/assets/images/05_worldclim.png "Worldclim variables")

#### Data preprocesing
We will mainly use the pandas and geopandas library for the data management and matplotlib.pyplot for the figures. For the clustering and predictive models we will use the modules of scikit learn. Aditionally we will use a couple of custom functions from the file sdm_functions.py



```python
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import rasterio
from rasterio.plot import show
from shapely.geometry import Polygon
from sklearn import decomposition
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import sys

from sdm_functions import sdm_functions as fun

```

We start by reading the data downloaded from GBIF:


```python
sp = pd.read_csv('magnolia_iltisiana.txt', sep='\t')
sp
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
      <td>Magnoliaceae</td>
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



As we can see the original file includes several columns. For our case we only need the georeference data. In this case, not all datapoints include coordinates so we will discard the localities without this information. Optionally we can save this data as a new file for future references.


```python
sp = pd.read_csv('magnolia_iltisiana.txt', sep='\t', usecols=['scientificName', 'decimalLatitude', 'decimalLongitude']).dropna().drop_duplicates()
sp['presence'] = True
sp = sp.reset_index()
# sp.to_csv('magnolia_iltisiana_vars.csv', index=False) ### optional: saves data to a new csv file
sp
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



Now we have a dataframe with only 64 datapoints. We will convert dataframe into a geopandas dataframe and then we will extract the extent of the points to define the study area. We use a distance of 1 degree around the points to define a margin.
 visualize it in a map as follows:


```python
sp = gpd.GeoDataFrame(sp, crs="epsg:4326", geometry=gpd.points_from_xy(sp.decimalLongitude, sp.decimalLatitude))    
extent = fun.extent_poly(df=sp, margin_size=1, crs="epsg:4326")

```

Now we can visualize our localities in a map. We are importing the Mexican states as background of our map.


```python
dest = gpd.read_file("shapefiles/dest20gw.zip")
fig, ax = plt.subplots()
ax.set_aspect("equal")
plt.xlim([extent.bounds['minx'][0], extent.bounds['maxx'][0]])
plt.ylim([extent.bounds['miny'][0], extent.bounds['maxy'][0]])

dest.plot(ax=ax, color='lightgrey', edgecolor='darkgrey')
sp.plot(ax = ax, color="green", edgecolor = "black") 
plt.show()
```


    
![png](README_files/README_11_0.png)
    

