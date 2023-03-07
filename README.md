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

As variables for our model we will use the 19 bioclimatic variables of [world clim](https://www.worldclim.org/data/worldclim21.html). We also going to include the elevation layer also from world clim. All layer were downloaded with a resolution of 30s.

![Worldclim variables](/assets/images/assets/images/05_worldclim.png "Worldclim variables")

### Data cleaning
Altough GBIF it is an important tool for research and does a great job in gattering info from all kind of sources, it is important to know that in many case this information could have some degree of [error](https://doi.org/10.1016/j.ecoinf.2013.11.002). Because of this, data from GBIF usually requieres a cleaning process wich could include both [automatic](https://peerj.com/articles/9916/#material-and-methods) and manual processes. For the manual processes it is important to consider the biology of the species to take decisions of the records to keep.

In our case we going to implement a semiautomatic method that takes into account the bioclimatic caracteristics of the species and performs a clustering analisis to group the occurrences. Then we use this grouping and the previous knowledge of the species to take a decission in the records to keep.

#### Data preprocesing
We will mainly use the pandas and geopandas library for the data management and matplotlib.pyplot for the figures. For the clustering and predictive models we will use the modules of scikit learn. Aditionally we will use a couple of custom functions from the file sdm_functions.py

```python
import geopandas as gpd
import logging as lg
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
The first step of our analysis is to extract the coordinates from the downloades GBIF file. For this we will use pandas to read and extract the columns scientificName, decimalLatitude and decimalLongitude. Aditionally we will add a new column call presence. This column will be used later to diferenciate the presence localities from the pseudoabsences. 

```python
sp = pd.read_csv(args.gbif_file, sep='\t', usecols=['scientificName', 'decimalLatitude', 'decimalLongitude']).dropna().drop_duplicates()
sp['presence'] = True
# sp.to_csv('locs_file.csv', index=False) ### optionally this data can be saved into a csv file
print(sp)
```







Keep only coord columns and remove NAs
If we filter the 

We goin to use the environment data to filter localities that differs from the majority of the points.





## References
GBIF.org (18 February 2023) GBIF Occurrence Download  https://doi.org/10.15468/dl.5vcrpr