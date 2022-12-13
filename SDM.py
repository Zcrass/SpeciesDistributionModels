import pandas as pd

locs = pd.read_csv("localities.csv")

sp1 = locs.loc[locs["species"]== "species_1", :]
coords = [(x,y) for x, y in zip(sp1.longitude, sp1.latitude)]

print(coords)
