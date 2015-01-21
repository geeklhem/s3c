import pandas as pd
import numpy as np
from itertools import combinations


def distances(sites):
    distance = {}
    for a,b in combinations(sites.index,2):
        d = np.linalg.norm(sites.loc[a,:].values-sites.loc[b,:].values)
        distance[(a,b)] = d
        distance[(b,a)] = d
    for a in sites.index:
        distance[(a,a)] = 0
    return distance 


def kde(census,distance,kernel,threshold=10,dampening=3):
    
    if kernel == "threshold":
        kernel = lambda d: d<=threshold
    elif kernel == "linear":
        kernel = lambda d: 1-dampening*d
    elif kernel == "gaussian":
        kernel = lambda d: 1/(dampening * np.sqrt(2*np.pi)) * np.exp(-d**2/2*dampening**2)
    else:
        try:
            kernel(0)
        except Exception:
            raise ValueError("Kernel must be a keyword or a valid univariate function")
    local_df = []
    for local_site in census.site.drop_duplicates():
        df = census.copy()
        df["multiplier"] = [kernel(distance[(x,local_site)]) for x in df.site]

        df["n"] *= df["multiplier"]
        df = df.loc[df["n"]>0,["species","n"]]

        df = df.groupby("species").sum()
        df.reset_index(inplace=True)
        df["site"] = local_site
        local_df.append(df)
    return pd.concat(local_df)
