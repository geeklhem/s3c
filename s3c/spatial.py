""" spatial.py - Functions to average observations based on distance"""
import pandas as pd
import numpy as np
from itertools import combinations
from s3c.columns import col_names_checker

def distances(sites,col_names=None):
    """ Compute a distance dictionnary 

    Args:
        sites (pandas.DataFrame): Required columns are "site" (site label),
        "x", "y" (position) 
        col_names (dict): Columns names correspondance if they are not default.

    Returns: 
        A dict. with each pair of site as a key and distance as value.
    """

    distance = {}
        
    # Check columns name sanity and drop unwanted columns.
    col_names = col_names_checker(col_names,[sites.columns])            
    sites = sites.set_index(col_names["site"])
    sites = sites.loc[:,[col_names["x"],col_names["y"]]]
    
    #Distance between two sites
    for a,b in combinations(sites.index,2):
        d = np.linalg.norm(sites.loc[a,:].values-sites.loc[b,:].values)
        distance[(a,b)] = d
        distance[(b,a)] = d

    # Distance to itself is null.
    for a in sites.index:
        distance[(a,a)] = 0
    return distance 


def kde(census,distance,kernel,
        col_names=None,threshold=10,dampening=3):
    """ Enrich observations on each site by using nearby sites.
    
    The way the information of other sites is used depends on the 
    kernel function used. 

    Args:
        census (pandas.DataFrame): Census dataframe. Required columns are
            "n" (number of individual), "species" and "site" (site label).
        distance (dict): Distance for each pair of site label.
        kernel (str or function): kernel function or keyword.
        threshold (float): maximum distance, used in the "threshold" kernel.
        dampening (float): used in the "gaussian" and "linear" kernels. 
        col_names (dict): Columns names correspondance if they are not default.  
    Return:
        A pandas.DataFrame akin to `census`.

    Predefined kernel functions:

    "threshold": w = 0 if d>threshold, w=1 if d>= threshold.
    "linear": w = 1-dampening*d.
    "gaussian": w = 1/(dampening * sqrt(2*np.pi)) * exp(-d**2/2*dampening**2).    
    """
    
    # Check columns name sanity.
    col_names = col_names_checker(col_names,[census.columns])            

    # Check kernel input
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
    
    # For each site...
    for local_site in census[col_names["site"]].drop_duplicates():
        df = census.copy()

        # Compute the weight of each site according to its distance to
        # the focal site.
        df["multiplier"] = [kernel(distance[(x,local_site)])
                            for x in df[col_names["site"]]]

        # Weigth sites census. 
        df["n"] *= df["multiplier"]

        # Remove unwanted columns and lines
        df = df.loc[df[col_names["n"]]>0,
                    [col_names["species"],col_names["n"]] ]

        # Sum species. 
        df = df.groupby(col_names["species"]).sum().reset_index()
        # Add site name.
        df[col_names["site"]] = local_site
        local_df.append(df)
        
    return pd.concat(local_df)
