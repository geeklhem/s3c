import pandas as pd 
import numpy as np
from s3c.columns import col_names_checker

def contrib(census_i, census_f, species,col_names=None):
    col_names = col_names_checker(col_names,[census_f.columns,census_i.columns,species])
    
    # Group observations by species and merge them. 
    census_i = census_i.groupby(col_names["species"]).sum()[col_names["n"]].reset_index()
    census_f = census_f.groupby(col_names["species"]).sum()[col_names["n"]].reset_index()
    census = pd.merge(census_i,census_f,
                      left_on=col_names["species"],right_on=col_names["species"],
                      suffixes=("_i","_f"),
                      how="outer").fillna(0)
    

    # Filter species list and merge them.
  
    species = species.loc[:,(col_names["species"],
                             col_names["trait_val"],
                             col_names["trait_var"])]
    
  

    census = pd.merge(census,species,
                      left_on=col_names["species"],right_on=col_names["species"],
                      how="left")
  

    # Originality is the diffenrence to the mean trait value.
    mean = census[col_names["trait_val"]].mean()
    census["originality"] = census[col_names["trait_val"]] - mean
    census.drop([col_names["trait_val"],col_names["trait_var"]],1,inplace=True)
    
    # Compute differences in relative abundances.
    census[col_names["n"]+"_i"] /= census[col_names["n"]+"_i"].sum()
    census[col_names["n"]+"_f"] /= census[col_names["n"]+"_f"].sum()
    census["dp"] =  census[col_names["n"]+"_f"] - census[col_names["n"]+"_i"]
    census.drop([col_names["n"]+"_i",col_names["n"]+"_f"], 1, inplace=True)

    # Contribution is the product originalitt * dp.
    census["contrib"] = census["originality"] * census["dp"]

    return census