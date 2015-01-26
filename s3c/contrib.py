import pandas as pd 
import numpy as np
import s3c.index 
from s3c.columns import col_names_checker

def contrib(census_i, census_f, species, col_names=None):
    """Compute specific contribution to community weighted indexes
    variations between two timepoint
    
    Args:
        census_i (pandas.DataFrame): Initial census dataframe. Required columns are
            "n" (number of individual), "species".
        census_f (pandas.DataFrame): Final census dataframe. Required columns are
            "n" (number of individual), "species".
        traits (pandas.DataFrame): trait value dataframe. Required columns are
            "species", "trait_val" (trait value) and optionnally "trait_var" (trait 
            intraspecific variance).
        col_names (dict): Columns names correspondance if they are not default.

    Returns:
        A pandas.Dataframe with the specific contribution and its decomposition.

    """
    # Check columns name sanity.
    col_names = col_names_checker(col_names,[census_f.columns,
                                             census_i.columns,
                                             species.columns])

    # If no known intraspecific variations, fix them to 0
    if col_names["trait_var"] not in species.columns:
        species[col_names["trait_var"]] = 0

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
    
    # Variance originality
    squared_mean = (census[col_names["trait_val"]]**2).mean()
    census["v_originality"] = census[col_names["trait_val"]]**2 - squared_mean

    # Sum of initial and final CWM
    cnames_i = col_names.copy()
    cnames_i["n"] = col_names["n"] + "_i"
    cnames_f = col_names.copy()
    cnames_f["n"] = col_names["n"] + "_f"
    S = s3c.index.mean(census,cnames_f) + s3c.index.mean(census,cnames_i)

    # Drop unwanted columns
    census.drop([col_names["trait_val"],col_names["trait_var"]],1,inplace=True)
    
    # Compute differences in relative abundances.
    census[col_names["n"]+"_i"] /= census[col_names["n"]+"_i"].sum()
    census[col_names["n"]+"_f"] /= census[col_names["n"]+"_f"].sum()
    census["dp"] =  census[col_names["n"]+"_f"] - census[col_names["n"]+"_i"]
    census.drop([col_names["n"]+"_i",col_names["n"]+"_f"], 1, inplace=True)

    # Contribution is the product originalitt * dp.
    census["contrib"] = census["originality"] * census["dp"]
    census["v_contrib"] = census["dp"] * (census["v_originality"] - census["originality"] * S)  
        
    return census

def trend_contrib(census, species, col_names = None):
    """Compute specific contribution to community weighted indexes variations through 
    time using linear tendencies. 
    
    Args:
        census_i (pandas.DataFrame): Initial census dataframe. Required columns are
            "n" (number of individuals), "species" and "date".
        traits (pandas.DataFrame): trait value dataframe. Required columns are
            "species", "trait_val" (trait value) and optionnaly "trait_var" (trait 
            intraspecific variance).
        col_names (dict): Columns names correspondance if they are not default.

    Returns:
        A pandas.Dataframe with the specific contribution and its decomposition. 
    """

    # Check columns name sanity.
    col_names = col_names_checker(col_names,[census.columns,species.columns])

    # Filter to keep only present species.
    present_sp = census[col_names["species"]].drop_duplicates()
    species = species.loc[present_sp,:]
    
    # If no known intraspecific variations, fix them to 0
    if col_names["trait_var"] not in species.columns:
        species[col_names["trait_var"]] = 0

    # Compute originality.
    species = species.set_index(col_names["species"])
    mean = species.loc[:,col_names["trait_val"]].mean()
    squared_mean = (species.loc[:,col_names["trait_val"]]**2
                    + species.loc[:,col_names["trait_var"]]).mean()

    species["originality"] = species[col_names["trait_val"]] - mean
    species["v_originality"] = (species[col_names["trait_val"]]**2
                                + species[col_names["trait_var"]]
                                - squared_mean

    # Compute relative abundance trends.
    dp = {}
    census = census.groupby([col_names["date"],
                             col_names["species"]]).sum()[col_names["n"]].reset_index()
    n_by_date = census.groupby(col_names["date"]).sum()[col_names["n"]]
    n_by_date = n_by_date.reset_index().rename({col_names["n"]:"total"})
    n_by_date.columns = [col_names["date"],"total"]
   
   
    census = pd.merge(census,
                      n_by_date,
                      how="left")
   
    census[col_names["n"]] /= census["total"]

    for spe,df in census.groupby(col_names["species"]):
        #Linear regression y = Ap, where A = [[x 1]]
        A = np.vstack([df[col_names["date"]].values,
                       np.ones(len(df[col_names["date"]].values))]).T
        dp[spe] = np.linalg.lstsq(A,
                                  df[col_names["n"]].values)[0][0]
    species["dp"] = pd.Series(dp)

    # Compute contributions 
    species["contrib"] = species["originality"] * species["dp"]
    species["v_contrib"] = species["dp"] * (species["v_originality"] - census["originality"] * S)

    # Sort output
    species = species.sort("contrib", ascending=False)
    return species.reset_index()
