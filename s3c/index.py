""" index.py - Functions to compute community weighted indexes """
import numpy as np
import pandas as pd
from s3c.columns import col_names_checker

def mean(df,col_names):
    """Compute the community weighted mean of the dataframe.

    Args:
        df (pandas.DataFrame): Census dataframe.
        col_names (dict): Columns names. Required names are "n" (number
            of individuals) and "trait_val" (trait value).

    Return: 
        Community weighted mean (float).
    """ 
    return np.dot(df[col_names["trait_val"]],
                  df[col_names["n"]]/float(df[col_names["n"]].sum()))

def var(df,col_names):
    """ Compute the community weighted variance of the dataframe.

    Args:
        df (pandas.DataFrame): Census dataframe.
        col_names (dict): Columns names. Required names are "n" (number
            of individuals) and "trait_val" (trait value).

    Return: 
        Community weighted variance (float).""" 
    av = mean(df, col_names)
    corrective_term = df[col_names["n"]].sum()/(df[col_names["n"]].sum()-1)
    n2 = np.dot(df[col_names["trait_val"]]**2,
                df[col_names["n"]]/float(df[col_names["n"]].sum()))    
    return corrective_term * (n2 - av**2) 


def bootstrap_cwi(df,col_names,k,bootstrap_ci):
    """ Bootstrap estimator of CWM/CWV.

    Bootstrap is performed /at individual level/. 

    Args:
        df (pandas.DataFrame): Census dataframe.
        col_names (dict): Columns names. Required names are "n" (number
            of individuals) and "trait_val" (trait value).
        k (int): Number of bootstrap re-sampling.
        bootstrap_ci (float): Percentile of bootstrap confidence interval.

    Returns:
        A dict. with the bootstrap estimators of CWM/CWV and the 
        corresponding confidence interval.
    """
    out = {}
    
    # Get relative abundances: this is the distribution from
    # which the bootstrap sample will be drawn.
    N = df[col_names["n"]].sum() # number of individuals. 
    df[col_names["n"]] /= df[col_names["n"]].sum()
    
    # Sort by relative abundances to speed up computations.
    df.sort(col_names["n"], ascending=False, inplace=True)
    df.reset_index(inplace=True)
    
    # Do the cummulative sum of all relative abundances to
    # divide the segment [0,1] between all species. 
    df["proba"] =  df[col_names["n"]].cumsum()
    sp = len(df.proba) # number of species.

    # Perform k bootstrap.
    cwm = np.zeros(k)
    cwv = np.zeros(k)
    for i in range(k):
        # Draw a new community composition from the distribution.
        hist= np.digitize(np.random.rand(N),
                           bins=df.proba.values)
        df[col_names["n"]] = np.bincount(hist,minlength=sp)

        # Compute the indicies. 
        cwm[i] = mean(df, col_names)
        cwv[i] = var(df, col_names)

    # Bootstrap estimators are derived from the bootstrap distribution. 
    out["bootstrap_cwm"] = np.mean(cwm)
    out["bootstrap_cwv"] = np.mean(cwv)
    out["bootstrap_cwm_lower_ci"] = np.percentile(cwm,1-bootstrap_ci)
    out["bootstrap_cwv_lower_ci"] = np.percentile(cwv,1-bootstrap_ci)
    out["bootstrap_cwm_higher_ci"] = np.percentile(cwm,bootstrap_ci)
    out["bootstrap_cwv_higher_ci"] = np.percentile(cwv,bootstrap_ci)

    return out

def cwi(census, traits, col_names=None,
        bootstrap = False, bootstrap_n = 100, bootstrap_ci=.95):
    """ Compute community weighted indexes of a community.

    Bootstrap is performed /at individual level/. 

    Args:
        census (pandas.DataFrame): Census dataframe. Required columns are
            "n" (number of individual), "species".
        traits (pandas.DataFrame): trait value dataframe. Required columns are
            "species", "trait_val" (trait value) and "trait_var" (trait 
            intraspecific variance).
        col_names (dict): Columns names correspondance if they are not default.
        bootstrap (bool): Perform bootstrap if true.
        bootstrap_n (int): Number of bootstrap re-sampling.
        bootstrap_ci (float): Percentile of bootstrap confidence interval.

    Returns:
        A dict. with the bootstrap estimators of CWM/CWV and the 
        corresponding confidence interval.
    """
    # Check columns name sanity.
    col_names = col_names_checker(col_names,[census.columns,traits.columns])            

    # Group all individuals of a given species.
    census = census.groupby(col_names["species"]).sum()[col_names["n"]]
    census = census.reset_index()

    # Merge census with species traits. 
    merged = pd.merge(census.loc[:,[col_names["species"],
                                    col_names["n"]]],
                      traits.loc[:,[col_names["species"],
                                    col_names["trait_val"],
                                    col_names["trait_var"]]])
                      
    # Compute indexes.
    out = {}
    out["cwm"] = mean(merged, col_names)
    out["cwv"] = var(merged, col_names)

    # Perform bootstrap if needed.
    if bootstrap:
        out_boot = bootstrap_cwi(merged,col_names,bootstrap_n,bootstrap_ci)
        out.update(out_boot)
        
    return out


def cwi_stratified(census,traits,col_names=None,
                   bootstrap = False, bootstrap_n = 100, bootstrap_ci=.95):
    """ Stratified computation of community weighted indexes of a community.

    Observations will be grouped by site and date.
    Bootstrap is performed /at individual level/. 

    Args:
        census (pandas.DataFrame): Census dataframe. Required columns are
            "n" (number of individual), "species". Optional columns are "date"
            and "site".
        traits (pandas.DataFrame): trait value dataframe. Required columns are
            "species", "trait_val" (trait value) and "trait_var" (trait 
            intraspecific variance).
        col_names (dict): Columns names correspondance if they are not default.
        bootstrap (bool): Perform bootstrap if true.
        bootstrap_n (int): Number of bootstrap re-sampling.
        bootstrap_ci (float): Percentile of bootstrap confidence interval.

    Returns:
        A dict. with the bootstrap estimators of CWM/CWV and the 
        corresponding confidence interval.
    """

    # Check columns name sanity.
    col_names = col_names_checker(col_names,[census.columns,traits.columns])
    drop_date = False
    drop_site = False 
    if col_names["date"] not in census.columns:
        census[col_names["date"]] == None
        drop_date = True
    if col_names["site"] not in census.columns:
        census[col_names["site"]] == None
        drop_site = True 

    # Prepare output. 
    out = []
    
    # Group all individuals of a given species.
    merged = pd.merge(census.loc[:,[col_names["species"],
                                    col_names["n"],
                                    col_names["site"],
                                    col_names["date"]]],
                      traits.loc[:,[col_names["species"],
                                    col_names["trait_val"],
                                    col_names["trait_var"]]])

    # Compute indexes. 
    for site,df in merged.groupby(col_names["site"]):
        for date,ddf in df.groupby(col_names["date"]):
            out.append({})
            
            # Compute indices
            out[-1]["cwm"] = mean(ddf, col_names)
            out[-1]["cwv"] = var(ddf, col_names)

            # Bootstrap if needed
            if bootstrap:
                out_boot = bootstrap_cwi(merged,
                                         col_names,
                                         bootstrap_n,
                                         bootstrap_ci)
                out[-1].update(out_boot)

            # Add misc. site informations. 
            out[-1][col_names["date"]] = date
            out[-1][col_names["site"]] = site
            out[-1][col_names["n"]] = ddf[col_names["n"]].sum()

    out = pd.DataFrame(out)
    # Drop unused columns
    if drop_date:
        out.drop(col_names["date"], 1, inplace=True)
    if drop_site:
        out.drop(col_names["site"], 1, inplace=True)
    return out
