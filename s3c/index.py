import numpy as np
import pandas as pd
from s3c.columns import col_names_checker

def mean(df,col_names):
    return np.dot(df[col_names["trait_val"]],
                  df[col_names["n"]]/float(df[col_names["n"]].sum()))

def var(df,col_names):
    av = mean(df,col_names)
    correct = (df[col_names["n"]]/float(df[col_names["n"]].sum())).sum()
    diff = np.array([p*(i-av)**2 for i,p in zip(df[col_names["trait_val"]],df[col_names["n"]])])
    return diff.sum()/correct


def bootstrap_cwi(df,col_names,k,bootstrap_ci):
    out = {}
    cwm = np.zeros(k)
    cwv = np.zeros(k)
    N = df[col_names["n"]].sum()
    # Get relative abundances
    df[col_names["n"]] /= df[col_names["n"]].sum()
    
    # Sort by relative abundances to speed up computations.
    df.sort(col_names["n"], ascending=False, inplace=True)
    df.reset_index(inplace=True)
    
    df["proba"] =  df[col_names["n"]].cumsum()
    sp = len(df.proba)
    #print df.head()
    #print df.tail()
    #print df.proba.values
    for i in range(k):
        #print "bootstrap {}".format(i)
        hist= np.digitize(np.random.rand(N),
                           bins=df.proba.values)
        df[col_names["n"]] = np.bincount(hist,minlength=sp)
        cwm[i] = mean(df, col_names)
        cwv[i] = var(df, col_names)

    out["bootstrap_cwm"] = np.mean(cwm)
    out["bootstrap_cwv"] = np.mean(cwv)
    out["bootstrap_cwm_lower_ci"] = np.percentile(cwm,1-bootstrap_ci)
    out["bootstrap_cwv_lower_ci"] = np.percentile(cwv,1-bootstrap_ci)
    out["bootstrap_cwm_higher_ci"] = np.percentile(cwm,bootstrap_ci)
    out["bootstrap_cwv_higher_ci"] = np.percentile(cwv,bootstrap_ci)

    return out

def cwi(census,traits,col_names=None, bootstrap = False, bootstrap_n = 100, bootstrap_ci=.95):
    out = {}
    col_names = col_names_checker(col_names,[census.columns,traits.columns])            

    census = census.groupby(col_names["species"]).sum()[col_names["n"]].reset_index()
    merged = pd.merge(census.loc[:,[col_names["species"],
                                    col_names["n"]]],
                      traits.loc[:,[col_names["species"],
                                    col_names["trait_val"],
                                    col_names["trait_var"]]])
                      

    out["cwm"] = mean(merged, col_names)
    out["cwv"] = var(merged, col_names)

    
    if bootstrap:
        out_boot = bootstrap_cwi(merged,col_names,bootstrap_n,bootstrap_ci)
        out.update(out_boot)
    return out


def cwi_stratified(census,traits,col_names=None):

    col_names = col_names_checker(col_names,[census.columns,traits.columns])
               
    out = {"cwm":[],
           "cwv":[],
           col_names["date"]:[],
           col_names["site"]:[],
           col_names["n"]:[]}

    merged = pd.merge(census.loc[:,[col_names["species"],
                                    col_names["n"],
                                    col_names["site"],
                                    col_names["date"]]],
                      traits.loc[:,[col_names["species"],
                                    col_names["trait_val"],
                                    col_names["trait_var"]]])
                      
    for site,df in merged.groupby(col_names["site"]):
        for date,ddf in df.groupby(col_names["date"]):
            out["cwm"].append(mean(ddf, col_names))
            out["cwv"].append(var(ddf, col_names))
            out[col_names["date"]].append(date)
            out[col_names["site"]].append(site)
            out[col_names["n"]].append(ddf[col_names["n"]].sum())
    out = pd.DataFrame(out)
    out = out.loc[:,[col_names["site"],
                     col_names["date"],
                    "cwm",
                    "cwv",
                     col_names["n"]]]

    return out
