import numpy as np
import pandas as pd

default_col_names = ["n","trait_val","trait_var","date","site","species"]
default_col_names = dict(zip(default_col_names,default_col_names))

def mean(df,col_names):
    return np.dot(df[col_names["trait_val"]],
                  df[col_names["n"]]/float(df[col_names["n"]].sum()))

def var(df,col_names):
    av = mean(df,col_names)
    correct = (df[col_names["n"]]/float(df[col_names["n"]].sum())).sum()
    diff = np.array([p*(i-av)**2 for i,p in zip(df[col_names["trait_val"]],df[col_names["n"]])])
    return diff.sum()/correct

def cwm(census,traits,col_names=None):

    ## Verification of columns names
    if col_names is None:
        col_names = default_col_names
    else:
        for k,v in col_names.items():
            if k not in default_col_names:
                raise ValueError("{} is not a valid column name".format(k))
            if ((v not in census.columns)
                and (v not in traits.columns)):
                raise ValueError("column '{}' is not in the input".format(v))
            
    for n in default_col_names.keys():
        if n not in col_names:
            col_names[n] = n
            
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
