import numpy as np

def col_names_checker(col_names,
                      existing,
                      default=["n","trait_val","trait_var","date","site","species","x","y"]):
    default_col_names = dict(zip(default,default))

    ## Verification of columns names
    if col_names is None:
        col_names = default_col_names
    else:
        for k,v in col_names.items():
            if k not in default_col_names:
                raise ValueError("{} is not a valid column name".format(k))
            elif not np.any([v in x for x in existing]):
                raise ValueError("column '{}' is not in the input".format(v))
    default_col_names.update(col_names)
    return default_col_names
