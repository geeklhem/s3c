import numpy as np

def col_names_checker(col_names,
                      existing,
                      default=["n","trait_val","trait_var",
                               "date","site","species","x","y"]):
    """ Verification of columns names dictionnary. 

    Args:
        col_names (dict): Columns names correspondance if they are not default.
        existing (list of list): list of columns in the input.
        default (dict): Default columns names. 
    
    Returns:
        A dict linking the default column name to the actual column name in 
        the input.

    Raises:
        ValueError: If a key of the input dict is not in the possible default 
            values or if a value of the input dict is not in the input columns
            name list (existing).
    """

    default_col_names = dict(zip(default,default))
    
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
