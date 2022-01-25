from .Dataset import Dataset
import pandas as pd


def read_csv(*args, **kwargs):
    '''Passes along all arguments and kwargs to
    :func:`pandas.read_csv` then casts to :class:`Dataset`.

    This method is just a helper wrapper function.
    '''

    return Dataset(pd.read_csv(*args, **kwargs))


def _is_equiv(a, b):
    
    if pd.isnull(a) and pd.isnull(b):
        return True
    
    return a == b

def _combine(objs, getter_name, merge_func, data):
    
    vals = {}
    for obj in objs:
        
        # Get col dict of attributes
        col_dict = getattr(obj, getter_name)()
        
        # For each column append value to vals
        for col in col_dict:

            try:
                vals[col].append(col_dict[col])
            except KeyError:
                vals[col] = [col_dict[col]]

            concat_val = col_dict[col]

    # Now we haves vals as dict mapping
    # col to list of all values from the different object
    # to concatenate
    new_vals = {}
    for col in vals:

        # Use merge func to get one value from all passed
        new_vals[col] = merge_func(vals[col], col, objs, data)

    return new_vals

def _error_merge(vals, col, objs=None, data=None):

    v1 = vals[0]
    for v2 in vals[1:]:
        if not _is_equiv(v1, v2):
            raise RuntimeError(f'Found conflicting values {v1} {v2} {col}')
    return v1

def _roles_merge(vals, col=None, objs=None, data=None):

    # If one unique val return that
    u_vals = set(vals)
    if len(u_vals) == 1:
        return vals[0]

    # Otherwise, prefer order as data, target, non input
    if 'data' in u_vals:
        return 'data'
    return 'target'

def _encoder_merge(vals, col, objs, data):

    # Check first to see if all the same
    flag = True
    v1 = vals[0]
    for v2 in vals[1:]:
        if not _is_equiv(v1, v2):
            flag = False
    
    if flag:
        return v1

    # In the case that they differ...
    originals = []
    for val_dict, obj in zip(vals, objs):
        originals.append(obj[col].replace(val_dict))

    # Concat original values together
    original_series = pd.concat(originals, axis=0)

    # Maybe other cases not thinking of, but for now,
    # replace values and ordinalize in place
    data.loc[:, col] = original_series
    data._ordinalize(col)

    # Return the new encoders for this column
    return data.encoders[col]

def _merge_subjects(objs, get_func):

    subjs = getattr(objs[0], get_func)()
    for obj in objs[1:]:
        obj_subjs = getattr(obj, get_func)()
        if subjs is None:
            subjs = obj_subjs
        else:
            subjs = subjs.union(obj_subjs)

    return subjs

def _handle_merge_data_files(objs, data):

    # Get cnt 
    objs[0]._check_file_mapping()
    cnt = objs[0]._get_next_ind()

    # Init file mapping + new col vals
    file_mapping = objs[0].file_mapping.copy()
    new_col_vals = {col: [objs[0][col]] for col in objs[0].get_cols(scope='data file')}

    # For rest of datasets
    for obj in objs[1:]:

        # Gen mapping update  + add to new concat file_mapping
        mapping_update = {}
        obj_file_mapping = obj.get_file_mapping()
        for k in obj_file_mapping:
            mapping_update[k] = k + cnt
            file_mapping[k + cnt] = obj_file_mapping[k]

        # Now need to get changed vals for each data file
        for col in obj.get_cols(scope='data file'):
            
            # Get new values based on mapping update
            values = obj[col].replace(mapping_update)
            
            # Add if already exists, otherwise init as list
            if col in new_col_vals:
                new_col_vals[col].append(values)
            else:
                new_col_vals[col] = [values]
        
        # Update cnt
        cnt += obj._get_next_ind()

    # Go through each of the new col vals and concat + replace
    for col in new_col_vals:
        data.loc[:, col] = pd.concat(new_col_vals[col])

    # Set new file mapping
    data.file_mapping = file_mapping
    data._check_file_mapping()

    # Return new data
    return data
    
def concat(objs, axis=0):

    # TODO warn if resulting concat subjects not unique ? ~
    # TODO add support for more concat pd options ? 
    # TODO write doc
    
    # Make sure passed objects valid
    for i, obj in enumerate(objs):

        # If DataFrame cast to Dataset
        if not isinstance(obj, Dataset) and isinstance(obj, pd.DataFrame):
            objs[i] = Dataset(objs[i])
        
        # Ensure dataset
        if not isinstance(obj, Dataset):
            raise RuntimeError(f'Passed object not Dataset or DataFrame {obj}')

    # Start by using base pandas concat
    data = Dataset(pd.concat(objs=objs, axis=axis))

    # Set verbose to max of any
    data.verbose = max([obj.verbose for obj in objs])
   
    # Merge roles + scopes
    data.roles = _combine(objs, getter_name='get_roles',
                          merge_func=_roles_merge, data=data)
    data.scopes = _combine(objs, getter_name='get_scopes',
                          merge_func=_error_merge, data=data)
    data._check_sr()

    # Handle encoders case
    data.encoders = _combine(objs, getter_name='_get_encoders',
                             merge_func=_encoder_merge, data=data)
    data._check_encoders()

    # Handle any train / test subjects
    data.test_subjects = _merge_subjects(objs, get_func='_get_test_subjects')
    data.train_subjects = _merge_subjects(objs, get_func='_get_train_subjects')
    data._check_test_subjects()
    data._check_train_subjects()

    # Handle any loaded data files
    data = _handle_merge_data_files(objs, data)
    
    return data
    
    
