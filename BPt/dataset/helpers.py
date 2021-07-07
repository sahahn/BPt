import pandas as pd
import numpy as np
import glob
import os
import warnings
from joblib import wrap_non_picklable_objects


def proc_fop(fop):

    # If provided as just % number, divide by 100
    if not isinstance(fop, tuple):
        fop /= 100
        return (fop, 1-fop)

    elif fop[0] is None:
        return tuple([None, 1-(fop[1] / 100)])

    elif fop[1] is None:
        return tuple([fop[0] / 100, None])

    return tuple([fop[0]/100, 1-(fop[1] / 100)])


def auto_file_to_subject(in_path):

    base_name = os.path.basename(in_path)
    return os.path.splitext(base_name)[0]


def proc_file_input(files, file_to_subject):

    if not isinstance(files, dict):
        raise ValueError('files must be passed as a python dict')

    if file_to_subject == 'auto':
        file_to_subject = auto_file_to_subject

    if file_to_subject is None:
        raise RuntimeError('file_to_subject must be specified!')

    # If not passed as dict, convert to dict
    if not isinstance(file_to_subject, dict):
        file_to_subject = {key: file_to_subject for key in files}

    # Check
    for key in files:
        if key not in file_to_subject:
            raise ValueError('If passing file_to_subject as a dict '
                             'then a value must be passed for all '
                             'keys in files. ' + repr(key) + 'was '
                             'not passed in this case.')

    # Compute pd series
    files_series = dict()
    for key in files:

        file_paths = files[key]

        # If passed file path as str, assume globbing
        if isinstance(file_paths, str):
            file_paths = glob.glob(file_paths)

        subjects = [file_to_subject[key](fp) for fp in file_paths]
        files_series[key] = pd.Series(file_paths, index=subjects,
                                      dtype='object')

    return files_series


def base_load_subjects(subjects):

    loaded_subjects = set()

    if isinstance(subjects, str):
        with open(subjects, 'r') as f:
            lines = f.readlines()

            for line in lines:
                subject = line.rstrip()

                try:
                    subject = eval(subject)
                except NameError:
                    subject = subject

                loaded_subjects.add(subject)

    else:
        loaded_subjects = set([s for s in subjects])

    return loaded_subjects


def save_subjects(loc, subjects):

    with open(loc, 'w') as f:
        for subject in subjects:
            f.write(repr(subject) + '\n')


def add_new_categories(existing, new_values):

    # This is only relevant for type category
    if existing.dtype.name != 'category':
        return existing

    # Only add new categories
    new_cats = set(pd.unique(new_values))
    existing_cats = set(existing.dtype.categories)
    to_add = new_cats - existing_cats

    # Add categories
    return existing.cat.add_categories(list(to_add))


def remove_unused_categories(existing):

    # This is only relevant for type category
    if existing.dtype.name != 'category':
        return existing

    to_remove = set(existing.dtype.categories) - set(pd.unique(existing))
    to_remove = to_remove - set([np.nan])
    return existing.cat.remove_categories(list(to_remove))


def get_str_round(val, places=3):

    if isinstance(val, int):
        return val

    return str(np.round(float(val), places))


def verbose_print(self, *args, **kwargs):
    '''Overriding the print function to allow for
    customizable verbosity.

    According to passed level:

    Warnings are level 0,
    Information on sizes / how many dropped are level 1.
    Set to level -1 to mute warnings too.

    Parameters
    ----------
    args
        Anything that would be passed to default python print
    '''

    if 'level' in kwargs:
        level = kwargs.pop('level')
    else:
        level = 1

    if self.verbose >= level:

        # Use warnings for level = 0
        if level == 0:

            # Conv print to str - then warn
            sep = ' '
            if 'sep' in kwargs:
                sep = kwargs.pop('sep')
            as_str = sep.join(str(arg) for arg in args)

            warnings.warn(as_str)

        # Use base print for rest
        else:
            print(flush=True, *args, **kwargs)


def mp_consol_save(data_files, index, cast_to, save_dr):

    # Load the subjects data
    subj_data = [df.load() for df in data_files]

    # Stack the subj data with extra columns at last axis
    subj_data = np.stack(subj_data, axis=-1)

    # Optional cast to dtype
    if cast_to is not None:
        subj_data = subj_data.astype(cast_to)

    # Save as name of index in save loc
    save_loc = os.path.join(save_dr, str(index) + '.npy')
    np.save(save_loc, subj_data)

    return save_loc


def wrap_load_func(load_func, _print):

    if load_func.__module__ == '__main__':
        wrapped_load_func = wrap_non_picklable_objects(load_func)
        _print('Warning: Passed load_func was defined within the',
               '__main__ namespace and therefore has been '
               'cloud wrapped.',
               'The function will still work, but it is '
               'reccomended to',
               'define this function in a separate file, '
               'and then import',
               'it , otherwise loader caching will be limited',
               'in utility!', level=0)
    else:
        wrapped_load_func = load_func

    return wrapped_load_func
