import pandas as pd
import numpy as np
import glob
import os


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
        files_series[key] = pd.Series(file_paths, index=subjects)

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
        return

    # Only add new categories
    new_cats = set(pd.unique(new_values))
    existing_cats = set(existing.dtype.categories)
    to_add = new_cats - existing_cats

    # Add in place
    existing.cat.add_categories(list(to_add), inplace=True)


def get_str_round(val, places=3):

    if isinstance(val, int):
        return val

    return str(np.round(float(val), places))
