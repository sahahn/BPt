import numpy as np

from ..helpers.Table_Helpers import (get_split_df, get_base_title, filter_out_nans,
                                     proc_group_by, get_subjects_by_group, save_table)


def Save_Table(self, save_loc, targets='SHOW_ALL',
               covars='SHOW_ALL', strat='SHOW_ALL',
               group_by=None, split=True, include_all=True,
               subjects=None, cat_show_original_name=True,
               shape='long', heading=None,
               center=True, rnd_to=2, style=None):

    '''This method is used to save a table with summary statistics in docx format.

    Warning: if there is any NaN data kept in any of the selected covars, then those
    subject's data will not be outputted to the table! Likewise, only overlapped subjects
    present in any loaded data, covars, strat, targets, ect... will be outputted to the table!.

    Note: you must have the optional library python-docx installed to use this function.

    Parameters
    ----------
    save_loc : str 
        The location where the .docx file with the table should be saved.
        You should include .docx in this save_loc.

    targets : str, int or list, optional
        The single (str) or multiple targets (list),
        in which to add to the outputed table. The str input
        'SHOW_ALL' is reserved, and set to default, for displaying
        all loaded targets.

        You can also pass the int index, (or indices).

        (default = 'SHOW_ALL')

    covars : str or list, optional
        The single (str) or multiple covars (list),
        in which to add to the outputed table. The str input
        'SHOW_ALL' is reserved, and set to default, for displaying
        all loaded covars.

        Warning: If there are any NaN's in selected covars,
        these subjects will be disregarded from all summary measures.

        (default = 'SHOW_ALL')

    strat : str or list, optional
        The single (str) or multiple strat (list),
        in which to add to the outputed table. The str input
        'SHOW_ALL' is reserved, and set to default, for displaying
        all loaded strat.

        Note, if strat is passed, then self.strat_u_name will be removed
        from every title before plotting, so if a same variables is loaded
        as a covar and a strat it will be put in the table twice, and if
        in a rarer case strat_u_name has been changed to a common value present
        in a covar name, then this common key will be removed.

        (default = 'SHOW_ALL')

    group_by : str or list, optional
        This parameter, by default None, controls if the table statistics
        should be further broken down by different binary / categorical (or multilabel)
        groups. For example, by passing 'split',
        (assuming the split param has been left as True), will output a table with
        statistics for each column as seperated by global train test split.
        
        Notably, 'split' is a special keyword, to split on any other group,
        the name of that feature/column should be passed. E.g., to split on a
        loaded covar 'sex', then 'sex' would be passed.

        If a list of values is passed, then each element will be used for its own
        seperate split. Further, if the `include_all` parameter is left as its default
        value of True, then in addition to a single or multiple group_by splits,
        the values over all subjects will also be displayed. (E.g. in the train test
        split case, statistics would be shown for the full sample, only the train subjects
        and only the test subjects).

        (default = None)

    split : bool, optional
        If True, then information about the global train test split will be added
        as a binary feature of the table. If False, or if a global split has not
        yet been defined, this split will not be added to the table. Note that
        'split' can also be passed to `group_by`.

        Note: If it is desired to create a table with just the train subjects (or test)
        for example, then the `subjects` parameter should be used. If subjects is set
        to 'train' or 'test', this `split` param will switch to False regardless of
        user passed input.

        (default = True)

    include_all : bool, optional
        If True, as default, then in addition to passed group_by param(s),
        statistics will be displayed over all subjects as well. If no group_by
        param(s) are passed, then `include_all` will be True regardless of user passed value.

        (default = True)

    subjects : None, 'train', 'test' or array-like, optional
        If left as None, display a table with all overlapping subjects.
        Alternatively this parameter can be used to pass just the train_subjects
        with 'train' or just the test subjects with 'test' or even a custom
        list or array-like set of subjects

        (default = None)

    cat_show_original_name : bool, optional
        If True, then when showing a categorical distribution (or binary)
        use the original loaded name in the table. Otherwise, if False, the
        internally used variable names will be used to determine table headers.

        (default = True)

    shape : {'long', 'wide'}, optional
        There are two options for shape. First 'long', which
        is selected by default, will create a table where each row
        is a summary statistic for a passed variable, and each column
        represents each `group_by` group if any. This is a good choice
        when the number of variable to plot is more than the number of
        groups to display values by. 

        Alternatively, you may pass 'wide', for the opposite behavior of
        'long', where in this case the variables to summarize will each
        be a column in the table, and the group_by groups will represent 
        rows.

        If your table ends up being too squished in either direction, you can 
        try the opposite shape. If both are squished, you'll need to reduce the
        number of group_by variables or targets, covars/ strat. 
        
        (default = 'long')

    heading : str, optional
        You may optionally pass a heading for the table as a str.
        By default no heading will be added.
        
        (default = None)
    
    center : bool, optional
        This parameter optionally determines if the values in the table
        along with the headers should be centered within each cell. If False,
        the values will be left alligned on the bottom.

        (default = True)

    rnd_to : int, optional
        This parameter determines how many decimal places each value
        to be added to the table should be rounded to. E.g., the default
        value of 2 will round each table entry to 2 decimal points, but if
        rnd_to 0 was passed, that would be no decimal points and -1, would be
        rounded to the nearest 10. 

        (default = 2)

    style : str or None, optional
        The default .docx table style in which to use, which contrals things like
        table color and fonts, ect... Keep style as None by default to just
        use the default style, or feel free to try passing any of a huge number of
        preset styles. These styles can be found at the bottom of
        https://python-docx.readthedocs.io/en/latest/user/styles-understanding.html
        under "Table styles in default template".

        Some examples are: 'Colorful Grid', 'Dark List', 'Light List', 'Medium List 2',
        'Medium Shading 2 Accent 6', ect...

        (default = None)
    '''

    try:
        import docx
    except ImportError:
        raise ImportError('Install python-docx to use this function!')

    if subjects == 'train' or subjects == 'test':
        split = False

    # Load the relevant single_dfs, encoders, ect.. nec to make the tables
    dfs, encoders, names = self._get_single_dfs(targets=targets, covars=covars,
                                                strat=strat, split=split,
                                                subjects=subjects)

    # Filter out NaN's w/ warning if any get removed
    before = len(dfs[0])
    dfs = filter_out_nans(dfs)
    if len(dfs[0]) != before:
        self._print('Warning:', before - len(dfs[0]),
                    'subjects will not be considered due to NaNs!')

    # Seperate out the group by dfs/encoders
    gb_dfs, gb_encoders, gb_names =\
        proc_group_by(group_by, include_all, dfs, encoders,
                      names, self.strat_u_name)

    # Get the values for each group by group
    gb_values = []

    for grp in range(len(gb_names)):
        subjects_by_group = get_subjects_by_group(gb_dfs[grp], gb_encoders[grp], gb_names[grp])
        
        for subjects in subjects_by_group:
            indv_titles, values =\
                self._get_table_contents(dfs, encoders, names,
                                         subjects=subjects,
                                         cat_show_original_name=cat_show_original_name,
                                         rnd_to=rnd_to)
            gb_values.append(values)

    # Get the specific group titles
    gb_titles = self._get_group_titles(gb_dfs, gb_encoders,
                                       gb_names, cat_show_original_name)

    # Replace the strat_u_name in the titles if present here
    gb_titles = [g.replace(self.strat_u_name, '') for g in gb_titles]
    indv_titles = [i.replace(self.strat_u_name, '') for i in indv_titles]

    # Save the table as a docx with the passed params
    save_table(save_loc, gb_titles, indv_titles, gb_values,
               shape=shape, heading=heading, style=style,
               center=center)


def _get_single_dfs(self, targets='SHOW_ALL', covars='SHOW_ALL', strat='SHOW_ALL', split=True,
                   show_only_overlap=True, subjects=None):
    
    all_names = [targets, covars, strat]
    class_dfs = [self.targets, self.covars, self.strat]
    funcs = [self._input_targets, self._input_covars, self._input_strat]
    all_encoders = [self.targets_encoders, self.covars_encoders, self.strat_encoders]
    
    return_dfs = []
    return_encoders = []
    return_names = []
    
    for names, class_df, func, encoders in zip(all_names, class_dfs, funcs, all_encoders):
        
        if (names is not None) and (len(class_df) > 0):
    
            if subjects is None:
                df = self._set_overlap(class_df, show_only_overlap).copy()
            else:
                df = self._proc_subjects(class_df, subjects)
                
            names = func(names)
            
            for name in names:
                single_df, encoder, _ = self._get_single_df(name, df, encoders)
                return_dfs.append(single_df)
                return_encoders.append(encoder)
                return_names.append(name)
                
    if split and self.train_subjects is not None:
                
        split_df = get_split_df(self.subject_id, self.train_subjects, self.test_subjects)
        split_encoders = {'Train': 'Train', 'Test': 'Test'}
        
        return_dfs.append(split_df)
        return_encoders.append(split_encoders)
        return_names.append('split')

    return return_dfs, return_encoders, return_names


def _get_table_contents(self, dfs, encoders, names,
                       subjects=None, cat_show_original_name=True,
                       rnd_to=2):
    
    titles = []
    values = []

    for df, encoder, name in zip(dfs, encoders, names):
        
        if subjects is not None:
            df = df.loc[subjects]

        # Regression
        if encoder is None:

            title = name + ' (std)'

            value = str(np.round(np.mean(df[name]), rnd_to))
            rnded = str(np.round(np.std(df[name]), rnd_to))
            value += ' (±' + rnded + ')'

            titles.append(title)
            values.append(value)

        # Binary/ordinal or one-hot
        else:

            display_df, _, _ = self._get_cat_display_df(df, encoder, name,
                                                        cat_show_original_name)

            for ind in display_df.index:
                
                title = get_base_title(display_df, name,
                                       ind, cat_show_original_name)
                title += ' (Freq.)'
                
                row = display_df.loc[ind]
                value = str(int(row['Counts']))
                rnded = str(np.round(row['Frequency'] * 100, rnd_to))
                value += ' (' + rnded + '%)'

                titles.append(title)
                values.append(value)
                
    return titles, values


def _get_group_titles(self, gb_dfs, gb_encoders, gb_names, cat_show_original_name):
    
    titles = []
    
    for grp in range(len(gb_dfs)):
        
        display_df, _, _ =\
            self._get_cat_display_df(gb_dfs[grp], gb_encoders[grp],
                                     gb_names[grp], cat_show_original_name)

        for ind in display_df.index:
            title = get_base_title(display_df, gb_names[grp],
                                   ind, cat_show_original_name)
            titles.append(title)
            
    return titles


