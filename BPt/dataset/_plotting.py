import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pandas.util._decorators import doc
from ..util import save_docx_table, get_top_substrs
from .dataset import _file_docs, _shared_docs

_plot_docs = _file_docs.copy()
_plot_docs['scope'] = _shared_docs['scope']
_plot_docs['subjects'] = _shared_docs['subjects']


_plot_docs['decode_values'] = '''decode_values : bool, optional
        When handling categorical variables
        that have been encoded through a BPt
        dataset method, e.g., :func:`Dataset.ordinalize`,
        then you may optionally either use either
        the original categorical values before encoding
        with decode_values = True, or use the current
        internal values with decode_values = False.

        ::

            default = True
'''


def _show_values(axs, count=True, space=0):
    '''code from https://www.statology.org/seaborn-barplot-show-values/'''

    def _single(ax):

        longest = 0

        for p in ax.patches:
            _x = p.get_x() + p.get_width() + float(space)
            _y = p.get_y() + p.get_height() - (p.get_height()*0.5)

            if count:
                value = f'{int(p.get_width())}'
            else:
                value = f'{p.get_width():.2f}'

            # Keep track of longest label
            if len(value) > longest:
                longest = len(value)

            ax.text(_x, _y, value, ha="left")

        # Scale x lim a bit to compensate for adding extra text
        # Specifically by longest value
        scale =  1  + (longest / 100)
        ax.set_xlim([scale*x for x in ax.get_xlim()])

    if isinstance(axs, np.ndarray):
        for ax in axs:
            _single(ax)

    else:
        _single(axs)


def nan_info(self, scope='all'):

    # Get data based on passed scope
    cols = self.get_cols(scope)
    data = self[cols]

    # Check for data files
    self._data_file_fail_check(cols)

    # Get NaN counts
    na_counts = data.isna().sum().sort_values(ascending=False)

    if na_counts.sum() > 0:
        print('Loaded NaN Info:')
        print('There are:', na_counts.sum(), 'total missing values')

        u_counts, c_counts = np.unique(na_counts, return_counts=True)
        u_counts, c_counts = u_counts[1:], c_counts[1:]

        inds = c_counts.argsort()
        u_counts = u_counts[inds[::-1]]
        c_counts = c_counts[inds[::-1]]

        for u, c in zip(u_counts, c_counts):
            if c > 1:

                keys = list(na_counts[na_counts == u].index)
                substrs = get_top_substrs(keys)

                print(c, ' columns found with ', u, ' missing values',
                         ' (column name overlap: ', substrs, ')', sep='')

        print()


def _cont_info(self, cont_cols, subjs, measures, decimals, **extra_args):

    if len(cont_cols) == 0:
        return pd.DataFrame()

    # Init info df
    info_df = pd.DataFrame(index=pd.Series(cont_cols))

    # For each float column in scope
    for col in cont_cols:

        # Get the values as a series
        values, info =\
            self._get_plot_values(col, subjs,
                                  print_info=False,
                                  **extra_args)

        # Compute each measure
        for measure in measures:

            measure = measure.replace('+-', '±')

            if measure == 'mean':
                info_df.loc[col, measure] = values.mean()

            elif measure == 'mean ± std':
                mean = str(np.round(values.mean(), decimals))
                std = str(np.round(values.std(), decimals))
                info_df.loc[col, measure] = f'{mean} ± {std}'

            elif measure == 'max':
                info_df.loc[col, measure] = values.max()

            elif measure == 'min':
                info_df.loc[col, measure] = values.min()

            elif measure == 'std':
                info_df.loc[col, measure] = values.std()

            elif measure == 'var':
                info_df.loc[col, measure] = values.var()

            elif measure == 'skew':
                info_df.loc[col, measure] = values.skew()

            elif measure == 'kurtosis':
                info_df.loc[col, measure] = values.kurtosis()

            elif measure == 'count':
                info_df.loc[col, measure] = info['n']
                info_df[measure] = info_df[measure].astype(pd.Int64Dtype())

            elif measure == 'nan count':
                info_df.loc[col, measure] = int(info['n_nan'])
                info_df[measure] = info_df[measure].astype(pd.Int64Dtype())

            else:
                raise RuntimeError('Invaid measure: ', measure)

    return info_df


def _cat_info(self, cat_cols, subjs, cat_measures, **extra_args):

    if len(cat_cols) == 0:
        return pd.DataFrame()

    # Init info df
    info_df = pd.DataFrame()

    # For each column
    for col in cat_cols:

        # Get the values as a series
        values, info =\
            self._get_plot_values(col, subjs,
                                  print_info=False,
                                  **extra_args)

        # Compute the unique categories + their counts
        cats, counts = np.unique(values, return_counts=True)

        # Add each requested measure
        for measure in cat_measures:

            # @TODO add skip if too many categories.

            # Add first the whole column summary
            if measure == 'count':
                info_df.loc[col, measure] = info['n']
                info_df[measure] = info_df[measure].astype(pd.Int64Dtype())

            elif measure == 'freq':
                info_df.loc[col, measure] = 1

            elif measure == 'nan count':
                info_df.loc[col, measure] = int(info['n_nan'])
                info_df[measure] = info_df[measure].astype(pd.Int64Dtype())

            else:
                pass

            # Now for each unique cat
            for cat, cnt in zip(cats, counts):
                name = col + '=' + repr(cat)

                if measure == 'count':
                    info_df.loc[name, measure] = cnt

                elif measure == 'freq':
                    info_df.loc[name, measure] = cnt / info['n']

                elif measure == 'nan count':
                    info_df.loc[name, measure] = 0

                else:
                    pass

    return info_df


@doc(**_plot_docs)
def summary(self, scope,
            subjects='all',
            measures=['count', 'nan count',
                      'mean', 'max',
                      'min', 'std', 'var',
                      'skew', 'kurtosis'],
            cat_measures=['count', 'freq', 'nan count'],
            decode_values=True,
            save_file=None,
            decimals=3,
            reduce_func=np.mean,
            n_jobs=-1):
    '''This method is used to generate a summary across
    some data.

    Parameters
    ------------
    {scope}

    {subjects}

    measures : list of str, optional
        The summary measures which should
        be computed for any float /  continuous type
        columns within the passed scope.

        Valid options are:

        - 'count'
            Calculates the number of non-missing data points
            for each column.

        - 'nan count'
            Calculates the number of missing data points in
            this column, which are excluded from other statistics.

        - 'mean'
            Calculates the mean value for each column.

        - 'max'
            Calculates the maximum value for each column.

        - 'min'
            Calculates the minimum value for each column.

        - 'std'
            Calculates the standard deviation for each column.

        - 'var'
            Calculates the variance for each column.

        - 'skew'
            Calculates the skew for each column.

        - 'kurtosis'
            Calculates the kurtosis for each column.

        - 'mean +- std'
            Return the mean and std as str rounded
            to decimals as mean ± std.

        These values should be passed as a list.

        ::

            default =  ['count', 'nan count',
                        'mean', 'max', 'min',
                        'std', 'var',
                        'skew', 'kurtosis']

    cat_measures : list of str, optional
        These measures will be used to compute statistics
        for every categorical column within the passed scope.
        Likewise, these measures will be used to compute statistics
        by each unique class value for each categorical measure.

        Valid options are:

        - 'count'
            Calculates the number of non-missing data points
            for each column and unique value.

        - 'freq'
            Calculate the percentage of values that
            each unique value makes up. Note:
            for column measures this will always be 1.

        - 'nan count'
            Calculates the number of missing data points in
            this column, which are excluded from other statistics.
            Note: For class values this will always be 0.


        These values should be passed as a list.

        ::

            default =  ['count', 'freq', 'nan count']

    {decode_values}

    save_file : None or str, optional
        You may optionally save this info
        description to a docx file in a table.
        If set to a str, this string should
        be the path to a docx file, where if it
        exists, the table will be added, and if it
        doesn't, the table with summary stats
        will be created as a new file.

        Keep as None, to skip this option.

        ::

            default = None

    decimals : int, optional
        If save_file is not None, then this
        parameter sets the number of decimal
        points to which values in the saved
        table will be rounded to.

        This parameter will also be used in the case
        that a special str measure is requested,
        e.g., mean +- std.

        ::

            default = 3

    {reduce_func}

    {n_jobs}

    Returns
    ------------
    cont_info_df : pandas DataFrame
        A dataframe containing the summary statistics
        as computed for any float / continuous type
        data. If None, then this DataFrame will be empty.

        This corresponds to the measures argument.

    cat_info_df : pandas DataFrame
        A dataframe containing the summary statistics
        as computed for any categorical type
        data. If None, then this DataFrame will be empty.

        This corresponds to the cat_measures argument.
    '''

    extra_args = {'decode_values': decode_values,
                  'reduce_func': reduce_func,
                  'n_jobs': n_jobs}

    # Get cols and subjects
    cols = self.get_cols(scope)
    subjs = self.get_subjects(subjects)

    # Compute for just the non-categorical columns
    cont_cols = [col for col in cols if 'category' not in self.scopes[col]]
    cont_info_df = _cont_info(self, cont_cols, subjs, measures,
                              decimals, **extra_args)

    # Compute for just categorical columns
    cat_cols = [col for col in cols if 'category' in self.scopes[col]]
    cat_info_df = _cat_info(self, cat_cols, subjs, cat_measures, **extra_args)

    # If save is not None
    if save_file is not None:

        if len(cont_info_df) > 0:
            cont_info_df.index.name = 'columns'
            save_docx_table(cont_info_df, save_file, decimals=decimals)

        if len(cat_info_df) > 0:
            cat_info_df.index.name = 'columns'
            save_docx_table(cat_info_df, save_file, decimals=decimals)

    return cont_info_df, cat_info_df

@doc(**_plot_docs)
def plot(self, scope,
         subjects='all',
         cut=0,
         decode_values=True,
         count=True,
         show=True,
         reduce_func=np.mean,
         n_jobs=-1,
         axes=None,
         ax=None):
    '''This function creates plots for each of the passed
    columns (as specified by scope) seperately.

    Parameters
    -----------
    {scope}

    {subjects}

    {decode_values}

    cut : float, optional
        Only for plotting non-categorical variables.
        Factor, multiplied by the smoothing bandwidth,
        that determines how far the evaluation grid
        extends past the extreme datapoints.
        When set to 0, truncate the curve at the data limits.

        ::

            default = 0

    count : bool, optional
        Only for plotting categorical variables.
        If True, then display the counts, if
        False, then display the frequency out of 1.

        ::

            default = True

    show : bool or 'default', optional
        If plt.show() from matplotlib should
        be called after plotting each column within
        the passed scope. You will typically
        want this parameter to be the default True,
        as when plotting multiple columns, they might
        otherwise overlap.

        In the case that you are only plotting
        one column, and want to make customizations,
        then you should pass this value as False,
        and you can make changes to the figure
        via matplotlib's global state system.

        Note: If axes or ax is passed as non-null,
        then default swaps to False. Otherwise default
        is True.

        ::

            default = 'default'

    {reduce_func}

    {n_jobs}

    axes : list of Axes or None, optional
        Can optionally pass a list of axes in which
        to add each  plot to. By default plot each seperate

        ::

            default = None

    ax : Axes or None, optional
        Simmilar to axes, but in the case of only
        plotting a single column, this parameter can also be
        used.

        ::

            default = None

    Examples
    ---------
    This example shows plotting a float feature.

    .. plot::
        :context: close-figs

        data = bp.Dataset()
        data['1'] = [.1, .1, .2, .2, .3, .3, .4, .4, .5, .5]
        data.plot(scope='1')

    '''

    plot_args = {'cut': cut,
                 'decode_values': decode_values,
                 'count': count,
                 'show': show,
                 'reduce_func': reduce_func,
                 'n_jobs': n_jobs}

    # Grab cols to plot
    # Get cols will call check scope.
    cols = self.get_cols(scope)

    # Handle axes / ax / show defalt
    if axes is None:
        axes = [None for _ in range(len(cols))]
    elif show == 'default':
        show = False

    if ax is not None:
        axes = [ax]
        
        if show == 'default':
            show = False

    # If show still default, set to True
    if show == 'default':
        show = True

    # Grab subjs to plot
    subjs = self.get_subjects(subjects, return_as='flat index')

    # Plot each column
    for col, ax in zip(cols, axes):
        self._plot_col(col=col, subjs=subjs, ax=ax, **plot_args)


@doc(**_plot_docs)
def plots(self, scope,
          subjects='all',
          ncols=3,
          figsize='default',
          cut=0, decode_values=True,
          count=True, show=True,
          reduce_func=np.mean, n_jobs=-1):
    '''This function creates a multi-figure plot containing all
    of the passed columns (as specified by scope) in their own axes.

    Parameters
    -----------
    {scope}

    {subjects}

    ncols : int, optional
        Number of columns to plot by.

        ::

            default =  3

    figsize : 'default' or tuple, optional
        The size of the subplot to initialize.

        Default will try to scale to number of rows and cols

        ::

            default = 'default'

    {decode_values}

    cut : float, optional
        Only for plotting non-categorical variables.
        Factor, multiplied by the smoothing bandwidth,
        that determines how far the evaluation grid
        extends past the extreme datapoints.
        When set to 0, truncate the curve at the data limits.

        ::

            default = 0

    count : bool, optional
        Only for plotting categorical variables.
        If True, then display the counts, if
        False, then display the frequency out of 1.

        ::

            default = True

    show : bool, optional
        If plt.show() from matplotlib should
        be called after plotting each column within
        the passed scope. You will typically
        want this parameter to be the default True,
        as when plotting multiple columns, they might
        otherwise overlap.

        If False, return (fig, axes), otherwise
        if True, return None.

        ::

            default = True

    {reduce_func}

    {n_jobs}

    Returns
    -----------
    fig, axes : Figure and Axes or None
        If show is True, None is returned, otherwise
        the subplot figure + its Axes are returned.

    Examples
    ---------
    This example shows plotting a simple collage over three fake features.

    .. plot::
        :context: close-figs

        data = bp.Dataset([[1, 2, 3], [2,  2, 2], [3, 3, 3]], columns=['f1', 'f2', 'f3'])
        data.plots(scope='all')
    '''

    # Fix show False
    plot_args = {'cut': cut,
                 'decode_values': decode_values,
                 'count': count,
                 'show': False,
                 'reduce_func': reduce_func,
                 'n_jobs': n_jobs}
    
    # Get cols + subjs
    cols = self.get_cols(scope)
    subjs = self.get_subjects(subjects, return_as='flat index')

    # Determine n_rows + figsize
    if len(cols) == 1:
        nrows = 1
    else:
        nrows = (len(cols) // ncols) + (len(cols) % ncols)

    if figsize == 'default':
        figsize =  (8 * ncols, 6 * nrows)
    
    # Init subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # Single figure case
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])

    # Plot each column
    for col, ax in zip(cols, axes.reshape(-1)):
        self._plot_col(col=col, subjs=subjs, ax=ax, **plot_args)

    # Set any remainder blank
    for ax in axes.reshape(-1)[len(cols):]:
        ax.axis('off')

    # Lastly call show if needed
    if show:
        plt.show()
    else:
        return fig, axes
        

def _plot_col(self, col, subjs,
              print_info=True, ax=None,
              **plot_args):

    # If categorical
    if 'category' in self.scopes[col]:
        return self._plot_category(col=col,
                                   subjs=subjs,
                                   print_info=print_info,
                                   ax=ax,
                                   **plot_args)

    # Otherwise float
    return self._plot_float(col=col, subjs=subjs,
                            print_info=print_info,
                            ax=ax, **plot_args)


def _plot_float(self, col, subjs,
                print_info=True, ax=None,
                **plot_args):

    # Get values to plot
    values, info = self._get_plot_values(col, subjs,
                                         print_info=print_info,
                                         **plot_args)

    # Plot values
    axes = sns.kdeplot(values, cut=plot_args['cut'], ax=ax)

    # Set axes labels
    axes.set_xlabel('Values')
    axes.set_title(str(col) + ' Distribution')

    if plot_args['show']:
        plt.show()

    return info


def _plot_category(self, col, subjs, print_info=True,
                   ax=None, **plot_args):

    # Get plot values
    values, info = self._get_plot_values(col, subjs,
                                         print_info=print_info,
                                         **plot_args)

    # Don't plot if more than 50 categories
    if len(values.unique()) >= 50:
        self._print('Skipping plot: ' + str(col) +
                    ' as >= categories!', level=0)
        return

    # Get counts
    counts = pd.DataFrame(
        values.value_counts(normalize=not plot_args['count'],
                            dropna=False))

    # Reset index
    counts = counts.reset_index()

    # Plot w/ axes level box plot arg
    axes = sns.barplot(x=col, y="index", data=counts,
                       orient='h', ci=None, ax=ax)

    # Add the raw value also as label
    # might fail w/ some library versions, if so, just skip
    try:
        _show_values(axes, count=plot_args['count'])
    except AttributeError:
        pass

    # Set axes label
    if plot_args['count']:
        axes.set_xlabel('Counts')
    else:
        axes.set_xlabel('Frequency')

    axes.set_ylabel('Categories')

    # Add title w/ col name + distribution tag
    axes.set_title(str(col) + ' Distribution')

    # If show, call show
    if plot_args['show']:
        plt.show()

    return info


def _get_plot_values(self, col, subjs,
                     print_info=True, **plot_args):

    # Get values
    values = self.get_values(col, dropna=True,
                             decode_values=plot_args['decode_values'],
                             reduce_func=plot_args['reduce_func'],
                             n_jobs=plot_args['n_jobs'])

    # Get subset of subjects
    overlap_subjs = subjs.intersection(values.index)
    values = values.loc[list(overlap_subjs)]

    # Compute info
    info = {'n': len(values),
            'n_nan': len(subjs) - len(values)}

    # Print info only if requested
    if print_info:
        self._print_plot_info(col, info)

    return values, info


def _print_plot_info(self, col, info):

    self._print(str(col) + ':', str(info['n']), 'rows', end='', level=2)

    if info['n_nan'] > 0:
        self._print(' (' + str(info['n_nan']), 'NaN)',
                    end='', level=2)

    self._print(level=2)


@doc(**_file_docs, subjects=_shared_docs['subjects'],
     decode_values=_plot_docs['decode_values'])
def plot_bivar(self, scope1, scope2, subjects='all',
               decode_values=True, show=True,
               reduce_func=np.mean,
               n_jobs=-1):
    '''This method can be used to plot the relationship
    between two variables. Different types of plots will
    be used based on the types of the underlying variables.

    Parameters
    -----------
    scope1 : :ref:`Scope`
        The name(s) of the first set of variables in
        which to plot against scopes2.

    scope2 : str
        The name(s) of the second set of variables in
        which to plot against scopes2.

    {subjects}

    {decode_values}

    show : bool, optional
        If plt.show() from matplotlib should
        be called after plotting each column within
        the passed scope.

        ::

            default = True

    {reduce_func}

    {n_jobs}

    '''

    plot_args = {'decode_values': decode_values,
                 'show': show,
                 'reduce_func': reduce_func,
                 'n_jobs': n_jobs}
    self._check_scopes()

    # Grab subjs to plot
    subjs = self.get_subjects(subjects, return_as='flat index')

    # Get each set of cols
    cols1 = self.get_cols(scope1)
    cols2 = self.get_cols(scope2)

    # Plot every combination of 1 and 2
    for c1 in cols1:
        for c2 in cols2:
            self._plot_bivar(c1, c2, subjs, **plot_args)

    # Something like this could be cool for 3x3 bivar
    # g = sns.PairGrid(data[list(data['data'])[:3]])
    # g.map_upper(sns.histplot)
    # g.map_lower(sns.kdeplot, fill=True)
    # g.map_diag(sns.histplot, kde=True)

def _plot_bivar(self, col1, col2, subjs, **plot_args):

    # Cat
    if 'category' in self.scopes[col1]:

        # Cat x Cat
        if 'category' in self.scopes[col2]:
            self._plot_cat_cat(col1, col2, subjs=subjs,
                               **plot_args)

        # Cat x Float
        else:
            self._plot_cat_float(cat_col=col1,
                                 float_col=col2,
                                 subjs=subjs,
                                 **plot_args)

    # Float
    else:

        # Float x Cat
        if 'category' in self.scopes[col2]:
            self._plot_cat_float(cat_col=col2,
                                 float_col=col1,
                                 subjs=subjs,
                                 **plot_args)

        # Float x Float
        else:
            self._plot_float_float(col1, col2, subjs=subjs, **plot_args)

    # Show at end if requested
    if plot_args['show']:
        plt.show()


def _plot_cat_cat(self, col1, col2, subjs, **plot_args):

    values1, _ = self._get_plot_values(col1, subjs, **plot_args)
    values2, _ = self._get_plot_values(col2, subjs, **plot_args)

    # Get overlap
    overlap = values1.index.intersection(values2.index)
    self._print('Plotting', len(overlap), 'overlap valid subjects.', level=2)

    values1 = values1.loc[overlap]
    values2 = values2.loc[overlap]

    # Convert to df
    df = values1.to_frame()
    df[col2] = values2

    # Get counts
    ct_counts = df.groupby([col1, col2]).size()
    ct_counts = ct_counts.reset_index(name='Count')
    ct_counts = ct_counts.pivot(index=col1, columns=col2, values='Count')

    sns.heatmap(ct_counts, annot=True, cmap="YlGnBu", fmt='g')


def _plot_float_float(self, col1, col2, subjs, **plot_args):

    values1, _ = self._get_plot_values(col1, subjs, **plot_args)
    values2, _ = self._get_plot_values(col2, subjs, **plot_args)

    # Get overlap
    overlap = values1.index.intersection(values2.index)
    self._print('Plotting', len(overlap), 'overlap valid subjects', level=2)

    values1 = values1.loc[overlap]
    values2 = values2.loc[overlap]

    sns.jointplot(x=values1, y=values2, kind="reg")


def _plot_cat_float(self, cat_col, float_col, subjs, **plot_args):

    cat_values, _ = self._get_plot_values(cat_col, subjs,
                                          **plot_args)
    float_values, _ = self._get_plot_values(float_col, subjs,
                                            **plot_args)

    # Get overlap
    overlap = cat_values.index.intersection(float_values.index)
    self._print('Plotting', len(overlap), 'overlap valid subjects.', level=2)

    sns.displot(x=float_values.loc[overlap], hue=cat_values.loc[overlap],
                kind="kde")


"""
Note sure if want to re-create this given how buggy it is, but here is
reference old code.

def Show_Data_Dist(self, data_subset='SHOW_ALL',
                   num_feats=20, feats='random',
                   reduce_func=None,
                   frame_interval=500,
                   plot_type='hist', show_only_overlap=True,
                   subjects=None, save=True, dpi='default',
                   save_name='data distribution', random_state='default',
                   return_anim=False):

    '''This method displays some summary statistics about
    the loaded targets, as well as plots the dist if possible.

    Note: to display loaded data files, pass a fun to reduce_func, otherwise
    they will not be displayed.

    Parameters
    ----------
    data_subset : 'SHOW_ALL' or array-like, optional
        'SHOW_ALL' is reserved for showing
        the distributions of loaded data.
        You may also pass a list/array-like to specify specific
        a custom source of features to show.

        If self.all_data is already prepared, this data subset can also include
        any float type features loaded as covar or target.

        ::

            default = 'SHOW_ALL'

    num_feats: int, optional
        The number of features' distributions in which to view.
        Note: If too many are selected it may take a long time to render
        and/or consume a lot of memory!

        ::

            default = 20

    feats : {'random', 'skew'}, optional
        The features in which to display, if 'random' then
        will select `num_feats` random features to display.
        If 'skew', will show the top `num_feats` features by
        absolute skew.

        If 'skew' and subjects == 'both', will
        compute the top skewed features based on
        the training set.

        ::

            default = 'random'

    reduce_func : python function or list of, optional
        If a function is passed here, then data files will be loaded
        and reduced to 1 number according to the passed function.
        For example, the default function is just to take the
        mean of each loaded file, and to compute outlier detection
        on the mean.

        To not display data files, if any, then just keep
        reduce func as None

        ::

            default = None

    frame_interval: int, optional
        The number of milliseconds between each frame.

        ::

            default = 500

    plot_type : {'bar', 'hist', 'kde'}
        The type of base seaborn plot to generate for each datapoint.
        Either 'bar' for barplot, or 'hist' for  dist plot, or
        'kde' for just a kernel density estimate plot.

        ::

            default = 'hist'

    show_only_overlap : bool, optional
        If True, then displays only the distributions for valid overlapping
        subjects across data, covars, ect... otherwise, if False,
        shows the current loaded distribution as is.

        If subjects is set (anything but None), this param will be ignored.

        ::

            default = True

    subjects : None, 'train', 'test', 'both' or array-like, optional
        If not None, then plot only the subjects loaded as train_subjects,
        or as test subjects, of you can pass a custom list or array-like of
        subjects.

        If 'both', then will plot the train and test distributions seperately.
        Note: This only works for plot_type == 'hist' or 'kde'.
        Also take into account, specifying 'both' will show some
        different information, then the default settings.

        ::

            default = None

    save : bool, optional
        If the animation should be saved as a gif, True or False.

        ::

            default = True

    dpi : int, 'default', optional
        The dpi in which to save the distribution gif.
        If 'default' use the class default value.

        ::

            default = 'default'

    save_name : str, optional
        The name in which the gif should be saved under.

        ::

            default = 'data distribution'

    random_state : 'default', int or None
        The random state in which to choose random features from.
        If 'default' use the class define value, otherwise set to the value
        passed. None for random.

        ::

            default = 'default'

    return_anim : bool, optional
        If True, return just the animation

        ::

            default = False

    '''

    # If data in low memory work for all data instead
    if len(self.data) == 0:
        valid_data = self.all_data.copy()[self.Data_Scopes.data_keys]
    else:
        valid_data = self.data.copy()
        if subjects is None:
            valid_data = self._set_overlap(valid_data, show_only_overlap)

    # If not all passed data subset are in data, try all data instead
    if data_subset != 'SHOW_ALL':
        if all([feat in valid_data for feat in data_subset]):
            valid_data = valid_data[data_subset]
        else:
            valid_data = self.all_data.copy()[data_subset]

    # If loading data files also
    if reduce_func is None:
        valid_data = valid_data.drop(self.data_file_keys, axis=1)
    else:
        data_file_data =\
            load_data_file_proxies(valid_data, [reduce_func],
                                   self.data_file_keys,
                                   self.file_mapping,
                                   n_jobs=self.n_jobs)[0]

        valid_data = valid_data.drop(self.data_file_keys, axis=1)
        valid_data = pd.merge(valid_data, data_file_data, on=self.subject_id)

    if random_state == 'default':
        random_state = self.random_state

    self._print('Plotting data distribution.')

    if subjects == 'both':
        if plot_type == 'bar':
            raise RuntimeWarning('Switching plot type to dist due to subjects',
                                 ' == "both"')
            plot_type = 'dist'

        data, test_data = self._proc_subjects(valid_data, subjects)

        self._print('Viewing train data with shape:', data.shape)
        self._print('Viewing test data with shape:', test_data.shape)

    else:

        data = self._proc_subjects(valid_data, subjects)

        self._print('Viewing data with shape:', data.shape)
        self._print()

        self._print('Loaded data top columns by skew:')
        self._print(valid_data.skew().sort_values())
        self._print()

    fig = plt.figure()

    def update(i):
        fig.clear()
        title = ''

        if subjects == 'both':
            non_nan_col_tr, n_tr = _get_col(data, i)
            non_nan_col_test, n_test = _get_col(test_data, i)

            if n_tr > 0 or n_test > 0:
                title = 'Train - NaN subjects not shown: ' + str(n_tr)
                title += '\nTest - NaN subjects not shown: ' + str(n_test)

            _plot_seaborn_dist(non_nan_col_tr, plot_type, label='train')
            _plot_seaborn_dist(non_nan_col_test, plot_type, label='test')

            plt.legend()

        else:
            non_nan_col, n = _get_col(data, i)

            if n > 0:
                title = 'NaN subjects not shown: ' + str(n)

            _plot_seaborn_dist(non_nan_col, plot_type)

        plt.title(title, fontdict={'fontsize': 'medium'})

    if 'skew':
        most_skewed = data.skew().abs().sort_values()[-num_feats:].index
        frames = [list(data).index(m) for m in most_skewed][::-1]
    else:
        np.random.seed(random_state)
        frames = np.random.randint(0, data.shape[1], size=num_feats)

    anim = FuncAnimation(fig, update, frames=frames, interval=500)
    if return_anim:
        return anim

    try:
        html = HTML(anim.to_html5_video())
    except RuntimeError:
        print('To see a gif of the data distribution, make sure you '
              'have ffmpeg installed!')
        return None

    if self.log_dr is not None:

        save_name = os.path.join(self.exp_log_dr,
                                 save_name.replace(' ', '_') + '.gif')

        try:
            anim.save(save_name, dpi=self.dpi, writer='imagemagick')
        except BrokenPipeError:
            print('Warning: could not save gif, please make sure you have',
                  'imagemagick installed')
        plt.close()

    if self.notebook:
        plt.close()
        return html

    return None


"""
