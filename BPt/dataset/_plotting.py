import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def show(self, scope):
    pass


def show_nan_info(self, scope):
    pass


def info(self, scope):
    pass


def plot(self, scope,
         subjects='all',
         cut=0,
         decode_values=True,
         count=True,
         show=True):
    '''This function creates plots for each of the passed
    columns (as specified by scope) seperately.

    Parameters
    -----------
    subjects : :ref:`Subjects`, optional
        Optionally restrict the plot to only a subset of
        subjects. This argument can be any of the BPt accepted
        :ref:`Subjects` style inputs.
        E.g., None, 'nan' for subjects
        with any nan data, 'train', the str location of a file
        formatted with one subject per line, or directly an
        array-like of subjects, to name some options.

        See :ref:`Subjects` for all options.

        ::

            default = 'all'

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

        In the case that you are only plotting
        one column, and want to make custimizations,
        then you should pass this value as False,
        and you can make changes to the figure
        via matplotlib's global state system.

        ::

            default = True

    '''

    # Grab cols to plot
    # Get cols will call check scope.
    cols = self.get_cols(scope)

    # Grab subjs to plot
    subjs = self.get_subjects(subjects, return_as='flat index')

    # Plot each column
    for col in cols:
        self._plot_col(col=col, subjs=subjs, cut=cut,
                       decode_values=decode_values,
                       count=count, show=show)


def _plot_col(self, col, subjs, cut,
              decode_values, count, show,
              print_info=True):

    # If categorical
    if 'category' in self.scopes[col]:
        return self._plot_category(col=col,
                                   subjs=subjs,
                                   decode_values=decode_values,
                                   count=count,
                                   show=show,
                                   print_info=print_info)

    # Otherwise float
    return self._plot_float(col=col, subjs=subjs,
                            cut=cut, show=show, print_info=print_info)


def _plot_float(self, col, subjs, cut, show, print_info=True):

    # Get values to plot
    values, info = self._get_plot_values(col, subjs,
                                         decode_values=False,
                                         print_info=print_info)

    # Plot values
    sns.kdeplot(values, cut=cut)
    plt.xlabel('Values')
    plt.title(str(col) + ' Distribution')

    if show:
        plt.show()

    return info


def _plot_category(self, col, subjs, decode_values, count,
                   show, print_info=True):

    # Get plot values
    values, info = self._get_plot_values(col, subjs,
                                         decode_values,
                                         print_info=print_info)

    # Don't plot if more than 50 categories
    if len(values.unique()) >= 50:
        self._print('Skipping plot: ' + str(col) +
                    ' as >= categories!', level=0)
        return

    # Get counts
    counts = pd.DataFrame(values.value_counts(normalize=not count))

    # Reset index
    counts = counts.reset_index()

    # Plot
    sns.catplot(x=col, y='index', data=counts,
                kind='bar', orient='h', ci=None)

    if count:
        plt.xlabel('Counts')
    else:
        plt.xlabel('Frequency')

    plt.ylabel('Categories')
    plt.title(str(col) + ' Distribution')

    # If show, call show
    if show:
        plt.show()

    return info


def _get_plot_values(self, col, subjs,
                     decode_values, print_info=True):

    # Get values
    values = self.get_values(col, dropna=True,
                             decode_values=decode_values)

    # Get subset of subjects
    overlap_subjs = subjs.intersection(values.index)
    values = values.loc[overlap_subjs]

    # Compute info
    info = {'n': len(values),
            'n_nan': len(subjs) - len(values)}

    # Print info only if requested
    if print_info:
        self._print_plot_info(col, info)

    return values, info


def _print_plot_info(self, col, info):

    self._print(str(col) + ':', str(info['n']), 'rows', end='', level=1)

    if info['n_nan'] > 0:
        self._print(' (' + str(info['n_nan']), 'NaN)',
                    end='', level=1)

    self._print(level=1)


def plot_bivar(self, col1, col2, subjects='all',
               decode_values=True, show=True):

    self._check_scopes()

    # Grab subjs to plot
    subjs = self.get_subjects(subjects, return_as='flat index')

    # Cat
    if 'category' in self.scopes[col1]:

        # Cat x Cat
        if 'category' in self.scopes[col2]:
            self._plot_cat_cat(col1, col2, subjs=subjs,
                               decode_values=decode_values)

        # Cat x Float
        else:
            self._plot_cat_float(cat_col=col1,
                                 float_col=col2,
                                 subjs=subjs,
                                 decode_values=decode_values)

    # Float
    else:

        # Float x Cat
        if 'category' in self.scopes[col2]:
            self._plot_cat_float(cat_col=col2,
                                 float_col=col1,
                                 subjs=subjs,
                                 decode_values=decode_values)

        # Float x Float
        else:
            self._plot_float_float(col1, col2, subjs=subjs)

    if show:
        plt.show()


def _plot_cat_cat(self, col1, col2, subjs, decode_values):

    values1, _ = self._get_plot_values(col1, subjs, decode_values)
    values2, _ = self._get_plot_values(col2, subjs, decode_values)

    # Get overlap
    overlap = values1.index.intersection(values2.index)
    self._print('Plotting', len(overlap), 'overlap valid subjects.', level=1)

    values1 = values1.loc[overlap]
    values2 = values2.loc[overlap]

    # Convert to df
    df = values1.to_frame()
    df[col2] = values2

    # Get counts
    ct_counts = df.groupby([col1, col2]).size()
    ct_counts = ct_counts.reset_index(name='Count')
    ct_counts = ct_counts.pivot(index=col1, columns=col2, values='Count')

    sns.heatmap(ct_counts, annot=True, cmap="YlGnBu")


def _plot_float_float(self, col1, col2, subjs):

    values1, _ = self._get_plot_values(col1, subjs, decode_values=False)
    values2, _ = self._get_plot_values(col2, subjs, decode_values=False)

    # Get overlap
    overlap = values1.index.intersection(values2.index)
    self._print('Plotting', len(overlap), 'overlap valid subjects', level=1)

    values1 = values1.loc[overlap]
    values2 = values2.loc[overlap]

    sns.jointplot(x=values1, y=values2, kind="reg")


def _plot_cat_float(self, cat_col, float_col, subjs, decode_values):

    cat_values, _ = self._get_plot_values(cat_col, subjs,
                                          decode_values=decode_values)
    float_values, _ = self._get_plot_values(float_col, subjs,
                                            decode_values=False)

    # Get overlap
    overlap = cat_values.index.intersection(float_values.index)
    self._print('Plotting', len(overlap), 'overlap valid subjects.', level=1)

    sns.displot(x=float_values.loc[overlap], hue=cat_values.loc[overlap],
                kind="kde")
