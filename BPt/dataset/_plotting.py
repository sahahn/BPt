import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def show(self, scope):
    pass


def show_nan_info(self, scope):
    pass


def info(self, scope):
    pass


def plot_vars(self, scope):
    pass


def plot(self, scope,
         subjects='all',
         show=True,
         cut=0,
         encoded_values=True,
         count=True):
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

    '''

    # Grab cols to plot
    cols = self.get_cols(scope)

    # Grab subjs to plot
    subjs = self.get_subjects(subjects, return_as='flat index')

    for col in cols:
        if 'category' in self.scopes[col]:
            self._plot_category(col=col,
                                subjs=subjs,
                                encoded_values=encoded_values,
                                count=count,
                                show=show)

        else:
            self._plot_float(col=col, subjs=subjs,
                             cut=cut, show=show)


def _plot_float(self, col, subjs, cut, show):

    # Get values to plot
    values = self._get_plot_values(col, subjs, encoded_values=False)

    # Plot values
    sns.kdeplot(values, cut=cut)
    plt.xlabel('Values')
    plt.title(str(col) + ' Distribution')

    if show:
        plt.show()


def _get_plot_values(self, col, subjs, encoded_values):

    # Check for encoded values
    if encoded_values:
        values = self.get_encoded_values(col)
    else:
        values = self.get_values(col)

    # Get subset of subjects
    overlap_subjs = subjs.intersection(values.index)
    values = values.loc[overlap_subjs]

    # Print info
    self._print_plot_info(col, values, subjs)

    return values


def _plot_category(self, col, subjs, encoded_values, count, show):

    # Get plot values
    values = self._get_plot_values(col, subjs, encoded_values)

    # Don't plot if more than 50 categories
    if len(values.unique()) >= 50:
        self._print('Skipping plot: ' + str(col) +
                    ' as >= categories!', level=0)
        return self

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

    return self


def _print_plot_info(self, col, values, subjs):

    self._print(col, level=1)
    self._print('Found', len(values), 'valid subjects.', level=1)

    n_nan_subjects = len(subjs) - len(values)
    if n_nan_subjects > 0:
        self._print('Skipping', n_nan_subjects, 'NaN values.', level=1)

    self._print(level=1)

    return self


def plot_bivar(self, col1, col2, subjects='all',
               encoded_values=True, show=True):

    self._check_scopes()

    # Grab subjs to plot
    subjs = self.get_subjects(subjects, return_as='flat index')

    # Cat
    if 'category' in self.scopes[col1]:

        # Cat x Cat
        if 'category' in self.scopes[col2]:
            self._plot_cat_cat(col1, col2, subjs=subjs,
                               encoded_values=encoded_values)

        # Cat x Float
        else:
            self._plot_cat_float(cat_col=col1,
                                 float_col=col2,
                                 subjs=subjs,
                                 encoded_values=encoded_values)

    # Float
    else:

        # Float x Cat
        if 'category' in self.scopes[col2]:
            self._plot_cat_float(cat_col=col2,
                                 float_col=col1,
                                 subjs=subjs,
                                 encoded_values=encoded_values)

        # Float x Float
        else:
            self._plot_float_float(col1, col2, subjs=subjs)

    if show:
        plt.show()


def _plot_cat_cat(self, col1, col2, subjs, encoded_values):

    values1 = self._get_plot_values(col1, subjs, encoded_values)
    values2 = self._get_plot_values(col2, subjs, encoded_values)

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

    return self


def _plot_float_float(self, col1, col2, subjs):

    values1 = self._get_plot_values(col1, subjs, encoded_values=False)
    values2 = self._get_plot_values(col2, subjs, encoded_values=False)

    # Get overlap
    overlap = values1.index.intersection(values2.index)
    self._print('Plotting', len(overlap), 'overlap valid subjects.', level=1)

    values1 = values1.loc[overlap]
    values2 = values2.loc[overlap]

    sns.jointplot(x=values1, y=values2, kind="reg")

    return self


def _plot_cat_float(self, cat_col, float_col, subjs, encoded_values):

    cat_values = self._get_plot_values(cat_col, subjs,
                                       encoded_values=encoded_values)
    float_values = self._get_plot_values(float_col, subjs,
                                         encoded_values=False)

    # Get overlap
    overlap = cat_values.index.intersection(float_values.index)
    self._print('Plotting', len(overlap), 'overlap valid subjects.', level=1)

    sns.displot(x=float_values.loc[overlap], hue=cat_values.loc[overlap],
                kind="kde")

    return self
