import matplotlib.pyplot as plt
import seaborn as sns
import warnings


def show(self, scope):
    pass


def info(self, scope):
    pass


def plot_vars(self, scope):
    pass


def plot(self, scope, subjects=None, original_values=True,
         cat_types='Counts'):

    # Run check name map
    self._check_encoders()

    # Grab cols to plot
    cols = self._get_cols_from_scope(scope)

    for col in cols:
        if 'category' in self.scopes[col]:
            if len(self[col].unique()) < 50:
                self._plot_category(col=col, original_values=original_values)
            else:
                warnings.warn('Skipping plot: ' + str(col) +
                              ' too many categories!')
        else:
            pass


def _plot_category(self, col, original_values=True):

    # Get a copy of the values
    values = self[col].copy()

    # Check for show_original_names
    if original_values:
        values = self._replace_values(col=col, values=values)

    # Plot
    sns.countplot(values, orient='h')
