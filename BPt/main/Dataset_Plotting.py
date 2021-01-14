import matplotlib.pyplot as plt
import seaborn as sns
import warnings


def show(self, scope):
    pass


def info(self, scope):
    pass


def plot_vars(self, scope):
    pass


def plot(self, scope, subjects=None, encoded_values=True,
         cat_types='Counts'):

    # Grab cols to plot
    cols = self.get_cols(scope)

    for col in cols:
        if 'category' in self.scopes[col]:
            if len(self[col].unique()) < 50:
                self._plot_category(col=col, encoded_values=encoded_values)
            else:
                warnings.warn('Skipping plot: ' + str(col) +
                              ' too many categories!')
        else:
            pass


def _plot_category(self, col, encoded_values=True):

    # Check for show_original_names
    if encoded_values:
        values = self.get_encoded_values(col)
    else:
        values = self[col].copy()

    # Plot
    sns.countplot(values, orient='h')
