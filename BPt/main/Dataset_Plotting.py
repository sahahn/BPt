import matplotlib.pyplot as plt
import seaborn as sns

def show(self, scope):
    pass

def info(self, scope):
    pass

def plot_vars(self, scope):
    pass


def plot(self, scope, subjects=None, show_original_names=True,
         cat_types='Counts'):

    # Run check name map
    self._check_name_map()

    # Grab cols to plot
    cols = self._get_cols_from_scope(scope)

    for col in cols:
        if 'category' in self.scopes[col]:
            if len(self[col].unique()) < 50:
                self._plot_category(col, show_original_names)
            else:
                print('Skipping plot:', col, 'too many categories!')

        else:
            pass


def _plot_category(self, col, show_original_names=True):

    # Get a copy of the values
    values = self[col].copy()

    # Check for show_original_names
    if col in self.name_map:
        values = values.replace(self.name_map[col])

    # Plot
    sns.countplot(values, orient='h')
