"""
_Plotting.py
====================================
Main class extension file for the some plotting functionality.
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import shap
import os
from IPython.display import display
from ABCD_ML.Data_Helpers import get_original_cat_names


def plot(self, title):

    if self.log_dr is not None:

        save_spot = os.path.join(self.exp_log_dr,
                                 title.replace(' ', '_') + '.png')
        plt.savefig(save_spot, dpi=100)

    if self.notebook:
        plt.show()


def Show_Targets_Dist(self, cat_show_original_name=True,
                      show_only_overlap=True):
    '''This method displays some summary statistics about
    the loaded targets, as well as plots the distibution if possible.

    Parameters
    ----------
    cat_show_original_name : bool, optional
        If True, then when showing a categorical distribution (or binary)
        make the distr plot using the original names. Otherwise,
        use the internally used names.

        (default = True)

    show_only_overlap : bool, optional
        If True, then displays only the distributions for valid overlapping
        subjects across data, covars, ect... otherwise, shows the current
        loaded distribution as is.

        (default = True)
    '''

    if show_only_overlap:
        overlap = self._get_overlapping_subjects()
        targets_df = self.targets[self.targets.index.isin(overlap)].copy()
    else:
        targets_df = self.targets.copy()

    self.show_dist(targets_df, 'Target', cat_show_original_name,
                   encoders=self.targets_encoder,
                   original_key=self.original_targets_key)


def Show_Covars_Dist(self, covars='SHOW_ALL', cat_show_original_name=True,
                     show_only_overlap=True):
    '''Plot a single or multiple covar distributions, along with
    outputting useful summary statistics.

    Parameters
    ----------
    covars : str or list, optional
        The single covar (str) or multiple covars (list),
        in which to display the distributions of. The str input
        'SHOW_ALL' is reserved, and set to default, for showing
        the distributions of all avaliable covars.

    cat_show_original_name : bool, optional
        If True, then when showing a categorical distribution (or binary)
        make the distr plot using the original names. Otherwise,
        use the internally used names.

        (default = True)

    show_only_overlap : bool, optional
        If True, then displays only the distributions for valid overlapping
        subjects across data, covars, ect... otherwise, shows the current
        loaded distribution as is.

        (default = True)
    '''

    if show_only_overlap:
        overlap = self._get_overlapping_subjects()
        covars_df = self.covars[self.covars.index.isin(overlap)].copy()
    else:
        covars_df = self.covars.copy()

    if covars == 'SHOW_ALL':
        covars = self._get_base_covar_names()

    if not isinstance(covars, list):
        covars = [covars]

    for covar in covars:
        self.show_covar_dist(covar, covars_df, cat_show_original_name)


def show_covar_dist(self, covar, covars_df, cat_show_original_name):

    # Binary or categorical
    if covar in self.covars_encoders:

        cov_encoders = self.covars_encoders[covar]

        if isinstance(cov_encoders, tuple):
            categories = cov_encoders[1].categories_[0]
            categories = sorted(categories)

            covar_df_names = [covar + '_' + str(c) for c in categories]
            valid_df_names = [c for c in covar_df_names if c in covars_df]

            covar_df = covars_df[valid_df_names].copy()

            # Recover dropped column if dummy coded
            dropped_name = set(covar_df_names).difference(set(valid_df_names))
            if len(dropped_name) > 0:

                dropped_name = list(dropped_name)[0]
                covar_df[dropped_name] =\
                    np.where(covar_df.sum(axis=1) == 1, 0, 1)

            covar_df = covar_df[covar_df_names]

        else:
            covar_df = covars_df[[covar]].copy()

        self.show_dist(covar_df, covar, cat_show_original_name,
                       encoders=cov_encoders, original_key=covar)

    # Regression
    elif covar in covars_df:

        covar_df = covars_df[[covar]].copy()
        self.show_dist(covar_df, covar, cat_show_original_name)

    else:
        self._print('No covar named', covar, 'found!')


def show_dist(self, data, plot_key, cat_show_original_name, encoders=None,
              original_key=None):

    self._print('Show', plot_key, 'distribution:')

    # Binary or categorical
    if data.dtypes[0].name == 'category':

        # Categorical
        if isinstance(encoders, tuple):
            encoder = encoders[0]
            sums = data.sum()

        # Binary
        else:
            encoder = encoders
            unique, counts = np.unique(data, return_counts=True)
            sums = pd.Series(counts, unique)

        original_names = get_original_cat_names(sums.index,
                                                encoder,
                                                original_key)

        display_df = pd.DataFrame(sums, columns=['Count'])
        display_df.index.name = 'Internal Name'
        display_df['Original Name'] = original_names
        display_df['Frequency'] = sums / len(data)
        display_df = display_df[['Original Name', 'Count', 'Frequency']]

        if self.notebook:
            display(display_df)

        self._print(display_df, dont_print=self.notebook)
        self._print(dont_print=self.notebook)

        display_names = sums.index
        if cat_show_original_name:
            display_names = original_names

        sns.barplot(x=sums.values, y=display_names, orient='h')

    # Regression, float / ordinal
    else:

        summary = data.describe()

        if self.notebook:
            display(summary)

        self._print(summary, dont_print=self.notebook)
        self._print(dont_print=self.notebook)

        vals = data[plot_key]
        self._print('Num. of unique vals:', len(np.unique(vals)))
        self._print()

        sns.distplot(data)

    title = plot_key + ' distributions'
    plt.title(title)
    self.plot(title)


def Plot_Base_Feat_Importances(self, top_n=10):

    top_x = self.Get_Base_Feat_Importances(top_n).index
    just_top = self.Model.feature_importances[top_x]

    self._plot_feature_importance(just_top, title='Base Feature Importances',
                                  xlabel='Feature Importance')


def Plot_Shap_Feat_Importances(self, top_n=10):

    top_x = self.Get_Shap_Feat_Importances(top_n).index
    just_top = np.abs(self.Model.shap_df)[top_x]

    self._plot_feature_importance(just_top, title='Shap Feature Importances',
                                  xlabel='Shap Feature Importance')


def _plot_feature_importance(self, data, title='Feature Importances',
                             xlabel='Feature Importance'):

    sns.barplot(orient='h', data=data)
    plt.title(title)
    plt.xlabel(xlabel)
    self.plot(title)


def Plot_Shap_Summary(self):

    assert len(self.Model.shap_df) > 0, \
        "calc_shap_feature_importances must be set to True!"

    shap_df = self.Model.shap_df

    shap_cols, shap_inds = list(shap_df), shap_df.index
    X = self.all_data.loc[shap_inds][shap_cols]

    shap.summary_plot(np.array(shap_df), X)
