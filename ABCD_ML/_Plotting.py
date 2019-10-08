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
from IPython.display import display, HTML
from matplotlib.animation import FuncAnimation
from ABCD_ML.Data_Helpers import get_original_cat_names


def _plot(self, save_name, show=True):

    if show:
        if self.log_dr is not None:

            save_spot = os.path.join(self.exp_log_dr,
                                     save_name.replace(' ', '_') + '.png')
            plt.savefig(save_spot, dpi=100, bbox_inches='tight')

        if self.notebook:
            plt.show()


def Show_Data_Dist(self, num_feats=20, frame_interval=500,
                   plot_type='bar', save=True,
                   save_name='data distribution'):

    '''This method displays some summary statistics about
    the loaded targets, as well as plots the distibution if possible.

    Parameters
    ----------
    num_feats: int, optional
        The number of random features's distributions in which to view.
        Note: If too many are selected it may take a long time to render
        and/or consume a lot of memory!

        (default = 20)

    frame_interval: int, optional
        The number of milliseconds between each frame.

        (default = 500)

    plot_type : {'bar', 'hist'}
        The type of base seaborn plot to generate for each datapoint.
        Either 'bar' for barplot, or 'hist' or seaborns dist plot.

        (default = 'bar')

    save : bool, optional
        If the animation should be saved as a gif, True or False.

        (default = True)

    save_name : str, optional
        The name in which the gif should be saved under.

        (default = 'data distribution')
    '''

    fig, ax = plt.subplots()

    def update(i):
        fig.clear()

        col = self.data[list(self.data)[i]]
        non_nan_col = col[~pd.isnull(col)]

        if plot_type == 'hist':
            sns.distplot(non_nan_col)
        else:
            sns.boxplot(non_nan_col)

    np.random.seed(1)
    frames = np.random.randint(0, self.data.shape[1], size=num_feats)
    anim = FuncAnimation(fig, update, frames=frames, interval=500)
    html = HTML(anim.to_html5_video())

    if self.log_dr is not None:

        save_name = os.path.join(self.exp_log_dr,
                                 save_name.replace(' ', '_') + '.gif')
        anim.save(save_name, dpi=80, writer='imagemagick')
        plt.close()

    if self.notebook:
        return html

    return None


def Show_Targets_Dist(self, targets='SHOW_ALL', cat_show_original_name=True,
                      show_only_overlap=True, show=True):
    '''This method displays some summary statistics about
    the loaded targets, as well as plots the distibution if possible.

    Parameters
    ----------
    targets : str or list, optional
        The single (str) or multiple targets (list),
        in which to display the distributions of. The str input
        'SHOW_ALL' is reserved, and set to default, for showing
        the distributions of loaded targets.

        (default = 'SHOW_ALL')

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

    show : bool, optional
        If True, then plt.show(), the matplotlib command will be called,
        and the figure displayed. On the other hand, if set to False,
        then the user can customize the plot as they desire.
        You can think of plt.show() as clearing all of the loaded settings,
        so in order to make changes, you can't call this until you are done.

        (default = True)
    '''

    if show_only_overlap:
        overlap = self._get_overlapping_subjects()
        targets_df = self.targets[self.targets.index.isin(overlap)].copy()
    else:
        targets_df = self.targets.copy()

    if targets == 'SHOW_ALL':
        targets = self._get_base_targets_names()

    if not isinstance(targets, list):
        targets = [targets]

    for target in targets:
        self._show_single_dist(target, targets_df, self.targets_encoders,
                               cat_show_original_name, show)
        self._print()


def Show_Covars_Dist(self, covars='SHOW_ALL', cat_show_original_name=True,
                     show_only_overlap=True, show=True):
    '''Plot a single or multiple covar distributions, along with
    outputting useful summary statistics.

    Parameters
    ----------
    covars : str or list, optional
        The single covar (str) or multiple covars (list),
        in which to display the distributions of. The str input
        'SHOW_ALL' is reserved, and set to default, for showing
        the distributions of all avaliable covars.

        (default = 'SHOW_ALL')

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

    show : bool, optional
        If True, then plt.show(), the matplotlib command will be called,
        and the figure displayed. On the other hand, if set to False,
        then the user can customize the plot as they desire.
        You can think of plt.show() as clearing all of the loaded settings,
        so in order to make changes, you can't call this until you are done.

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
        self._show_single_dist(covar, covars_df, self.covars_encoders,
                               cat_show_original_name, show)
        self._print()


def _show_single_dist(self, name, df, all_encoders, cat_show_original_name,
                      show=True):

    # Binary or categorical
    if name in all_encoders:

        dropped_name = None
        encoders = all_encoders[name]

        if isinstance(encoders, tuple):
            categories = encoders[1].categories_[0]
            categories = sorted(categories)

            df_names = [name + '_' + str(cat) for cat in categories]
            valid_df_names = [name for name in df_names if name in df]

            single_df = df[valid_df_names].copy()

            # Recover dropped column if dummy coded
            dropped_name = set(df_names).difference(set(valid_df_names))

            if len(dropped_name) > 0:
                dropped_name = list(dropped_name)[0]

                single_df[dropped_name] =\
                    np.where(single_df.sum(axis=1) == 1, 0, 1)

                single_df[dropped_name] =\
                    single_df[dropped_name].astype('category')

            # Back into order
            single_df = single_df[df_names]

        else:
            single_df = df[[name]].copy()

        self._show_dist(single_df, name, cat_show_original_name,
                        encoders=encoders, original_key=name,
                        dropped_name=dropped_name, show=show)

    # Regression
    elif name in df:

        single_df = df[[name]].copy()
        self._show_dist(single_df, plot_key=name,
                        cat_show_original_name=cat_show_original_name,
                        original_key=name, show=show)

    else:
        self._print('No col named', name, 'found!')


def _show_dist(self, data, plot_key, cat_show_original_name, encoders=None,
               original_key=None, dropped_name=None, show=True):

    # Ensure works with NaN data loaded
    no_nan_subjects = data[~data.isna().any(axis=1)].index
    nan_subjects = data[data.isna().any(axis=1)].index
    no_nan_data = data.loc[no_nan_subjects]

    self._print('--', plot_key, '--')

    # Binary or categorical
    if no_nan_data.dtypes[0].name == 'category':

        # Categorical
        if isinstance(encoders, tuple):
            encoder = encoders[0]
            sums = no_nan_data.sum()

        # Binary
        else:
            encoder = encoders
            unique, counts = np.unique(no_nan_data, return_counts=True)
            sums = pd.Series(counts, unique)

        original_names = get_original_cat_names(sums.index,
                                                encoder,
                                                original_key)

        display_df = pd.DataFrame(sums, columns=['Counts'])
        display_df.index.name = 'Internal Name'
        display_df['Original Name'] = original_names
        display_df['Frequency'] = sums / len(no_nan_data)
        display_df = display_df[['Original Name', 'Counts', 'Frequency']]

        self._display_df(display_df)

        if dropped_name is not None:
            if len(dropped_name) > 0:
                self._print('Note:', dropped_name, 'was dropped due to dummy',
                            'coding but is still shown.')

        display_names = sums.index
        if cat_show_original_name:
            display_names = pd.Index(original_names)
            display_names.name = 'Original Name'

        sns.barplot(x=sums.values, y=display_names, orient='h')
        plt.xlabel('Counts')

    # Regression, float / ordinal
    else:

        summary = no_nan_data.describe()

        self._display_df(summary)

        vals = no_nan_data[original_key]
        self._print('Num. of unique vals:', len(np.unique(vals)))
        self._print()

        sns.distplot(no_nan_data)

    # If any NaN
    if len(nan_subjects) > 0:
        self._print('Note:', len(nan_subjects), 'subject(s) with NaN',
                    'not included/shown!')

    title = plot_key + ' distributions'
    plt.title(title)
    self._plot(title, show)


def _display_df(self, display_df):

    if self.notebook:
        display(display_df)

    self._print(display_df, dont_print=self.notebook)
    self._print(dont_print=self.notebook)


def Plot_Base_Feat_Importances(self, top_n=10,
                               title='Base Feature Importances', show=True):
    '''Plots the base feature importances as calculated from the
    last run :func:`Evaluate` or :func:`Test`.
    See :func:`Get_Base_Feat_Importances` for more details.

    Parameters
    ----------

    top_n : int or None, optional
        If not None, then will only return the top_n
        number of features, by shap feature importance.

        (default = 10)

    title : str, optional
        The title used during plotting, and also used
        to save a version of the figure
        (with spaces in title replaced by _, and as a png).

        (default = 'Base Feature Importances')

    show : bool, optional
        If True, then plt.show(), the matplotlib command will be called,
        and the figure displayed. On the other hand, if set to False,
        then the user can customize the plot as they desire.
        You can think of plt.show() as clearing all of the loaded settings,
        so in order to make changes, you can't call this until you are done.

        (default = True)
    '''

    top_x = self.Get_Base_Feat_Importances(top_n).index
    just_top = self.Model_Pipeline.feature_importances[top_x]

    self._plot_feature_importance(just_top, title=title,
                                  xlabel='Feature Importance',
                                  show=show)


def Plot_Shap_Feat_Importances(self, top_n=10,
                               title='Shap Feature Importances', show=True):
    '''Plots the shap feature importances as calculated from the
    last run :func:`Evaluate` or :func:`Test`.
    See :func:`Get_Shap_Feat_Importances` for more details.

    Parameters
    ----------

    top_n : int or None, optional
        If not None, then will only return the top_n
        number of features, by shap feature importance.

        (default = 10)

    title : str, optional
        The title used during plotting, and also used
        to save a version of the figure
        (with spaces in title replaced by _, and as a png).

        (default = 'Shap Feature Importances')

    show : bool, optional
        If True, then plt.show(), the matplotlib command will be called,
        and the figure displayed. On the other hand, if set to False,
        then the user can customize the plot as they desire.
        You can think of plt.show() as clearing all of the loaded settings,
        so in order to make changes, you can't call this until you are done.

        (default = True)
    '''

    top_x = self.Get_Shap_Feat_Importances(top_n).index

    # For categorical support
    try:
        just_top = np.abs(self.avg_shap_df)[top_x]
    except AttributeError:
        just_top = np.abs(self.Model_Pipeline.shap_df)[top_x]

    self._plot_feature_importance(just_top, title=title,
                                  xlabel='Shap Feature Importance',
                                  show=show)


def _plot_feature_importance(self, data, title, xlabel, show):

    sns.barplot(orient='h', data=data)
    plt.title(title)
    plt.xlabel(xlabel)
    self._plot(title, show)


def Plot_Shap_Summary(self, top_n=10, title=None, cat_show_original_name=True,
                      show=True):

    assert len(self.Model_Pipeline.shap_df) > 0, \
        "calc_shap_feature_importances must be set to True!"

    # For grabbing right cols and subjects
    # And plotting correct values
    if 'pandas' not in str(type(self.Model_Pipeline.shap_df)):
        shap_df = self.Model_Pipeline.shap_df[0]
        shap_df_arrays = [np.array(df) for df in self.Model_Pipeline.shap_df]

        targets_key = self.Model_Pipeline.targets_key

        if isinstance(targets_key, list):
            base_target_key = '_'.join(targets_key[0].split('_')[:-1])
        else:
            base_target_key = targets_key

        if cat_show_original_name:
            class_names =\
                get_original_cat_names(targets_key,
                                       self.Model_Pipeline.targets_encoder[0],
                                       base_target_key)
        else:
            class_names = targets_key

    else:
        shap_df = self.Model_Pipeline.shap_df
        shap_df_arrays = np.array(self.Model_Pipeline.shap_df)
        class_names = None

    shap_cols, shap_inds = list(shap_df), shap_df.index
    X = self.all_data.loc[shap_inds][shap_cols]

    shap.summary_plot(shap_df_arrays, X, max_display=top_n, show=False,
                      class_names=class_names)

    if title is None:
        title = 'Shap Feat Importance Summary Plot'

    plt.title(title)
    self._plot(title, show)
