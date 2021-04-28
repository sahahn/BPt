
import numpy as np
import matplotlib.pyplot as plt


def _get_top_global(self, df, top_n, get_abs):

    if get_abs:
        imps = np.mean(np.abs(df))
    else:
        imps = np.mean(df)

    imps.sort_values(ascending=False, inplace=True)

    to_get = list(range(top_n))
    if not get_abs:
        to_get += [-i for i in range(1, top_n+1)][::-1]

    top = imps[to_get]
    feats = imps[to_get].index

    top_df = df[feats]

    if get_abs:
        top_df = top_df.abs()

    wide = top_df.melt()
    wide['mean'] = wide['variable'].copy()
    wide['mean'] = wide['mean'].replace(top.to_dict())

    return wide


def Plot_Global_Feat_Importances(
 self, feat_importances='most recent', top_n=10, show_abs=False,
 multiclass=False, ci=95, palette='default', figsize=(10, 10),
 title='default', titles='default', xlabel='default', n_cols=1, ax=None,
 show=True):
    '''Plots any global feature importance, e.g. base or shap, values per
    feature not per prediction.

    Parameters
    ----------
    feat_importances : 'most recent' or Feat_Importances object
        Input should be either a Feat_Importances object as output from a
        call to Evaluate, or Test, or if left as default 'most recent',
        the passed params will be used to plot any valid calculated feature
        importances from the last call to Evaluate or Test.

        Note, if there exist multiple valid feature importances in the
        last call, passing custom ax will most likely break things.

        (default = 'most recent')

    top_n : int, optional
        The number of top features to display. In the case where
        show_abs is set to True, or the feature importance being plotted
        is only positive, then top_n features will be shown. On the other
        hand, when show_abs is set to False and the feature importances being
        plotted contain negative numbers, then the top_n highest and top_n
        lowest features will be shown.

        (default = 10)

    show_abs : bool, optional
        In the case where the underlying feature importances contain
        negative numbers, you can either plot the top_n by absolute value,
        with show_abs set to True, or plot the top_n highest and lowest
        with show_abs set to False.

        (default = False)

    multiclass : bool, optional
        If multiclass is set to True, and the underlying feature importances
        were derived from a categorical problem type, then
        a seperate feature importance plot will be made for each class.
        Alternatively, if multiclass is set to False, then feature importances
        will be averaged over all classes.

        (default = False)

    ci : float, 'sd' or None, optional
        Size of confidence intervals to draw around estimated values.
        If 'sd', skip bootstrapping and draw the standard deviation
        of the feat importances. If None, no bootstrapping will be performed,
        and error bars will not be drawn.

        (default = 95)

    palette : Seaborn palette name, optional
        Color scheme to use. Search seaborn palettes for more information.
        Default for absolute is 'Reds', and default for both pos
        and neg is 'coolwarm'.

        (default = 'default')

    title : str, optional
        The title used during plotting, and also used
        to save a version of the figure
        (with spaces in title replaced by _, and as a png).

        When multiclass is True, this is the full figure title.

        (default = 'default')

    titles: list, optional
        This parameter is only used when multiclass is True.
        titles should be a list with the name for each classes plot.
        If left as default, it will just be named the original loaded name
        for that class.

        (default = 'default')

    xlabel : str, optional
        The xlabel, descriping the measure of feature importance.
        If left as 'default' it will change depend on what feature importance
        is being plotted.

        (default = 'default')

    n_cols: int, optional
        If multiclass, then the number of class plots to
        plot on each row.

        (default = 1)

    ax: matplotlib axis, or axes, optional
        A custom ax to plot to for an individual plot, or if using
        multiclass, then a list of axes can be passed here.

        (default = None)

    show : bool, optional
        If True, then plt.show(), the matplotlib command will be called,
        and the figure displayed. On the other hand, if set to False,
        then the user can customize the plot as they desire.
        You can think of plt.show() as clearing all of the loaded settings,
        so in order to make changes, you can't call this until you are done.

        (default = True)
    '''

    if feat_importances == 'most recent':
        for fis in self.evaluator.feat_importances:
            if 'global' in fis.scopes:
                self.Plot_Global_Feat_Importances(
                 fis, top_n, show_abs, multiclass, ci, palette, figsize, title,
                 titles, xlabel, n_cols, ax, show)

        return

    # Initial check to make sure valid feat importances passed
    if feat_importances.global_df is None:
        raise AttributeError('You must pass a feature importances object with',
                             'global_df, try plot local feat importance.')

    # If multiclass
    if isinstance(feat_importances.global_df, list):
        if multiclass:

            dfs = feat_importances.global_df

            return self._plot_multiclass_global_feat_importances(
                        dfs, feat_importances, top_n=top_n, show_abs=show_abs,
                        ci=ci, palette=palette, figsize=figsize, title=title,
                        titles=titles, xlabel=xlabel, n_cols=n_cols, ax=ax,
                        show=show)

        else:
            df = pd.concat(feat_importances.global_df)
    else:
        df = feat_importances.global_df

    # Non-multi-class case
    ax = self._plot_global_feat_importances(df, feat_importances,
                                            top_n=top_n, show_abs=show_abs,
                                            ci=ci, palette=palette,
                                            title=title, xlabel=xlabel,
                                            ax=ax, show=show)
    return ax


def _plot_multiclass_global_feat_importances(self, dfs, feat_importances,
                                             top_n=10, show_abs=False,
                                             ci=95, palette='default',
                                             figsize=(10, 10), title='default',
                                             titles='default',
                                             xlabel='default', n_cols=2,
                                             ax=None, show=True):

    # Grab the right classes
    target = feat_importances.target
    target_key = self._get_targets_key(target)
    classes = self.targets_encoders[target_key].classes_

    # Set default titles
    if titles == 'default':
        titles = classes

    if title == 'default':
        name = feat_importances.get_name()
        title = name + ' Top Features by Class'

    # If an ax is passed, then use that instead of making subplots
    n_rows = math.ceil(len(classes) / n_cols)
    if ax is not None:
        axes = ax
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # In the case where only 1 row or 1 col.
    single = False
    if n_rows == 1 or n_cols == 1:
        single = True

    for i in range(len(dfs)):

        row = i // n_cols
        col = i % n_cols
        rc = (row, col)

        if single:
            rc = i

        # Create the subplot
        axes[rc] = self._plot_global_feat_importances(
            dfs[i], feat_importances, top_n=top_n,
            show_abs=show_abs, ci=ci, palette=palette,
            title=titles[i], xlabel='default', ax=axes[rc])

    # Set any remaining subplots to nothing
    for i in range(len(dfs), n_rows * n_cols):

        row = i // n_cols
        col = i % n_cols
        rc = (row, col)

        if single:
            rc = i

        axes[rc].set_axis_off()

    # Set the title and layout if no custom ax passed
    if ax is None:
        fig.tight_layout()
        fig.subplots_adjust(top=.9, hspace=.25)
        fig.suptitle(title, fontsize="x-large")
        self._plot(title, show)

    else:
        return axes


def _plot_global_feat_importances(self, df, feat_importances, top_n=10,
                                  show_abs=False, ci=95,
                                  palette='default', title='default',
                                  xlabel='default', ax=None, show=True):

    # Distinguish between feat importances that have been set to abs
    min_val = np.min(np.min(df))
    add_abs_sign = True

    if min_val >= 0:
        show_abs = True
    else:
        add_abs_sign = False

    # Get df, with only top_n, in wide form for plotting
    top_df = self._get_top_global(df, top_n, show_abs)

    # Abs / all pos values
    if show_abs:
        if palette == 'default':
            palette = 'Reds'

        if title == 'default':
            name = feat_importances.get_name()
            title = name + ' Top ' + str(top_n) + ' Features'

        if xlabel == 'default':
            xlabel = feat_importances.get_global_label()

        sns.barplot(x='value', y='variable', hue='mean',
                    data=top_df, orient='h', ci=ci,
                    palette=palette, errwidth=2, ax=ax, dodge=False)

    # Pos + neg values
    else:
        if palette == 'default':
            palette = 'coolwarm'

        if title == 'default':
            name = feat_importances.get_name()
            title = name + ' Top ' + str(top_n) + ' +/- Features'

        if xlabel == 'default':
            xlabel = feat_importances.get_global_label()

            if add_abs_sign:
                xlabel = '|' + xlabel + '|'

        sns.barplot(x='value', y='variable', hue='mean',
                    data=top_df, orient='h', ci=ci,
                    palette=palette, errwidth=2, ax=ax,
                    dodge=False)

        # Add an extra line down the center
        if ax is None:
            current_ax = plt.gca()
            current_ax.axvline(0, color='k', lw=.2)
            sns.despine(ax=current_ax, top=True, left=True, bottom=True,
                        right=True)
        else:
            ax.axvline(0, color='k', lw=.2)
            sns.despine(ax=ax, top=True, left=True, bottom=True, right=True)

    # Process rest fo params based on if for axis or global
    if ax is None:
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('')
        plt.legend().remove()
        self._plot(title, show)

    else:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('')
        ax.get_legend().remove()
        return ax
