# coding: utf-8
from argparse import ArgumentParser
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import ConfigReader as cr
from pandas.tools.plotting import table as pd_table


def autolabel(rects, ax):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + rect.get_y(),
                '%d' % int(height),
                ha='center', va='bottom')
    return ax


def table_ax(ax, df):
    the_table = pd_table(ax,df,
                         colLabels=['hyperparameters'],
                         loc='lower right', colWidths=[0.25, 0.25])
    return the_table

def evaluation(dataset, data_dir, plot_dir):
    plt.rcdefaults()

    #Styles
    sns.set_style('whitegrid', {'axes.linewidth':1.25, 'axes.edgecolor':'0.15',
                                'grid.linewidth':1.5, 'grid.color':'gray'})
    sns.set_color_codes()
    plt.rcParams['figure.figsize'] = (12.0, 9.0)
    plt.rc('text', usetex=False)
    plt.rc('font', size=14.0, family='sans-serif')

    # Data location and scenario
    preprocessor='all'

    # Load configurations
    reader = cr.ConfigReader(data_dir=data_dir, dataset=dataset)
    tdf = reader.load_validation_trajectories(preprocessor=preprocessor, load_config=True)

    # Decode number of layers
    tdf.loc[:, ('classifier','num_layers')] = tdf['classifier']['num_layers'].apply(lambda X:ord(X)-ord('a'))

    ## Plot average best architectures
    top5 = tdf.sort_values([('smac','test_performance')]).head(1)
    lays = np.int(np.ceil(np.array(top5['classifier']['num_layers']).mean()))

    labels_list = ['Layer_'+str(i) for i in range(1,7)]
    pre_m = top5['preprocessor']['choice'].describe().top

    activations = []
    n_layers = []
    weights = []
    for i in np.arange(1, lays):
        activations.append(top5['classifier']['activation_layer_'+str(i)].describe().top)
        n_layers.append(top5['classifier']['num_units_layer_'+str(i)].mean())
        weights.append(top5['classifier']['weight_init_'+str(i)].describe().top)

    tab = top5.classifier.T.dropna()
    table_list = ['batch_size', 'dropout_output', 'learning_rate', 'lambda2', 'number_epochs', 'solver']
    t = tab.loc[table_list]
    t = t.append(top5['preprocessor']['choice'])

    a = pd.Series(np.array(n_layers))
    botoms = np.fabs(a.sub(a.max()))/2

    activ_list = ['relu', 'elu', 'leaky', 'sigmoid', 'tanh', 'scaledTanh', 'linear']
    colr_list = sns.xkcd_palette(["windows blue", "pastel blue", "grey blue", "red orange", "emerald", "pine green", "amber"])
    activation_color_codes = dict(zip(activ_list,colr_list))

    bar_width = 0.1
    colors_bars = [activation_color_codes.get(i) for i in activations]
    with sns.axes_style('ticks'):
        fig_arch = plt.figure(1, figsize=(15.,9.))
        ax_arch = plt.subplot(111)
        bars = ax_arch.bar(np.arange(lays-1)-(bar_width/2), a,
                           bottom=botoms, width=bar_width, color=colors_bars)
        sns.despine(left=True)
        ax_arch.set_ylabel('Number of units in Layer')
        ax_arch.set_yticklabels([])
        ax_arch.set_yticks([])
        ax_arch.set_xticks(np.arange(lays-1))
        ax_arch.set_xticklabels(labels_list[:lays-1])
        ax_arch = autolabel(bars, ax_arch)
        table_ax(ax_arch, t)
        ax_arch.legend([b for b in bars], activations, loc='best')
        ax_arch.set_title('Single best architecture found for dataset %s' % dataset)
        ax_arch.set_xlim(-0.5, lays-1)
        fig_arch.savefig(plot_dir + "Best_architecture_on_%s.pdf" % dataset)

    # Start filtering the error
    temp_df = tdf.copy()
    temp_df.columns = tdf.columns.droplevel(0)
    min_perf = temp_df['test_performance'].min()
    mean_perf = temp_df['test_performance'].mean()
    std_perf = temp_df['test_performance'].std()
    qtil_10 = temp_df['test_performance'].quantile(0.1)
    del temp_df

    m = tdf[('smac', 'test_performance')] <= qtil_10

    # Setting values to log scale and categorical values
    log_columns = ['beta1', 'beta2', 'gamma', 'lambda2', 'learning_rate', 'momentum','num_units_layer_1',
                   'num_units_layer_2', 'num_units_layer_3', 'num_units_layer_4', 'num_units_layer_5',
                   'num_units_layer_6', 'power', 'std_layer_1', 'std_layer_2', 'std_layer_3','std_layer_4',
                   'std_layer_5', 'std_layer_6']

    for lc in log_columns:
        try:
            tdf.loc[:, ('classifier', lc)] = np.log10(tdf.loc[:, ('classifier', lc)])
        except KeyError:
            continue

    ## After Setting the frames. Start with the plotting
    plt.clf()

    # Plot the empirical CDF
    sorted_train = (tdf['smac']['train_performance'].sort_values(ascending=True).values)
    sorted_test = (tdf['smac']['test_performance'].sort_values(ascending=True).values)
    ytrain = np.arange(len(sorted_train)) / float(len(sorted_train))
    ytest = np.arange(len(sorted_test)) / float(len(sorted_test))

    plt.step(sorted_train, ytrain, label="Train Performance", lw=2.5)
    plt.step(sorted_test, ytest, label="Test Performance", lw=2.5)
    plt.xlabel("Cross-validation error $y(x)$")
    plt.ylabel(r"Number of Configs (%)")
    plt.xlim(0.0, min(1.0, sorted_test.max()+0.01))
    plt.title("Empirical CDF of configurations based on error")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(plot_dir + 'CDF_Error_%s.pdf' % dataset)

    categories=['solver','lr_policy','num_layers']
    mask_filter = tdf[('smac','test_performance')] <= qtil_10
    filtered = tdf[mask_filter]
    for category in categories:
        fig_f, axs = plt.subplots(ncols=2, nrows=1, figsize=(15.0, 10.5))
        ax0, ax1 = axs.flat
        sns.boxplot(x=('classifier', category), y=('smac','test_performance'), data=filtered.sort_values(by=[('classifier', category)]), ax=ax0)
        ax0.set_xlabel(category)
        ax0.set_ylabel('Test error performance')
        ax0.set_title('Error distribution based on %s' % category)
        sns.countplot(x=('classifier', category), data=filtered.sort_values(by=[('classifier', category)]), ax=ax1)
        ax1.set_xlabel(category)
        ax1.set_ylabel('Times used')
        ax1.set_title('Bar plot of frequency of %s' % category)
        fig_f.suptitle("Descriptive stats of %s on dataset %s using 10%% of configurations" % (category, dataset), y=0.98)
        # fig_f.tight_layout()
        fig_f.savefig(plot_dir + 'Descriptive_plots_over_%s_on_%s.pdf' % (category, dataset))
        fig_f.show()

    ## Plot distro over learning rates
    # Create the grouping of the filtered DF
    classifier_df = tdf[m]['classifier']
    solver_filt = classifier_df.groupby('solver')

    # with sns.color_palette('Set1',8):
        # for name,groups in solver_filt:
            # plt.hist(groups.learning_rate.values, alpha=0.5, bins=20, label=name)
        # plt.legend()

    col_hist = sns.color_palette('Paired',8, desat=0.8)
    rows_to_plot = np.int(np.ceil(len(solver_filt)/2.))
    fig2, axs = plt.subplots(nrows=rows_to_plot, ncols=2, figsize=(12.,17.))
    fig2.suptitle('Distribution of learning rate values for each\
                  solver on dataset %s \n (based on 50%% best configurations)' % dataset, y=1.02)
    for ax, (name, groups) in zip(axs.flat,solver_filt):
        ax.hist(groups.learning_rate.values, bins=5, histtype='bar', fill=True,
                label=name, alpha=0.9, color=col_hist.pop())
        ax.set_xlabel('learning rate values (log scale)')
        ax.set_ylabel('# of Configs')
        ax.legend(loc='best')

    # plt.tight_layout()
    ax = axs.flat[-1]
    ax.set_visible(False)
    fig2.savefig(plot_dir + 'Histogram_of_learning_rate_solver_on_dataset_%s.pdf' % dataset)

    ## Plot over different preprocessing methods
    # Create the grouping of the filtered DF
    prepro_filt = tdf[m].groupby([('preprocessor','choice')])

    prepro_color = sns.color_palette('Paired',14, desat=0.8)
    fig4, axs = plt.subplots(nrows=3, ncols=5, sharex='col', figsize=(22.,12.))
    fig4.suptitle('Distribution of learning rate for each preprocessor on dataset %s'% dataset, y=1.02 )
    for ax, (name, grops) in zip(axs.flat,prepro_filt):
        groups = grops['classifier']
        ax.hist(groups.learning_rate.values, bins=5, histtype='bar', fill=True, label=name,
                color=prepro_color.pop())
        ax.set_xlabel('learning rate values (log scale)')
        ax.set_ylabel('# of Configs')
        ax.legend(loc='best')
    # plt.tight_layout()
    fig4.savefig(plot_dir + 'Histogram_of_learning_rate_prepro_on_dataset_%s.pdf' % dataset)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("experiment_dir")
    parser.add_argument("dataset")
    parser.add_argument("output_path")
    args = parser.parse_args()
    evaluation(args.dataset, args.experiment_dir, args.output_path)

