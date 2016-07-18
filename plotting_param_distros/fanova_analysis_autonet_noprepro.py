# coding: utf-8
from argparse import ArgumentParser
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pyfanova.fanova
import pyfanova.visualizer


def parameter_plotting(dataset, data_dir, plot_dir, pairwise=False):
    plt.rcdefaults()
    # IMO very sloppy way to do it
    clear_name = lambda X: X.split(':')[-1] if X.split(':')[0] == 'classifier' else X.split(':')[0]

    #Styles
    sns.set_style('whitegrid', {'axes.linewidth':1.25, 'axes.edgecolor':'0.15',
                            'grid.linewidth':1.5, 'grid.color':'gray'})
    sns.set_color_codes()
    plt.rcParams['figure.figsize'] = (12.0, 9.0)
    plt.rc('text', usetex=False)
    plt.rc('font', size=13.0, family='serif')

    preprocessor='NoPreprocessing'

    ## Parameter importance table
    state_run_dir = os.path.join(data_dir, dataset, preprocessor, 'merged_runs')
    # fanova_set = pyfanova.fanova.Fanova(state_run_dir)
    fanova_set = pyfanova.fanova.Fanova(state_run_dir, improvement_over='QUANTILE', quantile_to_compare=0.25)
    max_marginals = 7
    cols_imp_df = ['marginal', 'parameter']
    temp_df = pd.DataFrame(fanova_set.print_all_marginals(max_num=max_marginals, pairwise=pairwise), columns=cols_imp_df)
    flatex = '%d_marginal_table_for_%s_over_q1_noprepro.tex' % (max_marginals, dataset)
    # flatex = '%d_marginal_table_for_%s_default_noprepro.tex' % (max_marginals, dataset)
    # To avoid dots
    pd.set_option('display.max_colwidth', -1)
    temp_df.to_latex(os.path.join(plot_dir, 'tables', flatex))
    print("Done printing latex")
    pd.set_option('display.max_colwidth', 51)
    if pairwise:
        temp_df.loc[:, 'parameter'] = temp_df.parameter.str.split(' x ')

    ## Plot now the marginals
    viz_set = pyfanova.visualizer.Visualizer(fanova_set)
    categorical_params = fanova_set.get_config_space().get_categorical_parameters()
    for p in temp_df.parameter:
        fig_hyper, ax_hyper = plt.subplots(1,1)
        if len(p) == 1:
            label = clear_name(p[0])
            if p[0] not in categorical_params:
                viz_set.plot_marginal(p[0], ax=ax_hyper)
            else:
                viz_set.plot_categorical_marginal(p[0], ax=ax_hyper)
            ax_hyper.set_xlabel(label)
        else:
            label = clear_name(p[0]) +'_X_'+clear_name(p[1])
            if p[0] in categorical_params:
                if p[1] not in categorical_params:
                    viz_set.plot_categorical_pairwise(p[0], p[1], ax=ax_hyper)
                    ax_hyper.set_xlabel(clear_name(p[1]))
                    ax_hyper.legend(loc='best', title=clear_name(p[0]))
                else:
                    continue
            else:
                if p[1] not in categorical_params:
                    viz_set.plot_contour_pairwise(p[0], p[1], ax=ax_hyper)
                    ax_hyper.set_xlabel(clear_name(p[0]))
                    ax_hyper.set_ylabel(clear_name(p[1]))
                else:
                    viz_set.plot_categorical_pairwise(p[1], p[0], ax=ax_hyper)
                    ax_hyper.set_xlabel(clear_name(p[0]))
                    ax_hyper.legend(loc='best', title=clear_name(p[1]))
        plt.tight_layout()
        # fig_hyper.savefig(os.path.join(plot_dir, '%s_for_%s_noprepro.pdf' % (label, dataset)))
        fig_hyper.savefig(os.path.join(plot_dir, '%s_for_%s_over_q1_noprepro.pdf' % (label, dataset)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("experiment_dir")
    parser.add_argument("dataset")
    parser.add_argument("output_path")
    parser.add_argument("pairwise")
    args = parser.parse_args()
    parameter_plotting(args.dataset, args.experiment_dir, args.output_path, args.pairwise)

