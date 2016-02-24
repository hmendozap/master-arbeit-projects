# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 2016

@author: Hector

Class to load and store configurations
read from run or trajectory files
"""
import os
import pandas as pd
import natsort as _ns
import glob as _glob


def hyp_split(x, listing):
    param_value = x.strip().rstrip('"').replace("'", "").split('=')
    pname = param_value[0].replace("__", "")
    if pname not in listing:
        listing.append(pname)
    return param_value[1]


def load_run_configs(data_dir, scenario, preprocessor=None, full_config=False):
    """
    :param data_dir: Directory of where SMAC files live
    :param scenario: In this case, the dataset used to train the model
    :param preprocessor: Preprocessing method used in the data. None means all
    :param full_config: Whether to return also the configuration of the preprocessor, imputation and one-hot-encoding
    :return: pandas.DataFrame with the every performance (training errors) and the feed neural network configurations
    run by SMAC
    """
    run_filename = "runs_and_results-SHUTDOWN*"
    state_seed = "state-run*"
    if preprocessor is None:
        scenario_dir = os.path.join(data_dir, scenario, '*', scenario, state_seed, run_filename)
    else:
        scenario_dir = os.path.join(data_dir, scenario, preprocessor, scenario, state_seed, run_filename)

    dirs = _ns.natsorted(_glob.glob(scenario_dir))
    seeds = ['seed_' + itseeds.split('-')[-1].split('.')[0] for itseeds in dirs]
    all_runs = []
    runs_by_seed = []
    for fnames in dirs:
        try:
            run_res = load_run_by_file(fnames)
            all_runs.append(run_res)
            runs_by_seed.append(run_res.shape[0])
        except IndexError:
            print('CRASH in: ' + os.path.split(fnames)[1])

    # Treat each seed as independent runs
    runs_all_df = pd.concat(all_runs, axis=0)
    runs_all_df = runs_all_df.reset_index().drop('index', axis=1)
    # Try to convert to numeric type
    runs_all_df = runs_all_df.apply(pd.to_numeric, errors='ignore')

    return runs_all_df.copy()


def load_run_by_file(fname):
    """
    :param fname: filename to load
    :return: pandas.DataFrame with configuration and validation error
    """
    run_cols = ['config_id', 'response', 'runtime',
                'smac_iter', 'cum_runtime', 'run_result']

    rm_quote = lambda z: z.strip('" ')
    # TODO: Add try-catch statements
    run_df = pd.read_csv(fname,  delimiter=",", usecols=[1, 3, 7, 11, 12, 13],
                         skipinitialspace=False,
                         header=None, skiprows=1)

    run_df.columns = run_cols
    run_df.sort_values(by='response', axis=0, ascending=False, na_position='first', inplace=True)
    run_df.drop_duplicates('config_id', keep='last', inplace=True)

    base_dir = os.path.dirname(fname)
    config_filename = "paramstrings-SHUTDOWN*"
    confname = _glob.glob(os.path.join(base_dir, config_filename))[0]
    config_df = pd.read_csv(confname, delimiter=",|:\s", header=None)

    # Get the values of configuration parameters
    names = []
    config_df.iloc[:, 1:] = config_df.iloc[:, 1:].apply(lambda x: x.apply(hyp_split, args=(names,)))

    # Almost everything that goes from the second(:) is eliminated from names
    # list(map()) because python3
    classifier_names = list(map(lambda Y: Y.split(':')[-1], names))

    # Name column and remove not-classifier parameters
    config_df.columns = ['config_id'] + classifier_names
    cols_to_drop = [1, 31, 32, 33, 34]
    configuration_df = config_df.drop(config_df.columns[cols_to_drop], axis=1)
    run_config_df = pd.merge(run_df, configuration_df, on='config_id')

    return run_config_df.copy()


def load_trajectory_by_file(fname, full_config=False):
    """
    :param fname: filename to load
    :param full_config: Whether to return also the configuration of the preprocessor, imputation and one-hot-encoding
    :return: pandas.DataFrame with filtered columns
    """

    traj_cols = ['cpu_time', 'performance', 'wallclock_time',
    'incumbentID', 'autoconfig_time']

    rm_quote = lambda z: z.strip('" ')
    # TODO: Add try-catch statements
    traj_res = pd.read_csv(fname, delimiter=",",
    skipinitialspace=False, converters={5: rm_quote},
    header=None, skiprows=1)

    names = []
    traj_res.iloc[:, 1] = pd.to_numeric(traj_res.iloc[:, 1], errors='coerce')
    # Get the values of configuration parameters
    traj_res.iloc[:, 5:-1] = traj_res.iloc[:, 5:-1].apply(lambda x: x.apply(hyp_split, args=(names,)))

    # Almost everything that goes from the second(:) is eliminated
    classifier_names = list(map(lambda X: X.split(':')[-1], names))
    # Avoid this magic constant
    classifier_names[33] = 'preprocessor'

    traj_res.columns = traj_cols + classifier_names + ['expected']
    # Drop duplicated configuration and leave the best X-validation error
    traj_res.performance = pd.to_numeric(traj_res['performance'], errors='coerce')
    traj_res.sort_values(by='performance', axis=0, ascending=False, na_position='first', inplace=True)
    traj_res.drop_duplicates('incumbentID', keep='last', inplace=True)

    # Drop "unnecessary" columns
    cols_to_drop = [0, 2, 3, 4, 6, 35, 36, 37] + list(range(39, len(names)+6))
    class_df = traj_res.drop(traj_res.columns[cols_to_drop], axis=1)

    return class_df.copy()


def load_trajectories(data_dir, scenario, preprocessor=None, full_config=False):
    """
    :param data_dir: Directory of where SMAC files live
    :param scenario: Dataset used to train the model
    :param preprocessor: Preprocessing method used in the data. None means all
    :param full_config: Whether to return also the configuration of the preprocessor, imputation and one-hot-encoding
    :return: pandas.DataFrame with the performance (training errors) and the feed neural network configurations given
             by the detailed trajectory files
    """

    traj_filename =  "detailed-traj-run-*.csv"
    if preprocessor is None:
        scenario_dir = os.path.join(data_dir, scenario, '*', scenario, traj_filename)
    else:
        scenario_dir = os.path.join(data_dir, scenario, preprocessor, scenario, traj_filename)

    dirs = _ns.natsorted(_glob.glob(scenario_dir))
    seeds = ['seed_' + itseeds.split('-')[-1].split('.')[0] for itseeds in dirs]
    all_trajs = []
    runs_by_seed = []
    for fnames in dirs:
        try:
            run_res = load_trajectory_by_file(fnames)
            all_trajs.append(run_res)
            runs_by_seed.append(run_res.shape[0])
        except IndexError:
            print('CRASH in: ' + os.path.split(fnames)[1])

    # Treat each seed as independent runs
    traj_all_DF = pd.concat(all_trajs, axis=0)
    traj_all_DF = traj_all_DF.reset_index().drop('index', axis=1)
    # Try to convert to numeric type
    traj_all_DF = traj_all_DF.apply(pd.to_numeric, errors='ignore')

    return traj_all_DF.copy()
