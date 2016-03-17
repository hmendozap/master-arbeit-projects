# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 2016

@author: Hector

Class to load and store configurations
read from run or trajectory files
"""
import os
import numpy as _np
import pandas as _pd
import natsort as _ns
import glob as _glob


def _hyp_split(x, listing):
    param_value = x.strip().rstrip('"').replace("'", "").split('=')
    pname = param_value[0].replace("__", "")
    if pname not in listing:
        listing.append(pname)
    return param_value[1]

DEBUG = False


class ConfigReader:

    def __init__(self, data_dir=None, dataset=None):
        self.runs_df = None
        self.bests_df = None
        self.trajectories_df = None
        self.dataset = dataset
        self.data_dir = data_dir
        self.full_config = False

    def load_run_configs(self, data_dir=None, dataset=None,
                         preprocessor='no_preprocessing', full_config=False):
        """
        Loads all configurations run by SMAC, with validation error response

        :param data_dir: Directory of where SMAC files live
        :param dataset: In this case, the dataset used to train the model
        :param preprocessor: Preprocessing method used in the data. None means all
        :param full_config: Whether to return also the configuration of the preprocessor,
                            imputation and one-hot-encoding
        :return: pandas.DataFrame with the every performance (training errors) and the feed neural network
                 configurations run by SMAC
        """
        if data_dir is None and self.data_dir is None:
            raise ValueError('Location of information not given')
        elif self.data_dir is not None:
            data_dir = self.data_dir

        if dataset is None:
            if self.dataset is None:
                raise ValueError('Dataset not given')
            else:
                dataset = self.dataset

        run_filename = "runs_and_results-SHUTDOWN*"
        state_seed = "state-run*"
        if preprocessor == 'all':
            scenario_dir = os.path.join(data_dir, dataset, '*', dataset, state_seed, run_filename)
        elif preprocessor is not None:
            scenario_dir = os.path.join(data_dir, dataset, preprocessor, dataset, state_seed, run_filename)
        else:
            scenario_dir = os.path.join(data_dir, dataset, state_seed, run_filename)

        dirs = _ns.natsorted(_glob.glob(scenario_dir))
        if len(dirs) == 0:
            raise ValueError('No runs_and_results files found.')

        seeds_names = ['runs_' + itseeds.split('state-run')[-1].split('/')[0] for itseeds in dirs]

        all_runs = []
        all_best = []
        runs_by_seed = []
        for fnames in dirs:
            try:
                run_res, best_run = self.load_run_by_file(fnames, full_config=full_config)
                all_runs.append(run_res)
                all_best.append(best_run)
                runs_by_seed.append(run_res.shape[0])
            except IndexError:
                print('CRASH in: ' + os.path.split(fnames)[1])

        # Treat each seed as independent runs
        runs_all_df = _pd.concat(all_runs, axis=0)
        runs_all_df = runs_all_df.reset_index().drop('index', axis=1)
        # Try to convert to numeric type
        runs_all_df = runs_all_df.apply(_pd.to_numeric, errors='ignore')

        # Best config each run
        best_all_df = _pd.concat(all_best, axis=1)
        best_all_df = best_all_df.apply(_pd.to_numeric, errors='ignore')
        best_all_df.columns = seeds_names

        self.runs_df = runs_all_df.copy()
        self.bests_df = best_all_df.T.copy()

        return runs_all_df.copy(), best_all_df.T.copy()

    @staticmethod
    def load_run_by_file(fname, full_config=False):
        """
        Loads one single configuration file run by SMAC, with validation error response
        :param fname: filename to load
        :param full_config: Whether to return also the configuration of the preprocessor,
                            imputation and one-hot-encoding
        :return: pandas.DataFrame with configuration and validation error
        """
        run_cols = ['config_id', 'response', 'runtime',
                    'smac_iter', 'cum_runtime', 'run_result']

        try:
            run_df = _pd.read_csv(fname, delimiter=",", usecols=[1, 3, 7, 11, 12, 13],
                                  skipinitialspace=False,
                                  header=None, skiprows=1)
        except OSError:
            raise OSError('file %s does not exist. Please check path' % fname)

        run_df.columns = run_cols
        run_df.sort_values(by='response', axis=0, ascending=False, na_position='first', inplace=True)
        run_df.drop_duplicates('config_id', keep='last', inplace=True)

        base_dir = os.path.dirname(fname)
        # TODO: Add checks to wheter one is using a non runs_results file
        config_run_match = fname.rsplit('runs_and_results-')[1].rsplit('.')[0]
        config_filename = "paramstrings-" + config_run_match + "*"
        if DEBUG:
            print(config_filename)

        try:
            confname = _glob.glob(os.path.join(base_dir, config_filename))[0]
        except IndexError:
            raise IndexError("There is no parameter configuration file")

        config_df = _pd.read_csv(confname, delimiter=",|:\s", header=None)

        # Get the values of configuration parameters
        names = []
        config_df.iloc[:, 1:] = config_df.iloc[:, 1:].apply(lambda x: x.apply(_hyp_split, args=(names,)))

        # Almost everything that goes from the second(:) is eliminated from names
        # list(map()) because python3
        filtered_names = list(map(lambda X: X.split(':')[-1], filter(lambda Z: Z.split(':')[0] != 'classifier', names)))
        classifier_names = list(map(lambda Y: Y.split(':')[-1], names))

        # Name column and remove not-classifier parameters
        config_df.columns = ['config_id'] + classifier_names

        if not full_config:
            # Delete classifier:choice parameter
            configuration_df = config_df.drop(filtered_names, axis=1)
            run_config_df = _pd.merge(run_df, configuration_df, on='config_id')
        else:
            run_config_df = _pd.merge(run_df, config_df, on='config_id')

        # Filter configurations over the error to have a better fit
        run_config_df = run_config_df[run_config_df['response'] > 0.0]
        run_config_df = run_config_df[run_config_df['response'] < 1.0]
        best_config_response = run_config_df.ix[run_config_df['response'].idxmin()]
        # run_config_df = run_config_df.query('response > 0 and response < 1.0')

        return run_config_df.copy(), best_config_response.copy()

    @staticmethod
    def load_trajectory_by_file(fname, full_config=False):
        """
        :param fname: filename to load
        :param full_config: Whether to return also the configuration of the preprocessor, imputation and one-hot-encoding
        :return: pandas.DataFrame with filtered columns
        """

        traj_cols = ['cpu_time', 'performance', 'wallclock_time',
                     'incumbentID', 'autoconfig_time']

        rm_quote = lambda z: z.strip('" ')

        try:
            traj_res = _pd.read_csv(fname, delimiter=",",
                                    skipinitialspace=False, converters={5: rm_quote},
                                    header=None, skiprows=1)
        except OSError:
            print('file %s does not exist. Please check path' % fname)

        names = []
        traj_res.iloc[:, 1] = _pd.to_numeric(traj_res.iloc[:, 1], errors='coerce')
        # Get the values of configuration parameters
        traj_res.iloc[:, 5:-1] = traj_res.iloc[:, 5:-1].apply(lambda x: x.apply(_hyp_split, args=(names,)))

        # TODO: Improve unnecesary columns droping using filter()

        if full_config:
            smac_cols = [tuple([a]+[b]) for a, b in zip(['smac']*len(traj_cols), traj_cols)]

            from operator import itemgetter
            full_parameter_names = map(lambda y: itemgetter(0, -1)(y.split(':')), names)

            params_inx = _pd.MultiIndex.from_tuples(smac_cols + list(full_parameter_names))
            traj_res.columns = params_inx + ['expected']
            traj_res.performance = _pd.to_numeric(traj_res['performance'], errors='coerce')
            traj_res.sort_values(by='performance', axis=0, ascending=False, na_position='first', inplace=True)
            traj_res.drop_duplicates('incumbentID', keep='last', inplace=True)

            class_df = traj_res.drop('expected', axis=1)
        else:
            classifier_names = list(map(lambda X: X.split(':')[-1], names))
            # Avoid this magic constant
            classifier_names[33] = 'preprocessor'

            traj_res.columns = traj_cols + classifier_names + ['expected']
            # Drop duplicated configuration and leave the best X-validation error
            traj_res.performance = _pd.to_numeric(traj_res['performance'], errors='coerce')
            traj_res.sort_values(by='performance', axis=0, ascending=False, na_position='first', inplace=True)
            traj_res.drop_duplicates('incumbentID', keep='last', inplace=True)

            # Drop "unnecessary" columns
            cols_to_drop = [0, 2, 3, 4, 6, 35, 36, 37] + range(39, len(names)+6)
            class_df = traj_res.drop(traj_res.columns[cols_to_drop], axis=1)

        return class_df.copy()

    def load_trajectories(self, data_dir=None, dataset=None,
                          preprocessor=None, full_config=False):
        """
        :param data_dir: Directory of where SMAC files live
        :param dataset: Dataset used to train the model
        :param preprocessor: Preprocessing method used in the data. None means all
        :param full_config: Whether to return also the configuration of the preprocessor, imputation and one-hot-encoding
        :return: pandas.DataFrame with the performance (training errors) and the feed neural network configurations given
                 by the detailed trajectory files
        """
        if data_dir is None:
            if self.data_dir is None:
                raise ValueError('Location of information not given')
            else:
                data_dir = self.data_dir

        if dataset is None:
            if self.dataset is not None:
                dataset = self.dataset
            else:
                raise ValueError('Dataset not given')

        # Could be done with find-like method, but no
        traj_filename = "detailed-traj-run-*.csv"
        if preprocessor == 'all':
            scenario_dir = os.path.join(data_dir, dataset, '*', dataset, traj_filename)
        elif preprocessor is not None:
            scenario_dir = os.path.join(data_dir, dataset, preprocessor, dataset, traj_filename)
        else:
            scenario_dir = os.path.join(data_dir, dataset, traj_filename)

        dirs = _ns.natsorted(_glob.glob(scenario_dir))
        seeds = ['seed_' + itseeds.split('-')[-1].split('.')[0] for itseeds in dirs]
        all_trajs = []
        runs_by_seed = []
        for fnames in dirs:
            try:
                run_res = self.load_trajectory_by_file(fnames, full_config=full_config)
                all_trajs.append(run_res)
                runs_by_seed.append(run_res.shape[0])
            except IndexError:
                print('CRASH in: ' + os.path.split(fnames)[1])

        trajectories_df = _pd.concat(all_trajs, axis=0)
        trajectories_df = trajectories_df.reset_index().drop('index', axis=1)
        # Try to convert to numeric type
        trajectories_df = trajectories_df.apply(_pd.to_numeric, errors='ignore')

        self.trajectories_df = trajectories_df.copy()
        return trajectories_df.copy()

    def save_config_run(self, storage_dir):
        """
        Saves the configurations of all runs in a numpy array
        :param storage_dir: Directory where to store the data, if empty uses the same as where data was taken
        :return:
        """
        if storage_dir is None:
            storage_dir = self.data_dir
        _np.save(os.path.join(storage_dir, 'run_data_matrix'), self.runs_df)

    def save_config_trajectories(self, storage_dir):
        """
        Saves the configurations of detailed trajectories in a numpy array
        :param storage_dir: Directory where to store the data, if empty uses the same as where data was taken
        :return:
        """
        if storage_dir is None:
            storage_dir = self.data_dir
        _np.save(os.path.join(storage_dir, 'trajectory_data_matrix'), self.trajectories_df)
