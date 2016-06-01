from argparse import ArgumentParser
import os
import csv
import time
import numpy as np
import glob
import natsort as ns
from joblib import Parallel, delayed
from functools import partial

from autosklearn.data.competition_data_manager import CompetitionDataManager
from autosklearn.ensembles.ensemble_selection import EnsembleSelection
from autosklearn.evaluation.util import calculate_score


def figure_indexes(path_predictions):
    fname = os.path.join(path_predictions, 'predictions_ensemble_*.npy')
    file_names = ns.natsorted(glob.glob(fname))
    return file_names


def writing_to_csv():
    pass


def ensemble_loop(iterations, starttime, info,
                  ensemble_size, valid_labels, test_labels):

    written_first_line = False
    all_predictions = []
    all_test_predictions = []
    identifiers = []
    csv_writer_list = []

    # Assign the time of the "current" model
    time_function_evaluation = os.path.getmtime(iterations[-1]) - starttime

    ids = os.path.basename(iterations[-1]).split(".")[0].split('_')[-1]
    print(ids)

    for index, iters in enumerate(iterations):
        test_fname = iters.replace('ensemble', 'valid')

        if not os.path.isfile(test_fname):
            continue

        predictions = np.load(iters)
        all_predictions.append(predictions)

        identifiers.append(index)
        test_predictions = np.load(test_fname)
        all_test_predictions.append(test_predictions)

    # Build the ensemble
    start = time.time()

    es_cls = EnsembleSelection(ensemble_size, info["task"], info["metric"])
    es_cls.fit(np.array(all_predictions), valid_labels, identifiers)
    order = es_cls.indices_

    # Compute validation error
    s1 = time.time()
    ensemble_error = 1 - calculate_score(
        valid_labels,
        np.nanmean(np.array(all_predictions)[order], axis=0),
        info["task"], info["metric"], info['label_num'])

    # Compute test error
    ensemble_test_error = 1 - calculate_score(
        test_labels,
        np.nanmean(np.array(all_test_predictions)[order], axis=0),
        info["task"], info["metric"], info['label_num'])

    ensemble_time = time.time() - start

    # We have to add an additional row for the first iteration
    if len(iterations) == 1:
        csv_writer_list.append({'Time': 0,
                                'Training (Empirical) Performance': ensemble_error,
                                'Test Set Performance': ensemble_test_error,
                                'AC Overhead Time': 0,
                                'Validation Configuration ID': 0})
        written_first_line = True

    csv_writer_list.append({'Time': ensemble_time + time_function_evaluation,
                            'Training (Empirical) Performance': ensemble_error,
                            'Test Set Performance': ensemble_test_error,
                            'AC Overhead Time': 0,
                            'Validation Configuration ID': ids})

    return csv_writer_list


def evaluation(dataset, experiment_dir, run, output_path, ensemble_size=50):

    dataset_name = os.path.basename(dataset[:-1] if dataset.endswith("/")
                                    else dataset)
    try:
        os.makedirs(os.path.join(output_path, dataset_name))
    except:
        pass

    # Set path to the predictions and load the labels
    path_predictions_ensemble = os.path.join(experiment_dir, ".auto-sklearn",
                                             "predictions_ensemble")
    path_predictions_test = os.path.join(experiment_dir, ".auto-sklearn",
                                         "predictions_valid")

    valid_labels = np.load(os.path.join(experiment_dir,
                                        ".auto-sklearn",
                                        "true_targets_ensemble.npy"))

    D = CompetitionDataManager(dataset, encode_labels=True)
    test_labels = D.data["Y_valid"]

    # Create output csv file
    csv_file = open(os.path.join(output_path, dataset_name,
                                 "ensemble_validationResults-traj-run-" + str(run) + "-walltime.csv"), "w")
    fieldnames = ['Time', 'Training (Empirical) Performance', 'Test Set Performance', 'AC Overhead Time',
                  'Validation Configuration ID']
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    list_models = figure_indexes(path_predictions_ensemble)
    list_iter_models = [list_models[:j+1] for j in range(len(list_models))]

    info = dict()
    info["task"] = D.info["task"]
    info["metric"] = D.info["metric"]
    info["label_num"] = D.info['label_num']

    with open(os.path.join(experiment_dir, ".auto-sklearn",
                           "start_time_%d" % int(run)), "r") as fh:
        start_time = float(fh.read())

    pfunc = partial(ensemble_loop, starttime=start_time, info=info,
                    ensemble_size=ensemble_size, test_labels=test_labels,
                    valid_labels=valid_labels)

    list_to_write = Parallel(n_jobs=-1)(delayed(pfunc)(i) for i in list_iter_models)
    for lw in list_to_write:
        for l in lw:
            csv_writer.writerow(l)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("experiment_dir")
    parser.add_argument("dataset")
    parser.add_argument("run", type=int)
    parser.add_argument("output_path")
    parser.add_argument("ensemble_size", type=int)
    args = parser.parse_args()
    evaluation(args.dataset, args.experiment_dir, args.run, args.output_path,
               args.ensemble_size)

