from argparse import ArgumentParser
import os
import csv
import time
import numpy as np

from autosklearn.data.competition_data_manager import CompetitionDataManager
from autosklearn.ensembles.ensemble_selection import EnsembleSelection
from autosklearn.evaluation.util import calculate_score


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
    csv_file = open(os.path.join(output_path, dataset_name, "ensemble_validationResults-traj-run-" + str(run) + "-walltime.csv"), "w")
    fieldnames = ['Time', 'Training (Empirical) Performance', 'Test Set Performance', 'AC Overhead Time', 'Validation Configuration ID']
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    n_func_evals = len(os.listdir(path_predictions_ensemble))
    # assert n_func_evals == len(os.listdir(path_predictions_test))

    all_predictions = []
    all_test_predictions = []

    # Counts the number of function evaluation (also those that are broken)
    iters = 0
    # Counts the number of models
    idx = 0
    identifiers = []

    with open(os.path.join(experiment_dir, ".auto-sklearn",
                           "start_time_%d" % int(run)), "r") as fh:
        starttime = float(fh.read())
    # starttime = os.path.getmtime(os.path.join(experiment_dir, "space.pcs"))

    written_first_line = False
    while iters < n_func_evals:
        print("Iteration %d/%d" % (iters, n_func_evals))
        s = time.time()
        # Load predictions
        num_run = str(idx).zfill(5)
        file_name = "predictions_ensemble_%s_%s.npy" % (run, num_run)

        # If the current function evaluation failed continue with the next
        if not os.path.isfile \
                        (os.path.join(path_predictions_ensemble, file_name)):
            idx += 1
            continue

        # Read the modification time of the predictions file and compute the interval to the first prediction file.
        # This interval will be add to the time we needed to build the ensemble
        time_function_evaluation = os.path.getmtime(os.path.join(path_predictions_ensemble, file_name)) - starttime

        # Add the new prediction to the previous seen predictions
        predictions = np.load(os.path.join(path_predictions_ensemble, file_name))
        all_predictions.append(predictions)

        file_name = "predictions_valid_%s_%s.npy" % (run, num_run)

        if not os.path.isfile(os.path.join(path_predictions_test, file_name)):
            idx += 1
            continue

        identifiers.append(iters)
        test_predictions = np.load(os.path.join(path_predictions_test, file_name))
        all_test_predictions.append(test_predictions)
        # Build the ensemble
        start = time.time()
        es_cls = EnsembleSelection(ensemble_size, D.info["task"], D.info["metric"])
        es_cls.fit(np.array(all_predictions), valid_labels, identifiers)
        order = es_cls.get_model_identifiers()

        # Compute validation error
        s1 = time.time()
        ensemble_error = 1 - calculate_score(
            valid_labels,
            np.nanmean(np.array(all_predictions)[order], axis=0),
            D.info["task"], D.info["metric"], D.info['label_num'])

        # Compute test error
        ensemble_test_error = 1 - calculate_score(
            test_labels,
            np.nanmean(np.array(all_test_predictions)[order], axis=0),
            D.info["task"], D.info["metric"], D.info['label_num'])

        ensemble_time = time.time() - start

        # We have to add an additional row for the first iteration
        if not written_first_line:
            csv_writer.writerow({'Time': 0,
                                 'Training (Empirical) Performance': ensemble_error,
                                 'Test Set Performance': ensemble_test_error,
                                 'AC Overhead Time': 0,
                                 'Validation Configuration ID': 0})
            written_first_line = True

        csv_writer.writerow({'Time': ensemble_time + time_function_evaluation,
                             'Training (Empirical) Performance': ensemble_error,
                             'Test Set Performance': ensemble_test_error,
                             'AC Overhead Time': 0,
                             'Validation Configuration ID': idx})

        idx += 1
        iters += 1

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

