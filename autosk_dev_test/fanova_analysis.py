# -*- encoding: utf-8 -*-
"""
Script for fANOVA analysis
"""

from pyfanova.fanova import Fanova
from pyfanova.visualizer import Visualizer

dataset_dir = "/mhome/mendozah/constrained_DeepNet_Configs/adam_results/MNIST/inv/merged_runs_inv/"

# This only works because the scenario txt file was severely
# changed.

fano_MNIST = Fanova(dataset_dir)
viz_MNIST = Visualizer(fano_MNIST)

fano_MNIST.print_all_marginals()
viz_MNIST.create_all_plots("./fanova_plots/")

