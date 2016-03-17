# -*- encoding: utf-8 -*-
"""
Script for fANOVA analysis
"""

from pyfanova.fanova import Fanova

state_run_dir = "tmp/old_autosk_tmp/728_Testing/state-run50/"

# This only works because the scenario txt file was severely
# changed.

fano_728 = Fanova(state_run_dir)

fano_728.print_all_marginals()

