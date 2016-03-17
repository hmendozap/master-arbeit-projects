iterate_axis.hist(np.log10(groups.learning_rate.values), bins=15, histtype='bar', normed=1,
                         stacked=False, label=name, alpha=0.7, color=color_histograms.pop())