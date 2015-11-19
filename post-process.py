import csv
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT="out.csv"

# MADD cost in Joules
MADD_COST = 3.03E-10

X_THRESHOLD=None

def process(csvpath):

    # Obtain all of the files with the .csv extension
    csvFiles = []
    if os.path.isfile(csvpath):
        csvFiles.append(csvpath)
    else:
        for file in os.listdir(csvpath):
            if file.endswith(".csv"):
                csvFiles.append(csvpath+'/'+file)

    print("Found {} csv files in {}".format(len(csvFiles), csvpath))

    # Load in scatterplot data
    stats = []
    for fn in csvFiles:
        fStats = {"fn": os.path.splitext(fn)[0], "errorData": []}
        with open(fn, 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='\"')
            for row in spamreader:
                if len(row)==6:
                    fStats["errorData"].append([int(row[0]), int(row[1]), int(row[2]), float(row[3]), float(row[4]), 1-float(row[5])])
                elif len(row)==4:
                    fStats["errorData"].append([int(row[0]), int(row[1]), int(row[2]), float(row[3])])
        stats.append(fStats)

    # Seaborn settings
    sns.set_style("white")
    sns.set_style("ticks")
    palette = [ '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Y-labels
    yLabels = ["classification", "false pos", "recall"]

    # Dump to CSV file
    csvData = [["config", "features", "MADD", "rep"] + yLabels]
    for stat in stats:
        filename = str.split(stat["fn"], '/')
        config = filename[len(filename)-1]
        csvData += [[config]+x for x in sorted(stat["errorData"])]
    with open(OUTPUT, 'w') as f:
        for line in csvData:
            f.write("\t".join([str(x) for x in line])+"\n")

    # Plot accuracies
    numPlot = len(stats[0]["errorData"])-4
    f, axarr = plt.subplots(numPlot, sharex=True)

    # Multiple subplots for all of the errors
    for subplot in range(numPlot):

        print stat["errorData"]

        # Plot data
        plots=[None] * len(stats)
        legend=[None] * len(stats)
        for i, stat in enumerate(stats):
            x = np.array([p[2]*MADD_COST for p in stat["errorData"]])
            y = np.array([p[3+subplot] for p in stat["errorData"]])
            plots[i]=axarr[subplot].scatter(x, y, c=palette[i%len(palette)])
            filename=str.split(stat["fn"], '/')
            legend[i]=filename[len(filename)-1]

        # Plot legend
        axarr[subplot].legend(plots,
               legend,
               title="Window Size",
               scatterpoints=1,
               loc='upper left',
               ncol=1,
               fontsize=6)

        # Axes
        x1,x2,y1,y2 = axarr[subplot].axis()
        axarr[subplot].axis((1E-9,1E-5,0,1))
        axarr[subplot].set_xscale('log')
        if subplot==numPlot-1:
            axarr[subplot].set_xlabel("ANN invocation cost (J)")
        axarr[subplot].set_ylabel(yLabels[subplot])
        f.suptitle("Accuracy-Energy Trade-offs", fontsize=14, fontweight='bold')

        # X Threshold line
        if X_THRESHOLD:
            axarr[subplot].axvline(X_THRESHOLD)

    # Plot
    f.savefig('ann.pdf', bbox_inches='tight')


def cli():
    parser = argparse.ArgumentParser(
        description='Plot the training statistics (csv format)'
    )
    parser.add_argument(
        '-path', dest='csvpath', action='store', type=str, required=True,
        default=None, help='directory containing the csv files to process'
    )
    args = parser.parse_args()

    process(args.csvpath)

if __name__ == '__main__':
    cli()
