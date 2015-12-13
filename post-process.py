import csv
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from statistics import median
import seaborn as sns

OUTPUT="out.csv"

def process(csvpath):

    # Seaborn settings
    sns.set_context("poster", font_scale=1.7)
    sns.set_style("white")
    sns.set_style("ticks")
    palette = sns.color_palette()

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
        fStats = {"fn": os.path.splitext(fn)[0], "rawData": [], "plotData": {}}
        with open(fn, 'rb') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='\"')
            for row in csvreader:
                if len(row)==4:
                    fStats["rawData"].append([int(row[0]), int(row[1]), int(row[2]), float(row[3])])

        for e in fStats["rawData"]:
            madd = e[1]
            error = e[3]
            if madd not in fStats["plotData"]:
                fStats["plotData"][madd] = {"error": [error]}
            else:
                fStats["plotData"][madd]["error"].append(error)
        stats.append(fStats)

    # Y-labels
    yLabels = ["Error"]
    # Dump to CSV file
    csvData = [["config", "MADD", "Energy"] + yLabels]
    for stat in stats:
        filename = str.split(stat["fn"], '/')
        config = filename[len(filename)-1]
        csvData += [[config]+x for x in sorted(stat["rawData"])]
    with open(OUTPUT, 'w') as f:
        for line in csvData:
            f.write("\t".join([str(x) for x in line])+"\n")


    # Plot data
    plots=[None] * len(stats)
    legend=[None] * len(stats)
    for i, stat in enumerate(stats):
        datapoints = []
        for feat in stat["plotData"]:
            err_list = stat["plotData"][feat]["error"]
            xy_dat = [feat]
            xy_dat.append(median(err_list))
            xy_dat.append(min(err_list))
            xy_dat.append(max(err_list))
            xy_dat.append(np.std(np.array(err_list)).tolist())
            datapoints.append(xy_dat)

        datapoints = sorted(datapoints)

        x = np.array([dat[0] for dat in datapoints])
        y = np.array([dat[1] for dat in datapoints])
        y_min = np.array([dat[2] for dat in datapoints])
        y_max = np.array([dat[3] for dat in datapoints])
        y_std = np.array([dat[4] for dat in datapoints])
        y_min = y - y_min
        y_max = y_max - y
        plots[i]=plt.scatter(x, y, c=palette[i%len(palette)], s=100)
        filename=str.split(stat["fn"], '/')
        legend[i]=filename[len(filename)-1]+" window"

    # Plot legend
    plt.legend(plots,
           legend,
           scatterpoints=1,
           loc='upper right',
           ncol=1)
    # Axes
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,y1,y2))
    plt.xlabel("ANN invocation cost (MADD ops)")
    plt.ylabel(yLabels[0])

    # Plot
    plt.savefig('ann.pdf', bbox_inches='tight')


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
