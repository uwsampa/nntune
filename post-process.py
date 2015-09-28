import csv
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

EXT=".csv"

def process(csvdir):

    # Obtain all of the files with the .csv extension
    csvFiles = []
    for file in os.listdir(csvdir):
        if file.endswith(".csv"):
            csvFiles.append(file)

    print("Found {} csv files in {}".format(len(csvFiles), csvdir))

    # Load in scatterplot data
    stats = []
    for fn in csvFiles:
        fStats = {"fn": os.path.splitext(fn)[0], "errorData": []}
        with open(csvdir+'/'+fn, 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='\"')
            for row in spamreader:
                if len(row)==3:
                    fStats["errorData"].append([row[0], int(row[1]), float(row[2])*100.0])
        stats.append(fStats)

    # Seaborn settings
    sns.set_style("white")
    sns.set_style("ticks")
    palette = sns.color_palette("Set2")

    # Plot data
    plots=[None] * len(stats)
    legend=[None] * len(stats)
    for i, stat in enumerate(stats):
        x = np.array([p[1] for p in stat["errorData"]])
        y = np.array([p[2] for p in stat["errorData"]])
        plots[i]=plt.scatter(x, y, c=palette[i%len(palette)])
        legend[i]=stat["fn"]

    # Plot legend
    plt.legend(plots,
           legend,
           title="Face Size",
           scatterpoints=1,
           loc='upper left',
           ncol=2,
           fontsize=8)

    # Axes
    x1,x2,y1,y2 = plt.axis()
    plt.axis((10,x2,0,y2))
    plt.xscale('log')
    plt.xlabel("MADD ops per ANN invocation")
    plt.ylabel("Classification Error (%)")
    plt.suptitle("Classification Error vs. MADD ops", fontsize=14, fontweight='bold')

    # Plot
    plt.show()

def cli():
    parser = argparse.ArgumentParser(
        description='Plot the training statistics (csv format)'
    )
    parser.add_argument(
        '-dir', dest='csvdir', action='store', type=str, required=True,
        default=None, help='directory containing the csv files to process'
    )
    args = parser.parse_args()

    process(args.csvdir)

if __name__ == '__main__':
    cli()
