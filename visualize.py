import getopt
import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np

# matplotlib.style.use('ggplot')
sns.set_style("whitegrid")


def plot(output, file):
    print('PLOT')
    df = pd.read_table(file, header=None, sep=',')
    print('dim', len(df), 'x', len(df.columns))
    df = df.T
    df['bin'] = df.index
    pdf = pd.melt(df, id_vars='bin', var_name='band', value_name='value')
    pdf['value'] = pdf['value'].astype('float')
    pdf['bin'] = pdf['bin'].astype('int')

    print(pdf.head())
    # for c in pdf.columns:
    #     print(c, type(pdf[c]))
    # print('bins', set(pdf['bin']))
    # print('values', set(pdf['value']))

    plt.figure(figsize=(20, 5))
    sns.barplot(data=pdf, x='bin', y='value', capsize=.2, ci="sd", errwidth=2)
    xs = [str(i) for i in range(0, len(set(pdf['bin'])))]
    plt.xticks(np.arange(len(xs)), xs, rotation='vertical')
    plt.savefig(output)
    print('Saved to', output)
    plt.close()


def logfc(output, file1, file2):
    print('LOGFC')
    df1 = pd.read_table(file1, header=None, sep=',')
    print('dim', len(df1), 'x', len(df1.columns))

    df2 = pd.read_table(file2, header=None, sep=',')
    print('dim', len(df2), 'x', len(df2.columns))

    df1_avg = df1.T.mean(axis=1)
    # print(df1_avg.head())

    df2_avg = df2.T.mean(axis=1)
    print(len(df1_avg))
    # print(df2_avg.head())

    df_logfc = np.log(df1_avg / df2_avg)
    df_logfc.fillna(value=0, inplace=True)

    df_2plot = pd.DataFrame()
    df_2plot['bin'] = df_logfc.index
    df_2plot['logfc'] = df_logfc
    plt.figure(figsize=(20, 5))
    sns.barplot(data=df_2plot, x='bin', y='logfc', capsize=.2, ci="sd", errwidth=2)
    xs = [str(i) for i in range(0, len(set(df_2plot['bin'])))]
    plt.xticks(np.arange(len(xs)), xs, rotation='vertical')
    plt.savefig(output)
    plt.close()
    print('Saved', output)


def heatmap(output, files):
    print('HEATMAP')
    dfs = pd.DataFrame()
    for file in files:
        df = pd.read_table(file, header=None, sep=',')
        print('dim', len(df), 'x', len(df.columns))

        df_avg = df.T.mean(axis=1)
        dfs[os.path.basename(file)] = df_avg
    dfs = dfs.T
    print(dfs.head())
    print(len(dfs))
    dfs = dfs[dfs.columns].astype(float)
    plt.figure(figsize=(20, len(dfs)))
    sns.heatmap(dfs)
    xs = [str(i) for i in dfs.T.index]
    plt.xticks(np.arange(len(xs)), xs, rotation='vertical')
    ys = [str(i) for i in dfs.T.columns]
    print(ys)
    plt.yticks(np.arange(len(ys)), ys, rotation='horizontal')
    plt.savefig(output)
    plt.close()
    print('Saved', output)


def main():
    argv = sys.argv
    opts, args = getopt.getopt(argv[1:], "h", ["help"])
    # Process help
    for o, a in opts:
        if o in ("-h", "--help"):
            return 'NO HELP HERE'

    if len(args) < 3:
        print("ARGS: [plot|logfc|heatmap] OUTPUT.png FILE1.csv FILE2.csv ... FILEN.csv")
        return

    state = args[0]
    output = args[1]
    files = args[2:]
    if state == 'plot':
        if len(files) != 1:
            print("ERROR 1 plot file required")
            exit(1)
        plot(output, files[0])
    elif state == 'logfc':
        if len(files) != 2:
            print("ERROR 2 logfc files required")
            exit(1)
        logfc(output, files[0], files[1])
    elif state == 'heatmap':
        if len(files) == 0:
            print("ERROR at least 1 heatmap file required")
            exit(1)
        heatmap(output, files)
    else:
        print("ERROR unknown params")
        exit(1)


if __name__ == "__main__":
    main()
