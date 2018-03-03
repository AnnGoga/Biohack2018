import getopt
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# matplotlib.style.use('ggplot')
sns.set_style("whitegrid")


def plot(output, file):
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

    plt.figure(figsize=(10, 4))
    sns.barplot(data=pdf, x='bin', y='value', capsize=.2)

    # sns.regplot(pdf['bin'], pdf['value'])
    plt.savefig(output)
    print('Saved to', output)
    plt.close()


def logfc(output, param, param1):
    pass


def heatmap(output, files):
    pass


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
