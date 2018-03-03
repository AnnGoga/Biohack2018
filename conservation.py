import getopt
import sys
import tempfile
from subprocess import run

import pandas as pd
import subprocess


def main():
    argv = sys.argv
    opts, args = getopt.getopt(argv[1:], "h", ["help"])
    # Process help
    for o, a in opts:
        if o in ("-h", "--help"):
            return 'NO HELP HERE'

    if len(args) != 4:
        print("ARGS: CONSERVATION FILE BINS OUTPUT")
        return

    conservation = args[0]
    file = args[1]
    bins = int(args[2])
    output = args[3]
    df = pd.read_table(file, names=['chr', 'start', 'end', 'locus', 'name'])

    bins_cons = []
    for i in range(0, bins):
        print('Processing bin', i, 'of', bins)
        df['length'] = df['end'] - df['start']
        df['bin_start'] = df['start'] + (df['end'] - df['start']) * i / bins
        df['bin_end'] = df['start'] + (df['end'] - df['start']) * (i + 1) / bins
        df[['bin_start', 'bin_end']] = df[['bin_start', 'bin_end']].astype('int')
        df['id'] = [str(df['chr'][i]) + '#' + str(df['bin_start'][i]) + '#' + str(df['bin_end'][i]) for i in
                    range(0, len(df))]
        # print(df[['chr', 'bin_start', 'bin_end', 'id']].head())

        with tempfile.NamedTemporaryFile(mode='w', suffix='_bin{}.bed4'.format(i),
                                         prefix=str(df['name'][0]),
                                         delete=False) as tmp:
            df[['chr', 'bin_start', 'bin_end', 'id']].to_csv(tmp.name, sep='\t', header=False, index=None)

            tmp_cons = tmp.name + '_cons.tsv'
            subprocess.call('bigWigAverageOverBed {} {} {}'.format(conservation,
                                                                   tmp.name,
                                                                   tmp_cons), shell=True)
            print('Bins', i, tmp.name, tmp_cons)
            bin_cons = pd.read_table(tmp_cons, sep='\t',
                                     names=('id', 'coverage', 'mean0', 'mean', 'name'))['mean']
            bin_cons.index = df.index
            bins_cons.append(bin_cons)

    df_conservation = pd.concat(bins_cons, axis=1)
    df_conservation.to_csv(output, sep=',', header=False, index=None)
    print('Saved', output)


if __name__ == "__main__":
    main()
