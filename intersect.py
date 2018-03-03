import getopt
import sys
import tempfile

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
        print("ARGS: SORTED_FILE_STAIN SORTED_FILE_REGIONS BINS OUTPUT_CSV")
        return

    file, regions, bins, output, = args
    bins = int(bins)
    df = pd.read_table(file, names=['chr', 'start', 'end', 'locus', 'name'])

    dfs = []
    for i in range(0, bins):
        print('Processing bin', i, 'of', bins)
        df['length'] = df['end'] - df['start']
        df['bin_start'] = df['start'] + (df['end'] - df['start']) * i / bins
        df['bin_end'] = df['start'] + (df['end'] - df['start']) * (i + 1) / bins
        df[['bin_start', 'bin_end']] = df[['bin_start', 'bin_end']].astype('int')
        # print(df[['chr', 'bin_start', 'bin_end']].head())

        with tempfile.NamedTemporaryFile(mode='w', suffix='_bin{}.bed4'.format(i),
                                         delete=False) as tmp:
            df[['chr', 'bin_start', 'bin_end']].to_csv(tmp.name, sep='\t', header=False, index=None)

            intersection_name = tmp.name + '_intersect.bed'
            with open(intersection_name, 'w') as ino:
                subprocess.call(
                    'bedtools intersect -sorted -a {} -b {} -c -wa'.format(tmp.name,
                                                      regions), shell=True, stdout=ino)
            print('Bins intersection', i, tmp.name, 'vs', regions, intersection_name)
            bin_intersect_df = pd.read_table(intersection_name, sep='\t', names=['chr', 'start', 'end', 'count'])['count']
            bin_intersect_df.index = df.index
            dfs.append(bin_intersect_df)

    df_intersection = pd.concat(dfs, axis=1)
    df_intersection.to_csv(output, sep=',', header=False, index=None)
    print('Saved', output)


if __name__ == "__main__":
    main()
