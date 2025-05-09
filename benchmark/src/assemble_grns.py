import pandas as pd
import os
import os.path as op

def read_grns(output_dir):
    file_list = []
    for fl in os.listdir(output_dir):
        if fl.endswith('.feather'):
            df = pd.read_feather(op.join(output_dir, fl))
            file_list.append(df)
    df = pd.concat(file_list)
    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Process a file from the command line.")

    # Add the file argument
    parser.add_argument('-d', type=str, help='Input directory', default = None)
    parser.add_argument('-f', type=str, help='Output file', default=None)


    # Parse the arguments
    args = parser.parse_args()

    df = read_grns(args.d)     
    df.to_csv(args.f, sep = '\t', index = False)
