import argparse
from pathlib import Path
import os
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric')
    parser.add_argument('--it')
    args = parser.parse_args()

    data_dir = Path('/gscratch/xlab/alisaliu/cartography/data/glue/MNLI/al_0.1')
    prev_it_dir = data_dir / f'{args.metric}/{int(args.it)-1}/cartography_{args.metric}_19635/'
    curr_it_dir = data_dir / f'{args.metric}/{args.it}/cartography_{args.metric}_19635/'

    selected_file = curr_it_dir / 'selected.tsv'
    prev_train_file = prev_it_dir / 'train.tsv'
    new_train_file = curr_it_dir / 'train.tsv'
    
    os.rename(new_train_file, selected_file)

    with open(prev_train_file) as train, open(selected_file) as selected, open(new_train_file, 'w') as fo:
        train_lines = train.readlines()
        selected_lines = selected.readlines()[1:]
        train_lines.extend(selected_lines)
        for line in tqdm(train_lines):
            fo.write(line)
    
    print(f'Wrote augmented training data to {new_train_file}')


if __name__ == "__main__":
    main()
