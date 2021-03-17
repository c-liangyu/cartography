"""
Filtering and dataset mapping methods based on training dynamics.
By default, this module reads training dynamics from a given trained model and
computes the metrics---confidence, variability, correctness,
as well as baseline metrics of forgetfulness and threshold closeness
for each instance in the training data.
If specified, data maps can be plotted with respect to confidence and variability.
Moreover, datasets can be filtered with respect any of the other metrics.
"""
import argparse
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from collections import defaultdict
from typing import List

from cartography.data_utils import read_data, read_jsonl, copy_dev_test
from cartography.selection.selection_utils import read_training_dynamics

# TODO(SS): Named tuple for tasks and filtering methods.

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

ASCENDING_ORDER = {
    'variability': False,
    'mean_variability': False,
    'confidence': True,
    'final_confidence': True,
    'threshold_closeness': False,
    'forgetfulness': False,
    'correctness': True
}


def compute_forgetfulness(correctness_trend: List[float]) -> int:
    """
    Given a epoch-wise trend of train predictions, compute frequency with which
    an example is forgotten, i.e. predicted incorrectly _after_ being predicted correctly.
    Based on: https://arxiv.org/abs/1812.05159
    """
    if not any(correctness_trend):  # Example is never predicted correctly, or learnt!
        return 1000
    learnt = False  # Predicted correctly in the current epoch.
    times_forgotten = 0
    for is_correct in correctness_trend:
        if (not learnt and not is_correct) or (learnt and is_correct):
            # nothing changed.
            continue
        elif learnt and not is_correct:
            # Forgot after learning at some point!
            learnt = False
            times_forgotten += 1
        elif not learnt and is_correct:
            # Learnt!
            learnt = True
    return times_forgotten


def compute_correctness(trend: List[float]) -> float:
    """
    Aggregate #times an example is predicted correctly during all training epochs.
    """
    return sum(trend)


def compute_train_dy_metrics(training_dynamics, args):
    """
    Given the training dynamics (logits for each training instance across epochs), compute metrics
    based on it, for data map coorodinates.
    Computed metrics are: confidence, variability, correctness, forgetfulness, threshold_closeness---
    the last two being baselines from prior work
    (Example Forgetting: https://arxiv.org/abs/1812.05159 and Active Bias: https://arxiv.org/abs/1704.07433 respectively).
    Returns:
    - DataFrame with these metrics.
    - DataFrame with more typical training evaluation metrics, such as accuracy / loss.
    """
    confidence_ = {}
    final_confidence_ = {}
    variability_ = {}
    mean_variability_ = {}
    correctness_ = {}
    forgetfulness_ = {}
    threshold_closeness_ = {}

    # Functions to be applied to the data.
    def variability_func(conf): return np.std(conf)
    # Based on prior work on active bias (https://arxiv.org/abs/1704.07433)
    if args.include_ci:
        def variability_func(conf): return np.sqrt(
            np.var(conf) + np.var(conf) * np.var(conf) / (len(conf)-1))

    def threshold_closeness_func(conf): return conf * (1 - conf)

    loss = torch.nn.CrossEntropyLoss()

    num_tot_epochs = np.max([len(record['logits']) for record in training_dynamics.values()])
    logger.info(f"Computing training dynamics across {num_tot_epochs} epochs")
    logger.info("Metrics computed: confidence, final confidence, variability, mean variability, correctness, forgetfulness, threshold_closeness")

    logits = {i: [] for i in range(num_tot_epochs)}
    targets = {i: [] for i in range(num_tot_epochs)}
    training_accuracy = defaultdict(float)

    for guid in tqdm(training_dynamics):
        correctness_trend = []
        true_probs_trend = []
        probs_per_label = [[],[],[]]

        record = training_dynamics[guid]
        # skip examples that we do not have training dynamics for all epochs for
        if len(record['logits']) < num_tot_epochs:
            continue
        for i, epoch_logits in enumerate(record["logits"]):
            probs = torch.nn.functional.softmax(torch.Tensor(epoch_logits), dim=-1)
            for l in range(3):
                probs_per_label[l].append(float(probs[l]))
            
            true_class_prob = float(probs[record["gold"]])
            true_probs_trend.append(true_class_prob)

            prediction = np.argmax(epoch_logits)
            is_correct = (prediction == record["gold"]).item()
            correctness_trend.append(is_correct)

            training_accuracy[i] += is_correct
            logits[i].append(epoch_logits)
            targets[i].append(record["gold"])

        correctness_[guid] = compute_correctness(correctness_trend)
        confidence_[guid] = np.mean(true_probs_trend)
        final_confidence_[guid] = np.max([probs_per_label[l][-1] for l in range(3)])
        variability_[guid] = variability_func(true_probs_trend)
        mean_variability_[guid] = np.mean([variability_func(probs_per_label[l]) for l in range(3)])
        forgetfulness_[guid] = compute_forgetfulness(correctness_trend)
        threshold_closeness_[guid] = threshold_closeness_func(confidence_[
                                                              guid])

    # Should not affect ranking, so ignoring.
    epsilon_var = np.mean(list(variability_.values()))

    column_names = ['guid',
                    'index',
                    'threshold_closeness',
                    'confidence',
                    'final_confidence',
                    'variability',
                    'mean_variability',
                    'correctness',
                    'forgetfulness']
    df = pd.DataFrame([[guid,
                        i,
                        threshold_closeness_[guid],
                        confidence_[guid],
                        final_confidence_[guid],
                        variability_[guid],
                        mean_variability_[guid],
                        correctness_[guid],
                        forgetfulness_[guid],
                        ] for i, guid in enumerate(correctness_)], columns=column_names)

    df_train = pd.DataFrame([[i,
                              loss(torch.Tensor(logits[i]), torch.LongTensor(
                                  targets[i])).item() / len(training_dynamics),
                              training_accuracy[i] / len(training_dynamics)
                              ] for i in range(num_tot_epochs)],
                            columns=['epoch', 'loss', 'train_acc'])
    return df, df_train


def write_mixed_data(args, train_dy_metrics):
    # First save the args for filtering, to keep track of which model was used for filtering.
    argparse_dict = vars(args)
    with open(os.path.join(args.output_dir, f"filtering_configs.json"), "w") as outfile:
        outfile.write(json.dumps(
            argparse_dict, indent=4, sort_keys=True) + "\n")
    
    is_ascending = ASCENDING_ORDER['variability']
    if args.worst:
        is_ascending = not is_ascending
    sorted_ambi_scores = train_dy_metrics.sort_values(by=['variability'],
                                                ascending=is_ascending)

    is_ascending = not ASCENDING_ORDER['confidence']
    sorted_easy_scores = train_dy_metrics.sort_values(by=['confidence'],
                                                ascending=is_ascending)

    original_train_file = os.path.join(args.data_dir, f"train.tsv")
    train_numeric, header = read_data(original_train_file, task_name=args.task_name, guid_as_int=True)

    # only one fraction
    if not args.fraction:
        args.fraction = 0.05
    fractions_replace = [0.1, 0.2, 0.25, 0.33, 0.5]
    for fraction_replace in fractions_replace:
        outdir = os.path.join(args.output_dir, f"cartography_{args.metric}_{fraction_replace:.2f}")
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # Dev and test need not be subsampled.
        copy_dev_test(args.task_name,
                      from_dir=args.data_dir,
                      to_dir=outdir)

        num_easy_samples = int(args.fraction * fraction_replace * len(train_numeric))
        num_ambi_samples = int(args.fraction * len(train_numeric)) - num_easy_samples

        with open(os.path.join(outdir, f"train.tsv"), "w") as outfile:
            outfile.write(header + "\n")
            selected_easy = sorted_easy_scores.head(n=num_easy_samples+1)
            selected_ambiguous = sorted_ambi_scores.head(n=num_ambi_samples+1)
            selected = pd.concat([selected_easy, selected_ambiguous])
            selection_iterator = tqdm(range(len(selected)))
            for idx in selection_iterator:
                selection_iterator.set_description(f"mixed")
                selected_id = selected.iloc[idx]["guid"]
                if args.task_name in ["SNLI", "MNLI"]:
                    selected_id = int(selected_id)
                elif args.task_name == "WINOGRANDE":
                    selected_id = str(int(selected_id))
                record = train_numeric[selected_id]
                outfile.write(record + "\n")

        logger.info(f"Wrote {num_easy_samples + num_ambi_samples} samples to {outdir}.")


def write_filtered_data(args, train_dy_metrics):
    """
    Filter data based on the given metric, and write it in TSV format to train GLUE-style classifier.
    """
    # First save the args for filtering, to keep track of which model was used for filtering.
    argparse_dict = vars(args)
    with open(os.path.join(args.output_dir, f"filtering_configs.json"), "w") as outfile:
        outfile.write(json.dumps(
            argparse_dict, indent=4, sort_keys=True) + "\n")

    # sort by selection
    if args.metric == 'random':
        sorted_scores = train_dy_metrics.sample(frac=1, random_state=args.seed)
    else:
        # determine whether to sort data in ascending order or not, based on the metric
        is_ascending = ASCENDING_ORDER[args.metric]
        if args.worst:
            is_ascending = not is_ascending
        sorted_scores = train_dy_metrics.sort_values(by=[args.metric],
                                                    ascending=is_ascending)

    original_train_file = os.path.join(args.data_dir, args.data_file)
    train_numeric, header = read_data(original_train_file, task_name=args.task_name, guid_as_int=True)

    if args.n and args.n > 1:
        num_samples = [args.n]
    elif args.fraction and 0 < args.fraction < 1:
        fractions = [args.fraction]
        num_samples = [int(args.fraction * len(train_numeric))]
    else:
        fractions = [0.01, 0.05, 0.10, 0.1667, 0.25, 0.33, 0.50, 0.75]
        num_samples = [int(f * len(train_numeric)) for f in fractions]

    for i, n in enumerate(num_samples):
        # name and create folder
        if args.n:
            outdir = os.path.join(args.output_dir, f"cartography_{args.metric}_{n}")
        else:
            outdir = os.path.join(args.output_dir, f"cartography_{args.metric}_{fractions[i]:.2f}")
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # dev and test need not be subsampled
        copy_dev_test(args.task_name,
                      from_dir=args.data_dir,
                      to_dir=outdir)

        with open(os.path.join(outdir, f"train.tsv"), "w") as outfile:
            outfile.write(header + "\n")
            selected = sorted_scores.head(n=n)
            if args.both_ends:
                hardest = sorted_scores.head(n=int(n * 0.7))
                easiest = sorted_scores.tail(n=n-hardest.shape[0])
                selected = pd.concat([hardest, easiest])
                fm = args.metric
                logger.info(f"Selecting both ends: {fm} = "
                            f"({hardest.head(1)[fm].values[0]:3f}: {hardest.tail(1)[fm].values[0]:3f}) "
                            f"& ({easiest.head(1)[fm].values[0]:3f}: {easiest.tail(1)[fm].values[0]:3f})")

            selection_iterator = tqdm(range(len(selected)))
            for idx in selection_iterator:
                # set tqdm bar description
                if args.metric == 'random':
                    selection_iterator.set_description('Random')
                else:
                    selection_iterator.set_description(
                        f"{args.metric} = {selected.iloc[idx][args.metric]:.4f}")

                selected_id = selected.iloc[idx]["guid"]
                if args.task_name in ["SNLI", "MNLI"]:
                    selected_id = int(selected_id)
                elif args.task_name == "WINOGRANDE":
                    selected_id = str(int(selected_id))
                record = train_numeric[selected_id]
                outfile.write(record + "\n")

        logger.info(f"Wrote {n} samples to {outdir}.")


def plot_data_map(dataframe: pd.DataFrame,
                  plot_dir: os.path,
                  hue_metric: str = 'correct.',
                  task_name: str = '',
                  plot_title: str = None,
                  show_hist: bool = False,
                  max_instances_to_plot: int = 25000):
    # Set style.
    sns.set(style='whitegrid', font_scale=1.6, context='paper')
    logger.info(f"Plotting figure for {task_name} ...")

    # Subsample data to plot, so the plot is not too busy.
    dataframe = dataframe.sample(
        n=max_instances_to_plot if dataframe.shape[0] > max_instances_to_plot else len(dataframe))

    # Normalize correctness to a value between 0 and 1.
    dataframe = dataframe.assign(
        corr_frac=lambda d: d.correctness / d.correctness.max())
    dataframe['correct.'] = [f"{x:.1f}" for x in dataframe['corr_frac']]

    main_metric = 'variability'
    other_metric = 'confidence'

    hue = hue_metric
    num_hues = len(dataframe[hue].unique().tolist())
    style = hue_metric if num_hues < 8 else None

    if not show_hist:
        fig, ax0 = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = plt.figure(figsize=(14, 10), )
        gs = fig.add_gridspec(3, 2, width_ratios=[5, 1])
        ax0 = fig.add_subplot(gs[:, 0])

    pal = sns.diverging_palette(260, 15, n=num_hues, sep=10, center="dark")

    plot = sns.scatterplot(x=main_metric,
                           y=other_metric,
                           ax=ax0,
                           data=dataframe,
                           hue=hue,
                           palette=pal,
                           style=style,
                           s=30)

    # Annotate Regions.
    def bb(c): return dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")

    def func_annotate(text, xyc, bbc): return ax0.annotate(text,
                                                           xy=xyc,
                                                           xycoords="axes fraction",
                                                           fontsize=15,
                                                           color='black',
                                                           va="center",
                                                           ha="center",
                                                           rotation=350,
                                                           bbox=bb(bbc))
    an1 = func_annotate("ambiguous", xyc=(0.9, 0.5), bbc='black')
    an2 = func_annotate("easy-to-learn", xyc=(0.27, 0.85), bbc='r')
    an3 = func_annotate("hard-to-learn", xyc=(0.35, 0.25), bbc='b')

    if not show_hist:
        plot.legend(ncol=1, bbox_to_anchor=[0.175, 0.5], loc='right')
    else:
        plot.legend(fancybox=True, shadow=True,  ncol=1)
    plot.set_xlabel('variability')
    plot.set_ylabel('confidence')

    if show_hist:
        if not plot_title:
            plot_title = f"{task_name} Data Map"
        plot.set_title(plot_title, fontsize=17)

        # Make the histograms.
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[2, 1])

        plott0 = dataframe.hist(column=['confidence'], ax=ax1, color='#622a87')
        plott0[0].set_title('')
        plott0[0].set_xlabel('confidence')
        plott0[0].set_ylabel('density')

        plott1 = dataframe.hist(column=['variability'], ax=ax2, color='teal')
        plott1[0].set_title('')
        plott1[0].set_xlabel('variability')
        plott1[0].set_ylabel('density')

        plot2 = sns.countplot(x="correct.", data=dataframe,
                              ax=ax3, color='#86bf91')
        ax3.xaxis.grid(True)  # Show the vertical gridlines

        plot2.set_title('')
        plot2.set_xlabel('correctness')
        plot2.set_ylabel('density')

    fig.tight_layout()
    filename = f'{plot_dir}/{task_name}.png' if show_hist else f'figures/compact_{task_name}.png'
    fig.savefig(filename, dpi=300)
    logger.info(f"Plot saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter",
                        action="store_true",
                        help="Whether to filter data subsets based on specified `metric`.")
    parser.add_argument("--plot",
                        action="store_true",
                        help="Whether to plot data maps and save as `pdf`.")
    parser.add_argument("--model_dir",
                        "-o",
                        required=True,
                        type=os.path.abspath,
                        help="Directory where model training dynamics stats reside.")
    parser.add_argument("--data_dir",
                        "-d",
                        default="/Users/swabhas/data/glue/WINOGRANDE/xl/",
                        type=os.path.abspath,
                        help="Directory where data for task resides.")
    parser.add_argument("--data_file",
                        default='train.tsv',
                        type=str,
                        help="Name of data file.")
    parser.add_argument("--plots_dir",
                        default="./cartography/",
                        type=os.path.abspath,
                        help="Directory where plots are to be saved.")
    parser.add_argument("--plot_title",
                        default=None,
                        type=str,
                        help="Plot caption")
    parser.add_argument("--task_name",
                        "-t",
                        default="WINOGRANDE",
                        choices=("SNLI", "MNLI", "QNLI", "WINOGRANDE"),
                        help="Which task are we plotting or filtering for.")
    parser.add_argument('--metric',
                        choices=('threshold_closeness',
                                 'confidence',
                                 'final_confidence',
                                 'variability',
                                 'mean_variability',
                                 'correctness',
                                 'forgetfulness',
                                 'random',
                                 'mixed'),
                        help="Metric to filter data by.",)
    parser.add_argument("--seed",
                        type=int,
                        default=725862,
                        help="Random seed for sampling.")
    parser.add_argument('--n',
                        type=int,
                        help="Integer indicating number of random samples to select")
    parser.add_argument('--fraction',
                        type=float,
                        help="Number between 0 and 1, indicating fraction of random samples to select.")
    parser.add_argument("--include_ci",
                        action="store_true",
                        help="Compute the confidence interval for variability.")
    parser.add_argument("--output_dir",
                        "-f",
                        default="./filtered/",
                        type=os.path.abspath,
                        help="Output directory where filtered datasets are to be written.")
    parser.add_argument("--worst",
                        action="store_true",
                        help="Select from the opposite end of the spectrum acc. to metric,"
                             "for baselines")
    parser.add_argument("--both_ends",
                        action="store_true",
                        help="Select from both ends of the spectrum acc. to metric,")
    parser.add_argument("--overwrite_train_dy",
                        action="store_true",
                        help="Whether to overwrite previously computed training dynamics")
    parser.add_argument("--split",
                        type=str,
                        default='training',
                        help="Dataset split whose training dynamics to read")

    args = parser.parse_args()
    train_dy_filename = os.path.join(args.model_dir, f"{args.split}_td_metrics.jsonl")

    if args.overwrite_train_dy or not os.path.exists(train_dy_filename):
        training_dynamics = read_training_dynamics(args.model_dir, split=args.split)
        train_dy_metrics, _ = compute_train_dy_metrics(training_dynamics, args)
        train_dy_metrics.to_json(train_dy_filename, orient='records', lines=True)
        logger.info(f"Metrics for {args.split} data based on training dynamics written to {train_dy_filename}")
    else:
        logger.info(f"Read metrics for {args.split} data based on training dynamics from {train_dy_filename}")
        train_dy_metrics = pd.read_json(train_dy_filename, lines=True)

    if args.filter:
        assert args.output_dir
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        assert args.metric

        if args.metric == 'mixed':
            write_mixed_data(args, train_dy_metrics)
        else:
            write_filtered_data(args, train_dy_metrics)

    if args.plot:
        assert args.plots_dir
        if not os.path.exists(args.plots_dir):
            os.makedirs(args.plots_dir)
        plot_data_map(train_dy_metrics, args.plots_dir, task_name=args.task_name,
                      show_hist=True, plot_title=args.plot_title)
