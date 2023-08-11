import pandas as pd
import os
import numpy as np
import seaborn as sns
from datasets import load_dataset
import matplotlib.pyplot as plt


# Definitions {{{ #
# define manually as to keep constant for all datasets
categories = [
  'Legality_Constitutionality_and_jurisprudence',
  'Quality_of_life',
  'Cultural_identity',
  'Fairness_and_equality',
  'Health_and_safety',
  'Policy_prescription_and_evaluation',
  'Political',
  'Capacity_and_resources',
  'Economic',
  'Public_opinion',
  'Morality',
  'Crime_and_punishment',
  'External_regulation_and_reputation',
  'Security_and_defense',
  ]
categories.sort()
category_map = {cat: i for i, cat in enumerate(categories)}
# }}} Definitions #

# Dataset Loading {{{ #
def get_features_from_files(features_filenames):
    return load_dataset("text", data_files=features_filenames, sample_by="document", split="train")


def get_dataset(data_dir, prefix='train'):
    def attach_labels(record, idx):
        lbls = labels.iloc[idx]["labels"].split(",")
        return {"labels": [int(cat_name in lbls) for cat_name in categories]}

    features_dir_path = f"{data_dir}/{prefix}-articles-subtask-2"
    labels_path = f"{data_dir}/{prefix}-labels-subtask-2.txt"
    labels = pd.read_csv(labels_path, sep="\t", header=None, names=["ids", "labels"], index_col="ids")
    features_filenames = [os.path.join(features_dir_path, f"article{id}.txt") for id in labels.index]
    features = get_features_from_files(features_filenames)

    return features.map(attach_labels, with_indices=True)


def get_split_dataset(data_dir, split=0.2, seed=42):
    split_dataset = get_dataset(data_dir).train_test_split(split, seed=seed)
    return split_dataset["train"], split_dataset["test"]
# }}} Dataset Loading #

# Dataset Preprocessing {{{ #
def oversample(dataset, balance_factor):
    filtered_sets = [dataset.filter(lambda x: x["labels"][i]) for i in range(len(dataset["labels"][0]))]
    rng = np.random.default_rng(42)
    
    def get_balance_factor(labels):
        label_sums = np.sum(labels, axis=0)
        return np.max(label_sums) / np.min(label_sums)
    
    while get_balance_factor(dataset["labels"]) > balance_factor:
        min_label = np.argmin(np.sum(dataset["labels"], axis=0))
        sample_idx = rng.integers(low=0, high=len(filtered_sets[min_label]))
        sample = filtered_sets[min_label][int(sample_idx)]
        dataset = dataset.add_item(sample)
    return dataset
# }}} Dataset Preprocessing #

# Plotting and Scoring {{{ #
from sklearn.metrics import f1_score


def calculate_agreements(predictions, references, model1, model2):
    agreements = pd.DataFrame({
        'Labels': categories,
        'Correct Agreement': np.sum((predictions[model1] == references) & (predictions[model2] == references), axis=0) / len(references),
        f'{model1} correct': np.sum((predictions[model1] == references) & (predictions[model2] != references), axis=0) / len(references),
        f'{model2} correct': np.sum((predictions[model1] != references) & (predictions[model2] == references), axis=0) / len(references),
        'False Agreement': np.sum((predictions[model1] != references) & (predictions[model2] != references), axis=0) / len(references),
        f'{model1} MicroF1': f1_score(references, predictions[model1], average=None),
        f'{model2} MicroF1': f1_score(references, predictions[model2], average=None),
    })
    return agreements


def plot_agreements(agreements, train_labels):
    cats = agreements.Labels
    agreements = agreements.iloc[:, 1:]
    cumsum = agreements.cumsum(axis=1)

    fig, ax = plt.subplots()
    colors = plt.colormaps['RdYlGn_r'](np.linspace(.2, .85, 4))
    for i, color in enumerate(colors):
        lengths = agreements.iloc[:, i]
        starts = cumsum.iloc[:, i] - lengths
        rects = ax.barh(cats, lengths, left=starts, height=0.8, label=agreements.columns[i], color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.3 else 'darkgrey'
        labels = [round(l*100) if l > 0.05 else '' for l in lengths]
        ax.bar_label(rects, labels=labels, fmt='%.2f', label_type='center', color=text_color)
    ymin = np.arange(0, len(categories), 1) - .3
    model1 = agreements.columns[-2]
    model2 = agreements.columns[-1]    
    ax.vlines(agreements[model2], ymin=ymin, ymax=ymin+.6, color='white', linewidth=6, label=model2)
    ax.vlines(agreements[model1], ymin=ymin, ymax=ymin+.6, color='black', linewidth=2, label=model1)
    ax.legend(ncol=6, loc='upper center')
    ax.set_xlabel('% Agreements')
    plt.show()


def plot_predicted_label_counts(predictions, references, models):
    sum_pred = {k: v.sum(axis=0) for (k, v) in predictions.items() if k in models}
    sum_pred["ground"] = references.sum(axis=0)
    sum_pred["label"] = categories
    sum_pred = pd.DataFrame(sum_pred)
    sum_pred = sum_pred.melt(id_vars='label').sort_values('value')
    sns.barplot(data=sum_pred, x="value", y="label", hue="variable").set(title='Predicted Labels')
    plt.show()


def calculate_f1_scores(predictions, references):
    scores = pd.DataFrame({
        "Method": predictions.keys(),
        "MicroF1": [f1_score(references, p, average="micro") for p in predictions.values()],
        "MacroF1": [f1_score(references, p, average="macro") for p in predictions.values()],
    })
    return scores.sort_values(by="MicroF1", ascending=False)


def per_label_accuracy(predictions, references):
    predictions, references = np.array(predictions), np.array(references)
    per_label_accuracy = np.sum(predictions == references, axis=0) / len(references)
    print("Values:", sorted(per_label_accuracy))

    match_pred_ref = predictions == references
    correct = []
    label_names = []
    for c in range(len(categories)):
        correct.append(match_pred_ref[:, c])
        label_names.append([categories[c]] * len(predictions))
    correct = np.concatenate(correct)
    label_names = np.concatenate(label_names)
    df_correct_pred = pd.DataFrame({"correct": correct, "label_name": label_names})

    order = sorted(range(len(categories)), key=lambda i: per_label_accuracy[i])
    return sns.barplot(x="correct", y="label_name", data=df_correct_pred, order=np.array(categories)[order])


def per_label_f1(predictions, references):
    f1 = f1_score(references, predictions, average=None)
    print("f1:", f1)
    micro_f1 = f1_score(references, predictions, average="micro")
    print("micro-f1:", micro_f1)
    macro_f1 = f1_score(references, predictions, average="macro")
    print("macro-f1:", macro_f1)

    correct = []
    label_names = []
    for c in range(len(categories)):
        correct.append(f1[c])
        label_names.append(categories[c])
    correct = np.array(correct)
    label_names = np.array(label_names)
    df_correct_pred = pd.DataFrame({"f1_score": correct, "label_name": label_names})

    order = sorted(range(len(categories)), key=lambda i: f1[i])
    return sns.barplot(x="f1_score", y="label_name", data=df_correct_pred, order=np.array(categories)[order])
# }}} Plotting and Scoring #

# Utility Functions {{{ #
def print_sample(text, labels, predicted_labels=None):
    print(text)
    print(vec_to_label_names(labels))
    if predicted_labels:
        print(vec_to_label_names(predicted_labels))
    print()


def vec_to_label_names(vec):
    names = []
    for i, x in enumerate(vec):
        if x:
            names.append(categories[i])
    return ','.join(names)
# }}} Utility Functions #
