import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch

class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self, input_tokens, input_ids, label, group):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label
        self.group = group


class TextDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        args,
        cwe_label_map,
        group_label_map,
        file_type="train",
        dataset="json",
    ):
        if file_type == "train":
            file_path = args.train_data_file
        elif file_type == "eval":
            file_path = args.eval_data_file
        elif file_type == "test":
            file_path = args.test_data_file
        self.examples = []
        if dataset == "csv":
            df = pd.read_csv(file_path)
            funcs = df["func_before"].tolist()
            labels = df["CWE ID"].tolist()
            groups = df["cwe_abstract_group"].tolist()
        elif dataset == "json":
            df = pd.read_json(file_path)
            funcs = df["func"].tolist()
            labels = df["cwe"].tolist()
            df = pd.read_json("../../data/cwe_description.json")
            groups = [df[cwe][2] for cwe in labels]

        for i in tqdm(range(len(funcs))):
            label = cwe_label_map[labels[i]][1]
            group_label = group_label_map[groups[i]]
            self.examples.append(
                convert_examples_to_features(
                    funcs[i], label, group_label, tokenizer, args
                )
            )
        if file_type == "train":
            self.cwe_label_map = cwe_label_map

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (
            torch.tensor(self.examples[i].input_ids),
            torch.tensor(self.examples[i].label).float(),
            torch.tensor(self.examples[i].group).float(),
        )


def convert_examples_to_features(func, label, group_label, tokenizer, args):
    # source
    code_tokens = tokenizer.tokenize(str(func))[: args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, label, group_label)
