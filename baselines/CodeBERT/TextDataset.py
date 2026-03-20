import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch

cwe_map={
    "CWE-77": [
        0,
        "Improper Neutralization of Special Elements used in a Command ('Command Injection')",
        "variant",[0, 0, 1, 0, 0, 0]
    ],
    "CWE-20": [1, "Improper Input Validation", "base",[0, 0, 0, 1, 0, 0]],
    "CWE-682": [2, "Incorrect Calculation", "base",[0, 0, 0, 1, 0, 0]],
    "CWE-78": [
        3,
        "Improper Neutralization of Special Elements used in an OS Command ('OS Command Injection')",
        "variant",[0, 0, 1, 0, 0, 0]
    ],
    "CWE-362": [
        4,
        "Concurrent Execution using Shared Resource with Improper Synchronization ('Race Condition')",
        "base",[0, 0, 0, 1, 0, 0],
    ],
    "CWE-754": [5, "Improper Check for Unusual or Exceptional Conditions", "base",[0, 0, 0, 1, 0, 0]],
    "CWE-476": [6, "NULL Pointer Dereference", "base",[0, 0, 0, 1, 0, 0]],
    "CWE-287": [7, "Improper Authentication", "base",[0, 0, 0, 1, 0, 0]],
    "CWE-269": [8, "Improper Privilege Management", "base",[0, 0, 0, 1, 0, 0]],
    "CWE-59": [
        9,
        "Improper Link Resolution Before File Access ('Link Following')",
        "base",[0, 0, 0, 1, 0, 0],
    ],
    "CWE-18": [10, "Improper Sanitization of Filename", "variant",[0, 0, 1, 0, 0, 0]],
    "CWE-264": [11, "Improper Access Control", "base",[0, 0, 0, 1, 0, 0]],
    "CWE-416": [12, "Use After Free", "base",[0, 0, 0, 1, 0, 0]],
    "CWE-79": [
        13,
        "Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')",
        "base",[0, 0, 0, 1, 0, 0],
    ],
    "CWE-617": [14, "Reachable Assertion", "base",[0, 0, 0, 1, 0, 0]],
    "CWE-399": [15, "Resource Exhaustion", "deprecated",[0, 0, 0, 0, 1, 0]],
    "CWE-125": [16, "Out-of-bounds Read", "base",[0, 0, 0, 1, 0, 0]],
    "CWE-22": [
        17,
        "Improper Limitation of a Pathname to a Restricted Directory ('Path Traversal')",
        "base",[0, 0, 0, 1, 0, 0],
    ],
    "CWE-285": [18, "Improper Authorization", "base",[0, 0, 0, 1, 0, 0]],
    "CWE-404": [19, "Improper Resource Shutdown or Release", "base",[0, 0, 0, 1, 0, 0]],
    "CWE-189": [20, "Numeric Errors", "deprecated",[0, 0, 0, 0, 1, 0]],
    "CWE-369": [21, "Divide By Zero", "variant",[0, 0, 1, 0, 0, 0]],
    "CWE-674": [22, "Uncontrolled Recursion", "variant",[0, 0, 1, 0, 0, 0]],
    "CWE-190": [23, "Integer Overflow or Wraparound", "base",[0, 0, 0, 1, 0, 0]],
    "CWE-415": [24, "Double Free", "base",[0, 0, 0, 1, 0, 0]],
    "CWE-835": [25, "Infinite Loop", "variant",[0, 0, 1, 0, 0, 0]],
    "CWE-17": [26, "Improper Handling of Multiple Inputs to a Function", "deprecated",[0, 0, 0, 0, 1, 0]],
    "CWE-134": [27, "Use of Externally-Controlled Format String", "base",[0, 0, 0, 1, 0, 0]],
    "CWE-19": [28, "Improper Input Handling", "deprecated",[0, 0, 0, 0, 1, 0]],
    "CWE-254": [29, "Security Features Bypass", "base",[0, 0, 0, 1, 0, 0]],
    "CWE-704": [30, "Incorrect Type Conversion or Cast", "base",[0, 0, 0, 1, 0, 0]],
    "CWE-732": [31, "Incorrect Permission Assignment for Critical Resource", "base",[0, 0, 0, 1, 0, 0]],
    "CWE-400": [
        32,
        "Uncontrolled Resource Consumption ('Resource Exhaustion')",
        "base",[0, 0, 0, 1, 0, 0],
    ],
    "CWE-311": [33, "Missing Encryption of Sensitive Data", "base",[0, 0, 0, 1, 0, 0]],
    "CWE-94": [34, "Improper Control of Generation of Code ('Code Injection')", "base",[0, 0, 0, 1, 0, 0]],
    "CWE-200": [
        35,
        "Exposure of Sensitive Information to an Unauthorized Actor",
        "base",[0, 0, 0, 1, 0, 0],
    ],
    "CWE-388": [36, "Improper Handling of Exceptional Conditions", "deprecated",[0, 0, 0, 0, 1, 0]],
    "CWE-772": [37, "Missing Release of Resource after Effective Lifetime", "variant",[0, 0, 1, 0, 0, 0]],
    "CWE-834": [38, "Excessive Iteration", "variant",[0, 0, 1, 0, 0, 0]],
    "CWE-119": [
        39,
        "Improper Restriction of Operations within the Bounds of a Memory Buffer",
        "class",[0, 1, 0, 0, 0, 0]
    ],
    "CWE-787": [40, "Out-of-bounds Write", "base",[0, 0, 0, 1, 0, 0]],
    "CWE-284": [41, "Improper Access Control", "deprecated",[0, 0, 0, 0, 1, 0]],
    "CWE-310": [42, "Cryptographic Issues", "category",[1, 0, 0, 0, 0, 0]],
    "CWE-358": [
        43,
        "Improperly Implemented Security Check for Certificate Validity or Revocation",
        "variant",[0, 0, 1, 0, 0, 0]
    ],
}

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
            cwe_labels = df["cwe"].tolist()
            vul_labels = df["vul"].tolist()
            groups = [cwe_map[cwe][3] for cwe in cwe_labels]
            if args.data_type == "vul":
                label = [[1,0] if l == 1 else [0,1] for l in vul_labels]
            elif args.data_type == "cwe":
                label = [cwe_label_map[cwe][1] for cwe in cwe_labels]

        for i in tqdm(range(len(funcs))):

            self.examples.append(
                convert_examples_to_features(
                    funcs[i], label[i], groups[i], tokenizer, args
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
