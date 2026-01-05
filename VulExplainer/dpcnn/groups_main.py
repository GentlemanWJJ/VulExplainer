from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import pickle
import random
import numpy as np
import torch
import json
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup, RobertaTokenizer, RobertaModel
from torch.optim import AdamW

from tqdm import tqdm
from groups_model import GroupModel
from TextDataset import TextDataset
import pandas as pd

# metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, eval_dataset):
    """Train the model"""

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, num_workers=4, shuffle=True
    )

    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
    )

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    early_stopping_patience = getattr(args, "early_stopping_patience", 3)
    early_stopping_metric = "eval_acc"
    patience_counter = 0
    best_acc = 0
    best_model_state_dict = None

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d",
        args.train_batch_size // max(args.n_gpu, 1),
    )
    logger.info(
        "  Total train batch size = %d",
        args.train_batch_size * args.gradient_accumulation_steps,
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    logger.info("  Early stopping patience = %d", early_stopping_patience)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0

    model.zero_grad()

    stop_training = False
    for idx in range(args.epochs):
        if stop_training:
            logger.info("Early stopping triggered. End training at epoch %d.", idx)
            break
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            (input_ids, labels, groups) = [x.to(args.device) for x in batch]
            model.train()
            loss = model(input_ids=input_ids, labels=labels, groups=groups)
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss

            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(
                    np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4
                )

                if global_step % args.save_steps == 0:
                    results = evaluate(
                        args, model, tokenizer, eval_dataset, eval_when_training=True
                    )

                    # Early stopping logic
                    cur_acc = results.get(early_stopping_metric, 0)
                    if cur_acc > best_acc:
                        best_acc = cur_acc
                        patience_counter = 0
                        logger.info("  " + "*" * 20)
                        logger.info("  Best Acc:%s", round(best_acc, 4))
                        logger.info("  " + "*" * 20)
                        # Save model checkpoint
                        checkpoint_prefix = "checkpoint-best-acc"
                        output_dir = os.path.join(
                            args.output_dir, "{}".format(checkpoint_prefix)
                        )
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )
                        output_dir = os.path.join(
                            output_dir, "{}".format(args.model_name)
                        )
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                    else:
                        patience_counter += 1
                        logger.info(
                            "Early stopping patience counter: %d/%d",
                            patience_counter,
                            early_stopping_patience,
                        )
                        if patience_counter >= early_stopping_patience:
                            logger.info(
                                "Early stopping: no improvement after %d checkpoints. Stopping training.",
                                early_stopping_patience,
                            )
                            stop_training = True
                            break


def evaluate(args, model, tokenizer, eval_dataset, eval_when_training=False):
    # build dataloader
    # eval_sampler = SequentialSampler(eval_dataset)
    # eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=args.eval_batch_size,num_workers=0)
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.eval_batch_size, num_workers=4, shuffle=False
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    y_preds = []
    y_trues = []
    for batch in eval_dataloader:
        (input_ids, labels, groups) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            prob = model(input_ids=input_ids, labels=None, groups=None)
            y_preds += list((np.argmax(prob.cpu().numpy(), axis=1)))
            y_trues += list((np.argmax(groups.cpu().numpy(), axis=1)))
    # calculate scores
    acc = accuracy_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds, average="weighted")
    recall = recall_score(y_trues, y_preds, average="weighted")
    f1 = f1_score(y_trues, y_preds, average="weighted")
    result = {
        "eval_acc": float(acc),
        "eval_precision": float(precision),
        "eval_recall": float(recall),
        "eval_f1": float(f1),
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def test(args, model, tokenizer, test_dataset):
    # build dataloader
    # test_sampler = SequentialSampler(test_dataset)
    # test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.eval_batch_size, num_workers=4, shuffle=False
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    y_preds = []
    y_trues = []
    for batch in test_dataloader:
        (input_ids, labels, groups) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            prob = model(input_ids=input_ids, labels=None, groups=None)
            y_preds += list((np.argmax(prob.cpu().numpy(), axis=1)))
            y_trues += list((np.argmax(groups.cpu().numpy(), axis=1)))
    # calculate scores
    acc = accuracy_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds, average="weighted")
    recall = recall_score(y_trues, y_preds, average="weighted")
    f1 = f1_score(y_trues, y_preds, average="weighted")
    result = {
        "test_accuracy": float(acc),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
    }

    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return y_trues, y_preds


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str)
    parser.add_argument("--eval_data_file", default=None, type=str)
    parser.add_argument("--test_data_file", default=None, type=str)
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--use_logit_adjustment", action="store_true", default=False, help=""
    )
    parser.add_argument(
        "--use_focal_loss",
        action="store_true",
        default=False,
        help="Whether to use focal loss",
    )
    parser.add_argument(
        "--tau", default=1, type=float, help="The initial learning rate for Adam."
    )
    ## Other parameters
    parser.add_argument(
        "--model_type",
        default="bert",
        type=str,
        help="The model architecture to be fine-tuned.",
    )
    parser.add_argument(
        "--block_size",
        default=512,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )

    parser.add_argument(
        "--model_name", default="model.bin", type=str, help="Saved model name."
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--use_non_pretrained_model",
        action="store_true",
        default=False,
        help="Whether to use non-pretrained model.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_test", action="store_true", help="Whether to run eval on the dev set."
    )

    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_token_level_eval",
        default=False,
        action="store_true",
        help="Whether to do local explanation. ",
    )
    parser.add_argument(
        "--reasoning_method",
        default="attention",
        type=str,
        help="Should be one of 'attention', 'shap', 'lime', 'lig'",
    )

    parser.add_argument(
        "--train_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight deay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument("--epochs", type=int, default=1, help="training epochs")
    parser.add_argument("--hidden_size", default=256, type=int, help="hidden size.")
    parser.add_argument("--num_GNN_layers", default=2, type=int, help="num GNN layers.")
    parser.add_argument(
        "--att_op",
        default="mul",
        type=str,
        help="using attention operation for attention: mul, sum, concat",
    )
    parser.add_argument(
        "--window_size", default=3, type=int, help="window_size to build graph"
    )
    parser.add_argument(
        "--early_stopping_patience",
        default=3,
        type=int,
        help="Number of evaluations after which training will stop if no improvement.",
    )
    parser.add_argument("--dataset", default="json", type=str)

    args = parser.parse_args()
    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1
    args.device = device

    with open("../../data/big_vul/cwe_label_map.pkl", "rb") as f:
        cwe_label_map = pickle.load(f)
    group_label_map = {
        "category": [1, 0, 0, 0, 0, 0],
        "class": [0, 1, 0, 0, 0, 0],
        "variant": [0, 0, 1, 0, 0, 0],
        "base": [0, 0, 0, 1, 0, 0],
        "deprecated": [0, 0, 0, 0, 1, 0],
        "pillar": [0, 0, 0, 0, 0, 1],
    }
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        "device: %s, n_gpu: %s",
        device,
        args.n_gpu,
    )
    # Set seed
    set_seed(args)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    encoder = RobertaModel.from_pretrained(args.model_name_or_path)

    model = GroupModel(encoder=encoder, tokenizer=tokenizer, args=args, num_class=6)

    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = TextDataset(
            tokenizer,
            args,
            cwe_label_map,
            group_label_map,
            file_type="train",
            dataset=args.dataset,
        )
        eval_dataset = TextDataset(
            tokenizer,
            args,
            cwe_label_map,
            group_label_map,
            file_type="eval",
            dataset=args.dataset,
        )
        # train(args, train_dataset, groups_model, tokenizer, eval_dataset, train_dataset.cwe_label_map)
        train(
            args,
            train_dataset,
            model,
            tokenizer,
            eval_dataset,
        )
    # Evaluation
    results = {}
    if args.do_test:
        checkpoint_prefix = f"checkpoint-best-acc/{args.model_name}"
        output_dir = os.path.join(args.output_dir, "{}".format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir, map_location=args.device))
        model.to(args.device)
        test_dataset = TextDataset(
            tokenizer,
            args,
            cwe_label_map,
            group_label_map,
            file_type="test",
            dataset=args.dataset,
        )
        y_trues, y_preds = test(args, model, tokenizer, test_dataset)
    return results


if __name__ == "__main__":
    main()
