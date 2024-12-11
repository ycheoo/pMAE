import re
import os
import csv
import shutil
import numpy as np

import json
import argparse

from src.utils.conf import log_path, result_path

meta_source_dir = f"{log_path()}/sync"
meta_target_dir = f"{result_path()}/sync"
seeds = ["seed2023", "seed2024", "seed2025"]


def task_number(file_path):
    match = re.search(r"task_(\d+)_cmt.npy", file_path)
    if match:
        return int(match.group(1))
    return 0


def list_files(dir):
    all_files = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            all_files.append(os.path.join(root, file))
    all_files.sort(key=task_number)
    return all_files


def arrange_files(communication_epoch, datasets, models):
    def extract_epoch_number(file):
        start_str = "epoch_"
        end_str = "_cmt"

        start_idx = file.find(start_str)
        end_idx = file.find(end_str)

        if start_idx != -1 and end_idx != -1:
            number_str = file[start_idx + len(start_str) : end_idx]
            return int(number_str)
        else:
            return None

    def move_and_rename_files(communication_epoch_ptask, files, target_dir):
        for file in files:
            original_name = os.path.basename(file)
            if "task" in original_name:
                new_name = original_name
            else:
                epoch_number = extract_epoch_number(original_name)
                if (epoch_number + 1) % communication_epoch_ptask != 0:
                    continue
                task_number = epoch_number // communication_epoch_ptask
                new_name = f"task_{task_number}_cmt.npy"
            target_path = os.path.join(target_dir, new_name)
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy(file, target_path)

    for dataset, [n_task, n_class_ptask] in datasets.items():
        communication_epoch_ptask = communication_epoch // n_task
        for model in models:
            exp_dir = os.path.join(meta_source_dir, dataset, model)
            target_dir = os.path.join(meta_target_dir, dataset, model)
            for seed in seeds:
                cmts_dir = os.path.join(exp_dir, seed, "cmts")
                cmts_files = list_files(cmts_dir)
                target_dir_cur = os.path.join(target_dir, seed)
                move_and_rename_files(
                    communication_epoch_ptask, cmts_files, target_dir_cur
                )


def calculate_results(datasets, models):
    def write_acc_csv(acc_list_path, acc_list):
        with open(acc_list_path, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["n_class", "acc"])
            for index, acc_value in enumerate(acc_list):
                csvwriter.writerow([(index + 1) * n_class_ptask, acc_value])

    for dataset, [n_task, n_class_ptask] in datasets.items():
        for model in models:
            exp_dir = os.path.join(meta_target_dir, dataset, model)
            print(exp_dir)
            for seed in seeds:
                task_acc_list = []
                class_acc_list = []
                acc_matrix = np.zeros((n_task, n_task))
                cmts_dir = os.path.join(exp_dir, seed)
                cmts_files = list_files(cmts_dir)
                for task_i in range(n_task):
                    cmt = np.load(cmts_files[task_i])
                    for task_j in range(task_i + 1):
                        class_indices = np.arange(
                            task_j * n_class_ptask, (task_j + 1) * n_class_ptask
                        )
                        correct = np.sum(cmt[class_indices, class_indices])
                        total = np.sum(cmt[class_indices, :])
                        acc = correct / total
                        acc_matrix[task_i][task_j] = acc
                    class_acc_list.append(np.sum(np.diag(cmt)) / np.sum(cmt))
                    task_acc_list.append(np.mean(acc_matrix[task_i][: task_i + 1]))
                task_acc_list_path = os.path.join(exp_dir, f"task_{seed}acc.csv")
                class_acc_list_path = os.path.join(exp_dir, f"class_{seed}acc.csv")
                write_acc_csv(task_acc_list_path, task_acc_list)
                write_acc_csv(class_acc_list_path, class_acc_list)
                acc_matrix_path = os.path.join(exp_dir, f"{seed}amt.npy")
                np.save(acc_matrix_path, acc_matrix)


def collect_statistics(datasets, models):
    def calculate_acc(acc_csv_path):
        acc_values = []
        with open(acc_csv_path, "r") as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)
            for row in csvreader:
                acc_values.append(float(row[1]))
        averaged_acc = sum(acc_values) / len(acc_values)
        last_acc = acc_values[-1]
        return averaged_acc, last_acc

    def calculate_forgetting(amt_npy_path, n_task):
        amt = np.load(amt_npy_path)
        forgetting_ptask = []
        for task in range(n_task):
            forgetting_ptask.append(np.max(amt[:, task]) - amt[n_task - 1, task])
        averaged_forgetting = sum(forgetting_ptask) / len(forgetting_ptask)
        return averaged_forgetting

    def get_statistics(values):
        average = sum(values) / len(values)
        variance = np.std(values)
        return average, variance

    def write_csv(csv_file_path, model, value, variance):
        with open(csv_file_path, "a", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            if os.path.getsize(csv_file_path) == 0:
                csvwriter.writerow(["model", "value", "variance"])
            csvwriter.writerow([model, value, variance])

    for dataset, [n_task, n_class_ptask] in datasets.items():
        dataset_dir = os.path.join(meta_target_dir, dataset)
        metric_avgacc_path = os.path.join(dataset_dir, "metric_avgacc.csv")
        metric_lastacc_path = os.path.join(dataset_dir, "metric_lastacc.csv")
        metric_forgetting_path = os.path.join(dataset_dir, "metric_forgetting.csv")
        for model in models:
            exp_dir = os.path.join(dataset_dir, model)
            avgacc_list = []
            lastacc_list = []
            forgetting_list = []
            for seed in seeds:
                acc_csv_path = os.path.join(exp_dir, f"task_{seed}acc.csv")
                averaged_acc, last_acc = calculate_acc(acc_csv_path)
                avgacc_list.append(averaged_acc)
                lastacc_list.append(last_acc)
                amt_npy_path = os.path.join(exp_dir, f"{seed}amt.npy")
                forgetting_list.append(calculate_forgetting(amt_npy_path, n_task))
            avgacc_average, avgacc_variance = get_statistics(avgacc_list)
            lastacc_average, lastacc_variance = get_statistics(lastacc_list)
            forgetting_average, forgetting_variance = get_statistics(forgetting_list)
            print(avgacc_average, avgacc_variance)
            print(lastacc_average, lastacc_variance)
            print(forgetting_average, forgetting_variance)
            write_csv(metric_avgacc_path, model, avgacc_average, avgacc_variance)
            write_csv(metric_lastacc_path, model, lastacc_average, lastacc_variance)
            write_csv(
                metric_forgetting_path, model, forgetting_average, forgetting_variance
            )


def parse_args():
    parser = argparse.ArgumentParser("Federated continual learning", add_help=False)

    parser.add_argument("--model", default="fppl", type=str, help="Experiment model")
    parser.add_argument("--dataset", default="imagenetr", type=str, help="Dataset")
    args = parser.parse_args()
    return args


def load_json(setting_path):
    with open(setting_path) as f:
        param = json.load(f)
    return param


def read_configs(dataset, model):
    exp_dir = os.path.join(meta_source_dir, dataset, model)
    config_file = os.path.join(exp_dir, seeds[0], "configs.json")
    config_param = load_json(os.path.join(config_file))
    communication_epoch = config_param["communication_epoch"]
    n_task = config_param["n_task"]
    n_class_ptask = config_param["inc_class"]
    return communication_epoch, n_task, n_class_ptask


def main():
    args = parse_args()
    communication_epoch, n_task, n_class_ptask = read_configs(args.dataset, args.model)
    args.dataset = {args.dataset: [n_task, n_class_ptask]}
    arrange_files(communication_epoch, args.dataset, [args.model])
    calculate_results(args.dataset, [args.model])
    collect_statistics(args.dataset, [args.model])


if __name__ == "__main__":
    main()
