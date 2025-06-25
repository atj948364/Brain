from .utils import Dice_coef, Jaccard_coef, Accuracy, Recall, save_results
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import copy
from tqdm import tqdm
from plot import plot_metrics
import os
# from torchsummary import summary  # 新增导入
from torchinfo import summary  # 替换 torchsummary

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model, input_size, device):
    """打印模型结构和参数量"""
    print("\n模型结构:")
    model.to(device)
    # 将 torch.device 对象转换为字符串
    device_str = str(device)
    # 添加 batch_size = 1 作为第一维
    summary(model, input_size=(1,) + input_size, device=device.type)
    # summary(model, input_size=input_size, device=device_str)
    print(f"总参数量: {count_parameters(model):,}")
    print("-" * 50)

def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    dataloaders,
    dataset_sizes,
    device,
    save_path=None,
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    train_epoch_losses = []
    valid_epoch_losses = []
    train_epoch_dices = []
    valid_epoch_dices = []
    train_epoch_jaccs = []
    valid_epoch_jaccs = []
    train_epoch_accuracies = []
    valid_epoch_accuracies = []
    train_epoch_recalls = []
    valid_epoch_recalls = []

    epochs_lr_list = []

    dice_cf = Dice_coef()
    jaccard_cf = Jaccard_coef()
    accuracy_cf = Accuracy()
    recall_cf = Recall()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_dice = 0.0
            running_jacc = 0.0
            running_accuracy = 0.0
            running_recall = 0.0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)

                    loss = criterion(outputs, labels)
                    jacc = jaccard_cf(torch.round(outputs), labels)
                    dice = dice_cf(torch.round(outputs), labels)
                    accuracy = accuracy_cf(torch.round(outputs), labels)
                    recall = recall_cf(torch.round(outputs), labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_dice += dice.item() * inputs.size(0)
                running_jacc += jacc.item() * inputs.size(0)
                running_accuracy += accuracy.item() * inputs.size(0)
                running_recall += recall.item() * inputs.size(0)

            if phase == "train":
                scheduler.step()
                print("LR:{}".format(optimizer.param_groups[0]["lr"]))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_dice = running_dice / dataset_sizes[phase]
            epoch_jacc = running_jacc / dataset_sizes[phase]
            epoch_accuracy = running_accuracy / dataset_sizes[phase]
            epoch_recall = running_recall / dataset_sizes[phase]

            # print(
            #     "{}:\t Loss: {:.4f}\t Dice: {:.4f}\t Jacc: {:.4f}".format(
            #         phase, epoch_loss, epoch_dice, epoch_jacc
            #     )
            # )
            print(
                "{}:\t Loss: {:.4f}\t Dice: {:.4f}\t Jacc: {:.4f}\t Accuracy: {:.4f}\t Recall: {:.4f}".format(
                    phase, epoch_loss, epoch_dice, epoch_jacc, epoch_accuracy, epoch_recall
                )
            )

            if phase == "train":
                train_epoch_losses.append(epoch_loss)
                train_epoch_dices.append(epoch_dice)
                train_epoch_jaccs.append(epoch_jacc)
                train_epoch_accuracies.append(epoch_accuracy)
                train_epoch_recalls.append(epoch_recall)

            elif phase == "val":
                valid_epoch_losses.append(epoch_loss)
                valid_epoch_dices.append(epoch_dice)
                valid_epoch_jaccs.append(epoch_jacc)
                valid_epoch_accuracies.append(epoch_accuracy)
                valid_epoch_recalls.append(epoch_recall)

            epochs_lr_list.append(optimizer.param_groups[0]["lr"])

            # deep copy the model
            if phase == "val" and epoch_loss <= best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

        if (save_path is not None) and ((epoch + 1) % 5 == 0):
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                },
                os.path.join(save_path, "model_{}.pt".format(epoch + 1)),
            )

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Loss: {:4f}".format(best_loss))

    epoch_losses = {"train": train_epoch_losses, "val": valid_epoch_losses}
    epoch_dices = {"train": train_epoch_dices, "val": valid_epoch_dices}
    epoch_jaccs = {"train": train_epoch_jaccs, "val": valid_epoch_jaccs}
    epoch_accuracies = {"train": train_epoch_accuracies, "val": valid_epoch_accuracies}
    epoch_recalls = {"train": train_epoch_recalls, "val": valid_epoch_recalls}

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, epoch_losses, epoch_dices, epoch_jaccs, epoch_accuracies, epoch_recalls


def test_model(model, criterion, dataloaders, dataset_sizes, device):

    model.eval()

    dice_cf = Dice_coef()
    jaccard_cf = Jaccard_coef()
    accuracy_cf = Accuracy()
    recall_cf = Recall()

    total_test_loss = 0
    total_test_dice = 0
    total_test_jacc = 0
    total_test_accuracy = 0
    total_test_recall = 0

    with torch.no_grad():

        for inputs, labels in tqdm(dataloaders["test"]):

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            dice = dice_cf(torch.round(outputs), labels)
            jacc = jaccard_cf(torch.round(outputs), labels)
            accuracy = accuracy_cf(torch.round(outputs), labels)
            recall = recall_cf(torch.round(outputs), labels)

            total_test_loss += loss.item() * inputs.size(0)
            total_test_dice += dice.item() * inputs.size(0)
            total_test_jacc += jacc.item() * inputs.size(0)
            total_test_accuracy += accuracy.item() * inputs.size(0)
            total_test_recall += recall.item() * inputs.size(0)

    test_loss = total_test_loss / dataset_sizes["test"]
    test_dice = total_test_dice / dataset_sizes["test"]
    test_jacc = total_test_jacc / dataset_sizes["test"]
    test_accuracy = total_test_accuracy / dataset_sizes["test"]
    test_recall = total_test_recall / dataset_sizes["test"]
    print("\n\nResults on Test Set:")
    print("-" * 22)
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Dice: {test_dice:.4f}")
    print(f"  Jaccard: {test_jacc:.4f}")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  Recall: {test_recall:.4f}")


def train_and_evaluate(
    model,
    dataloaders,
    dataset_sizes,
    model_type,
    saves_path,
    weights_path,
    plots_path,
    num_epochs,
    lr,
    weight_decay,
    step_size,
    gamma,
    device,
):
    """
    Trains and evaluates a given model with specified hyperparameters.
    """
    print(f"\nTraining {model.__class__.__name__}...\n")

    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_sched = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    model_ft, epoch_losses, epoch_dices, epoch_jaccs, epoch_accuracies, epoch_recalls = train_model(
        model,
        criterion,
        optimizer,
        lr_sched,
        num_epochs,
        dataloaders,
        dataset_sizes,
        device,
        weights_path,
    )

    # Save training results
    save_results(
        saves_path, weights_path, model_ft, epoch_losses, epoch_dices, epoch_jaccs, epoch_accuracies, epoch_recalls
    )

    # Plot evaluation metrics
    plot_metrics(epoch_losses, "Loss", "Epoch Losses", model_type, plots_path)
    plot_metrics(epoch_dices, "Dice", "Epoch Dices", model_type, plots_path)
    plot_metrics(epoch_jaccs, "Jaccard", "Epoch Jaccards", model_type, plots_path)
    plot_metrics(epoch_accuracies, "Accuracy", "Epoch Accuracies", model_type, plots_path)
    plot_metrics(epoch_recalls, "Recall", "Epoch Recalls", model_type, plots_path)

    test_model(model_ft, criterion, dataloaders, dataset_sizes, device)

    # return model_ft
