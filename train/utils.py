import torch
import torch.nn as nn
import pickle


class Dice_coef(nn.Module):

    def __init__(self):
        super(Dice_coef, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2.0 * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return dsc


class Jaccard_coef(nn.Module):

    def __init__(self):
        super(Jaccard_coef, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        jac = (intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() - intersection + self.smooth
        )
        return jac

class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()
        self.smooth = 1e-6

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        correct = (y_pred.round() == y_true).sum()
        total = y_true.size(0)
        accuracy = (correct + self.smooth) / (total + self.smooth)
        return accuracy

class Recall(nn.Module):
    def __init__(self):
        super(Recall, self).__init__()
        self.smooth = 1e-6

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        tp = (y_pred * y_true).sum()
        fn = ((1 - y_pred) * y_true).sum()
        recall = (tp + self.smooth) / (tp + fn + self.smooth)
        return recall

def save_results(
    saves_path, weights_path, model, epoch_losses, epoch_dices, epoch_jaccs, epoch_accuracies, epoch_recalls
):
    """
    Saves the model and training metrics.
    """
    torch.save(model, f"{weights_path}/best_model.pt")
    with open(f"{saves_path}/epoch_losses.pkl", "wb") as f:
        pickle.dump(epoch_losses, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"{saves_path}/epoch_dices.pkl", "wb") as f:
        pickle.dump(epoch_dices, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"{saves_path}/epoch_jaccs.pkl", "wb") as f:
        pickle.dump(epoch_jaccs, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"{saves_path}/epoch_accuracies.pkl", "wb") as f:
        pickle.dump(epoch_accuracies, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"{saves_path}/epoch_recalls.pkl", "wb") as f:
        pickle.dump(epoch_recalls, f, protocol=pickle.HIGHEST_PROTOCOL)
