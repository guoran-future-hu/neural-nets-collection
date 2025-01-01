import os
import glob
import torch

def save_best_model(curr_metric, best_metric, model, experiment_name, epoch, metric_name="metric", mode="max"):
    """
    Updates the best model based on the current performance metric.

    Parameters:
    - curr_metric (float): Current metric value (e.g., accuracy, sensitivity, loss).
    - best_metric (float): The best metric value so far.
    - model (torch.nn.Module): The PyTorch model.
    - experiment_name (str): The name of the experiment.
    - epoch (int): The current epoch number.
    - metric_name (str): The name of the metric (default is "metric").
    - mode (str): Whether to "max" or "min" the metric (default is "max").

    Returns:
    - best_metric (float): Updated best metric value.
    """
    if (mode == "max" and curr_metric >= best_metric) or (mode == "min" and curr_metric <= best_metric):
        best_metric = curr_metric
        
        prev_best_model = glob.glob(f'experiments/{experiment_name}/best_model_{metric_name}*.pt')
        if len(prev_best_model) > 0:
            os.remove(prev_best_model[0])
        
        torch.save(model.state_dict(), f'experiments/{experiment_name}/best_model_{metric_name}_{epoch + 1}_epoch.pt')

    return best_metric


class EarlyStopManager:
    def __init__(self, patience, mode, min_delta=0, verbose=False):
        """
        Args:
            patience (int): Number of epochs to wait before early stopping
            min_delta (float): Minimum change in monitored value to qualify as an improvement
            mode (str): One of {'min', 'max'}. In 'min' mode, training stops when metric stops decreasing;
                       in 'max' mode, training stops when metric stops increasing
            verbose (bool): If True, prints improvement messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.epochs_no_improve = 0
        self.mode = mode.lower()
        
        if self.mode not in ['min', 'max']:
            raise ValueError("Only 'min' and 'max' modes are supported")
            
        # Initialize best score based on mode
        self.best_score = float('inf') if self.mode == 'min' else float('-inf')
        
    def _is_improvement(self, current_score):
        if self.mode == 'min':
            return current_score < (self.best_score - self.min_delta)
        return current_score > (self.best_score + self.min_delta)
    
    def update(self, current_score):
        """
        Returns True if training should stop, False otherwise
        """
        if self._is_improvement(current_score):
            if self.verbose:
                print(f"Metric {'decreased' if self.mode == 'min' else 'increased'} "
                      f"from {self.best_score:.6f} to {current_score:.6f}")
            self.best_score = current_score
            self.epochs_no_improve = 0
            return False
            
        self.epochs_no_improve += 1
        if self.verbose:
            print(f"Metric did not improve. {self.patience - self.epochs_no_improve} "
                  f"epochs remaining until early stopping.")
            
        if self.epochs_no_improve >= self.patience:
            print("Early stopping triggered.")
            return True
            
        return False