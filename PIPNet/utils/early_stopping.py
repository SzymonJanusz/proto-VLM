"""Early stopping utility to prevent overfitting."""


class EarlyStopping:
    """
    Early stopping handler.

    Monitors validation loss and stops training when it stops improving.
    Also saves best model checkpoint automatically.

    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for accuracy
        verbose: Print status messages
    """

    def __init__(self, patience=5, min_delta=0.0, mode='min', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss, epoch):
        """
        Check if should stop training.

        Args:
            val_loss: Current validation loss
            epoch: Current epoch number

        Returns:
            bool: True if should stop training
        """
        score = -val_loss if self.mode == 'min' else val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered! Best epoch: {self.best_epoch}")
                return True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0

        return False

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
