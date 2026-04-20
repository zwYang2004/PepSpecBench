"""Learning rate schedulers with warmup support.

This module provides unified scheduler implementations for the benchmark,
including warmup + cosine annealing strategy.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineScheduler(LRScheduler):
    """Cosine annealing scheduler with linear warmup.
    
    During warmup phase, learning rate increases linearly from 0 to base_lr.
    After warmup, learning rate follows cosine annealing to min_lr.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps (or epochs if step_mode='epoch')
        total_steps: Total number of training steps (or epochs)
        min_lr: Minimum learning rate at the end of cosine annealing
        last_epoch: The index of last epoch (default: -1)
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


class WarmupLinearScheduler(LRScheduler):
    """Linear decay scheduler with linear warmup.
    
    During warmup phase, learning rate increases linearly from 0 to base_lr.
    After warmup, learning rate decreases linearly to min_lr.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr: Minimum learning rate at the end
        last_epoch: The index of last epoch (default: -1)
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Linear decay
            progress = (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            linear_factor = 1.0 - progress
            return [
                self.min_lr + (base_lr - self.min_lr) * linear_factor
                for base_lr in self.base_lrs
            ]


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int = 1,
    min_lr: float = 0.0,
    step_mode: str = "epoch",
) -> LRScheduler:
    """Factory function to create a scheduler.
    
    Args:
        optimizer: Wrapped optimizer
        scheduler_type: Type of scheduler ('cosine', 'linear', 'constant')
        warmup_epochs: Number of warmup epochs
        total_epochs: Total number of training epochs
        steps_per_epoch: Number of steps per epoch (for step-level scheduling)
        min_lr: Minimum learning rate
        step_mode: 'epoch' for epoch-level, 'step' for step-level scheduling
        
    Returns:
        Configured learning rate scheduler
    """
    if step_mode == "step":
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = total_epochs * steps_per_epoch
    else:
        warmup_steps = warmup_epochs
        total_steps = total_epochs
    
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == "cosine":
        return WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr,
        )
    elif scheduler_type == "linear":
        return WarmupLinearScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr,
        )
    elif scheduler_type == "constant":
        # No decay, just warmup then constant
        return WarmupLinearScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=warmup_steps + 1,  # Effectively no decay after warmup
            min_lr=optimizer.defaults["lr"],
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Use 'cosine', 'linear', or 'constant'.")


class EarlyStopping:
    """Early stopping handler based on validation metric.
    
    Args:
        patience: Number of epochs to wait for improvement
        mode: 'max' if higher metric is better, 'min' if lower is better
        min_delta: Minimum change to qualify as improvement
    """
    
    def __init__(
        self,
        patience: int = 10,
        mode: str = "max",
        min_delta: float = 0.0,
    ) -> None:
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        
        self.best_value: Optional[float] = None
        self.best_epoch: int = 0
        self.counter: int = 0
        self.should_stop: bool = False
    
    def __call__(self, value: float, epoch: int) -> bool:
        """Check if training should stop.
        
        Args:
            value: Current validation metric value
            epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_value is None:
            self.best_value = value
            self.best_epoch = epoch
            return False
        
        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def is_best(self, value: float) -> bool:
        """Check if current value is the best so far.
        
        Args:
            value: Current validation metric value
            
        Returns:
            True if this is the best value
        """
        if self.best_value is None:
            return True
        
        if self.mode == "max":
            return value > self.best_value
        else:
            return value < self.best_value
    
    def state_dict(self) -> dict:
        """Return state for checkpointing."""
        return {
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
            "should_stop": self.should_stop,
        }
    
    def load_state_dict(self, state: dict) -> None:
        """Load state from checkpoint."""
        self.best_value = state.get("best_value")
        self.best_epoch = state.get("best_epoch", 0)
        self.counter = state.get("counter", 0)
        self.should_stop = state.get("should_stop", False)
