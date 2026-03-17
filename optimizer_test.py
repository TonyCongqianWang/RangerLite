import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import itertools

from ranger21 import Ranger21
from ranger_lite import RangerLite

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def verify_optimizers_match(seed, use_weight_decay=True):
    print(f"START verify_optimizers_match {seed=} {use_weight_decay=}")
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_old = ToyModel()
    model_new_0 = copy.deepcopy(model_old)
    model_new_1 = copy.deepcopy(model_old)

    # Create explicit parameter groups
    def get_param_groups(model):
        return [
            {"params": model.conv1.parameters(), "lr": 1.5, "weight_decay": 1e-4 if use_weight_decay else 0.0},
            {"params": model.bn1.parameters(), "lr": 1.0, "weight_decay": 0.0 if use_weight_decay else 0.0},
            {"params": model.fc1.parameters(), "lr": 0.5, "weight_decay": 1e-5 if use_weight_decay else 0.0},
            {"params": model.fc2.parameters(), "lr": 1.0, "weight_decay": 1e-2 if use_weight_decay else 0.0},
        ]

    # Initialize OLD Optimizer
    # (Assuming `Ranger21` is defined in your environment)
    opt_old  = Ranger21(
            get_param_groups(model_old),
            lr=1.0,
            betas=(0.9, 0.999),
            eps=1.0e-7,
            using_gc=False,
            using_normgc=False,
            weight_decay=0.0,
            num_batches_per_epoch=1024,
            num_epochs=800,
            warmdown_active=False,
            use_warmup=False,
            use_adaptive_gradient_clipping=False,
            softplus=False,
            pnm_momentum_factor=0.0,
        )

    opt_new_0 = RangerLite(
        get_param_groups(model_new_0),
        lr=1.0,
        weight_decay=0.0,
        use_legacy_scoping_bug=False,
    )
    opt_new_1 = RangerLite(
        get_param_groups(model_new_1),
        lr=1.0,
        weight_decay=0.0,
        use_legacy_scoping_bug=True,
    )

    optimizers = [opt_old, opt_new_0, opt_new_1]

    models = {
        "ranger21" : model_old, "unbloated" : model_new_0, "unbloated_legacy" : model_new_1
    }

    batch_size = 8
    num_steps = 32
    inputs = torch.randn(num_steps, batch_size, 3, 16, 16)
    targets = torch.randint(0, 10, (num_steps, batch_size))
    criterion = nn.CrossEntropyLoss()

    print("Running step-by-step evaluation...")

    # Pair them up
    pairs = list(zip(optimizers, models.items()))

    for step in range(num_steps):
        x = inputs[step]
        y = targets[step]

        losses = []

        for opt, (name, model) in pairs:
            opt.zero_grad()
            loss = criterion(model(x), y)  # Now stepping the correct model
            loss.backward()
            opt.step()
            losses.append(loss)

        loss_diffs = [
            abs(a.item() - b.item())
            for a, b in itertools.combinations(losses, 2)
        ]

        print(f"Step {step + 1:2d} | Diff (Old vs New0): {loss_diffs[0]:.8f} | Diff (Old vs New1): {loss_diffs[1]:.8f} | Diff (New0 vs New1): {loss_diffs[2]:.8f}")


    for pair in itertools.combinations(models.items(), 2):
        ((name_a, model_a), (name_b, model_b)) = pair
        print(f"\nVerifying parameter state parity ({name_a} vs {name_b})...")
        all_match = True
        for (name, p_a), (_, p_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
            is_close = torch.allclose(p_a, p_b, atol=1e-6, rtol=1e-5)
            if not is_close:
                all_match = False
                max_diff = torch.max(torch.abs(p_a - p_b)).item()
                print(f"Mismatch in {name}: Max diff = {max_diff:.8f}")

        if all_match:
            print("\nSUCCESS: Both optimizers yield mathematically identical results across diverse parameter groups.")

        print(32 * "#")
    print(32 * "#")
    print(32 * "#")
    print(32 * "#")


if __name__ == "__main__":
    verify_optimizers_match(0)
    verify_optimizers_match(1)
    verify_optimizers_match(2)
    verify_optimizers_match(3)
    verify_optimizers_match(0, False)
    verify_optimizers_match(1, False)
    verify_optimizers_match(2, False)
    verify_optimizers_match(3, False)