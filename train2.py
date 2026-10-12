import os
import torch
import torch.nn as nn
from torch.optim import AdamW

from model import DiffusionModelConfig, HybridUMambaDiffusion
from scheduler import DDPMScheduler
from test import testing_model, evaluate_test_metrics

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(net, loader, opt, noise_scheduler, loss_fn, dev):
    net.train()
    running_loss = 0.0

    for x_clean, cond in loader:
        x_clean = x_clean.to(dev)
        cond = cond.to(dev)
        b = x_clean.size(0)

        eps = torch.randn_like(x_clean)
        t = torch.randint(0, noise_scheduler.num_timesteps, (b,), device=dev).long()

        x_noisy = noise_scheduler.add_noise(x_clean, eps, t)

        opt.zero_grad()
        eps_pred = net(x_noisy, t, cond)

        loss = loss_fn(eps_pred, eps)
        loss.backward()

        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(net, loader, noise_scheduler, loss_fn, dev):
    net.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x_clean, cond in loader:
            x_clean = x_clean.to(dev)
            cond = cond.to(dev)
            b = x_clean.size(0)

            eps = torch.randn_like(x_clean)
            t = torch.randint(0, noise_scheduler.num_timesteps, (b,), device=dev).long()

            x_noisy = noise_scheduler.add_noise(x_clean, eps, t)
            eps_pred = net(x_noisy, t, cond)

            loss = loss_fn(eps_pred, eps)
            val_loss += loss.item()

    return val_loss / len(loader)


def main():
    epochs = 20
    lr = 2e-4
    wd = 1e-4
    seq_len = 256
    save_path = "checkpoints/best_mamba_diffusion.pt"

    os.makedirs("checkpoints", exist_ok=True)

    # Need to plug loaders here
    train_loader = None
    val_loader = None
    test_loader = None


    # Model architecture setup
    cfg = DiffusionModelConfig(
        in_channels=3,
        hidden_dim=128,
        cond_dim=3,
        d_state=16,
        d_conv=4,
        expand=2
    )
    model = HybridUMambaDiffusion(cfg).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()
    scheduler = DDPMScheduler(num_timesteps=1000, device=device)

    print(f"Using device: {device} | Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        tr_loss = train_model(model, train_loader, optimizer, scheduler, criterion, device)
        va_loss = validate(model, val_loader, scheduler, criterion, device)

        print(f"[{epoch:02d}/{epochs:02d}] train: {tr_loss:.4f} | val: {va_loss:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), save_path)
            print(f"  --> saved checkpoint to {save_path} (val: {best_val:.4f})")

    # Evaluation on test split
    print("\nLoading best weights for sampling...")
    model.load_state_dict(torch.load(save_path, map_location=device))

    preds = testing_model(
        model=model,
        scheduler=scheduler,
        test_dataloader=test_loader,
        seq_length=seq_len,
        device=device
    )

    targets = torch.cat([batch[0] for batch in test_loader], dim=0)
    evaluate_test_metrics(preds, targets)


if __name__ == "__main__":
    main()