import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from Diffusion_Framework_For_Earthquake_Waveform_Generation.model import DiffusionModelConfig, HybridMambaDiffusionModel
from test import testing_model
from scheduler import DDPMScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_model(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:

        clean_data, cond = batch
        clean_data, cond = clean_data.to(device), cond.to(device)

        noise = torch.randn_like(clean_data)
        bsz = clean_data.shape[0]
        timesteps = torch.randint(
            0, scheduler.num_timesteps, (bsz,), device=device).long()

        noisy_data = scheduler.add_noise(clean_data, noise, timesteps)

        optimizer.zero_grad()
        noise_pred = model(noisy_data, timesteps, cond)

        loss = criterion(noise_pred, noise)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, scheduler, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            clean_data, cond = batch
            clean_data, cond = clean_data.to(device), cond.to(device)

            noise = torch.randn_like(clean_data)
            bsz = clean_data.shape[0]
            timesteps = torch.randint(
                0, scheduler.num_timesteps, (bsz,), device=device).long()

            noisy_data = scheduler.add_noise(clean_data, noise, timesteps)
            noise_pred = model(noisy_data, timesteps, cond)

            loss = criterion(noise_pred, noise)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 5
    seq_length = 256

    config = DiffusionModelConfig(
        in_channels=3,
        cond_channels=7,
        hidden_channels=128,
        d_state=128,
        d_conv=16,
        expand=2
    )

    model = HybridMambaDiffusionModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    scheduler = DDPMScheduler(num_timesteps=1000, device=device)

    # DataLoaders
    train_dataloader = ...  # Training data idhar daal
    val_dataloader = ...   # Validation data idhar daal
    # Testing data idhar daal (Should yield condition vectors)
    test_dataloader = ...

    print(f"Starting training on {device} for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        train_loss = train_model(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device
        )

        val_loss = evaluate_model(
            model=model,
            dataloader=val_dataloader,
            scheduler=scheduler,
            criterion=criterion,
            device=device
        )

        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    print("\nStarting generation (Testing)...")

    generated_samples = testing_model(
        model=model,
        scheduler=scheduler,
        cond_batches=test_dataloader,
        seq_length=seq_length,
        device=device
    )

    print(f"Generation complete. Output shape: {generated_samples.shape}")

    print("Training complete. Saving model...")
    torch.save(model.state_dict(), "mamba_diffusion_weights.pth")


if __name__ == "__main__":
    main()
