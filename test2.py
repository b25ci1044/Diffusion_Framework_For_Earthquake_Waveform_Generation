import torch


@torch.no_grad()
def testing_model(net, noise_scheduler, loader, seq_len=256, dev="cuda"):
    net.eval()
    all_preds = []

    print(f"--> sampling {seq_len}-length waveforms via reverse diffusion...")

    for batch_idx, batch in enumerate(loader):
        c = batch[1] if isinstance(batch, (list, tuple)) else batch
        c = c.to(dev)
        bsz = c.size(0)

        xt = torch.randn(bsz, 3, seq_len, device=dev)

        for step in reversed(range(noise_scheduler.num_timesteps)):
            t = torch.full((bsz,), step, device=dev, dtype=torch.long)

            pred_noise = net(xt, t, c)
            xt = noise_scheduler.step(pred_noise, step, xt)

        all_preds.append(xt.cpu())

    return torch.cat(all_preds, dim=0)


def evaluate_test_metrics(y_pred, y_true):
    y_pred, y_true = y_pred.float(), y_true.float()

    mse_val = torch.mean((y_pred - y_true) ** 2).item()

    peak_pred = torch.max(torch.abs(y_pred), dim=-1).values
    peak_true = torch.max(torch.abs(y_true), dim=-1).values

    eps = 1e-6
    pga_diff = torch.abs(peak_pred - peak_true) / (peak_true + eps)
    pga_err = torch.mean(pga_diff).item() * 100.0

    print("\n" + "=" * 38)
    print("           EVALUATION METRICS         ")
    print("=" * 38)
    print(f"MSE Loss       : {mse_val:.4f}")
    print(f"Mean PGA Error : {pga_err:.2f}%")
    print("=" * 38 + "\n")

    return {"mse": mse_val, "pga_err": pga_err}