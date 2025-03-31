from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from contextlib import nullcontext
import pandas as pd


def empirical_est(x, y, type, order, window=0, beta=1, save_counts=False):
    assert x.size(0) == 1
    assert beta > 0

    seq_length = x.size(1)
    device = x.device
    x = x.float().squeeze()
    y = y.float().squeeze()
    powers = torch.Tensor([2**i for i in reversed(range(order))]).to(device)
    idx = F.conv1d(x.view(1, -1), powers.view(1, 1, -1)).squeeze()
    est_vec = []
    if type == "jump-markov":
        jump = seq_length // 2
        if window in range(1, jump): # Windowed add-beta estimator
            # First half of sequence
            idx1 = idx[:jump-order+1]
            y1 = y[order-1:jump]
            for i in range(2**order):
                mask = (idx1 == i)
                s = torch.stack((y1[order-1:] * mask.int(), mask.int()))
                s = F.pad(s, (window, 0))[:,:-1]
                s = F.conv1d(s, torch.ones(2, 1, window, device=device), groups=2)
                p = (s[0] + beta) / (s[1] + 2*beta)
                est_vec.append(p[mask])
            # Second half of sequence
            idx2 = idx[jump-order+1:]
            y2 = y[jump:]
            for i in range(2**order):
                mask = (idx2 == i)
                s = torch.stack((y2[order-1:] * mask.int(), mask.int()))
                s = F.pad(s, (window, 0))[:,:-1]
                s = F.conv1d(s, torch.ones(2, 1, window, device=device), groups=2)
                p = (s[0] + beta) / (s[1] + 2*beta)
                est_vec[i] = torch.cat((est_vec[i], p[mask]))
        else: # Standard add-beta estimator
            # First half of sequence
            idx1 = idx[:jump-order+1]
            y1 = y[order-1:jump]
            for i in range(2**order):
                mask = (idx1 == i)
                s = y1[mask][:-1]
                p = (s.cumsum(0) + beta) / (torch.arange(1, len(s)+1, device=device) + 2*beta)
                p = F.pad(p, (1, 0), value=0.5)
                est_vec.append(p)
            # Second half of sequence
            idx2 = idx[jump-order+1:]
            y2 = y[jump:]
            for i in range(2**order):
                mask = (idx2 == i)
                s = y2[mask][:-1]
                p = (s.cumsum(0) + beta) / (torch.arange(1, len(s)+1, device=device) + 2*beta)
                p = F.pad(p, (1, 0), value=0.5)
                est_vec[i] = torch.cat((est_vec[i], p))
    else:
        if window in range(1, seq_length-order): # Windowed add-beta estimator
            for i in range(2**order):
                mask = (idx == i)
                s = torch.stack((y[order-1:] * mask.int(), mask.int()))
                s = F.pad(s, (window, 0))[:,:-1]
                s = F.conv1d(s, torch.ones(2, 1, window, device=device), groups=2)
                p = (s[0] + beta) / (s[1] + 2*beta)
                est_vec.append(p[mask])
        else: # Standard add-beta estimator
            counts = []
            totals = []
            for i in range(2**order):
                mask = (idx == i)
                s = y[order-1:][mask][:-1]
                count = s.cumsum(0)
                count = F.pad(count, (1, 0))
                total = torch.arange(len(s)+1, device=device)
                p = (count + beta) / (total + 2*beta)
                est_vec.append(p)
                counts.append(count)
                totals.append(total)
    if save_counts:
        return est_vec, counts, totals
    else:
        return est_vec

def optimal_est(P, sequence_length, extra_args):
    weather_data = pd.read_csv(extra_args.dataset)  # Ensure it's preprocessed correctly
    weather_data = torch.tensor(weather_data.values, dtype=extra_args.dtype, device=extra_args.device)
    
    num_samples, num_features = weather_data.shape  # Should be (N, 6)
    assert num_features == 6, "Expected 6 weather features per timestamp."

    # Get batch of real sequences (ordered time series data)
    x, y, _ = get_batch(P, "weather", extra_args.order, sequence_length, 4096, extra_args, start_index=0)

    # Transform input sequence into cluster indices using transition matrix P
    cluster_assignments = torch.matmul(x, P)  # Assuming P maps features to clusters
    cluster_assignments = torch.argmax(cluster_assignments, dim=-1)  # Get closest cluster index

    # Get transition probabilities for each time step
    opt_logits = torch.zeros(x.size(0), x.size(1), P.size(1), device=P.device)

    for i in range(sequence_length - 1):
        current_cluster = cluster_assignments[:, i]
        next_cluster = cluster_assignments[:, i + 1]
        opt_logits[:, i, :] = P[current_cluster, next_cluster]  # Lookup transition probability

    # Apply log-softmax for loss calculation
    opt_logits = torch.log(opt_logits + 1e-9)  # Prevent log(0) issues
    opt_loss = F.nll_loss(opt_logits.view(-1, opt_logits.size(-1)), y.view(-1), ignore_index=-1)

    return opt_loss

# Optimized Markov data generation (thank you @cekbote!)

def get_batch(seq_length, batch_size, extra_args, start_index=0):
    """Load sequential batches from the weather dataset instead of randomly sampling."""

    # Load the dataset
    weather_data = pd.read_csv(extra_args.dataset)  # Assuming a CSV file
    weather_data = torch.tensor(weather_data.values, dtype=extra_args.dtype, device=extra_args.device)

    num_samples, num_features = weather_data.shape  # Should match 6 columns

    # Ensure the dataset has enough rows for batch processing
    assert num_samples > seq_length + 30, "Dataset is too small for the required sequence length."

    # Sequential batch selection
    end_index = start_index + batch_size
    if end_index + seq_length + 30 >= num_samples:
        start_index = 0  # Restart if we reach the end of the dataset
        end_index = batch_size

    x = torch.stack([weather_data[i:i + seq_length] for i in range(start_index, end_index)])  # Inputs
    y = torch.stack([weather_data[i + seq_length:i + seq_length + 30] for i in range(start_index, end_index)])  # Targets

    start_index = end_index  # Move forward for next batch
    return x, y, start_index




@torch.no_grad()
def eval(model, P, sequence_length, batch_size, extra_args, max_num_batches=24, ctx=nullcontext()):
    assert not model.training, "Model should be in eval mode"
    assert P is not None, "Transition matrix P is required"

    loss_list_val, acc_list = [], []

    # Load weather dataset
    weather_data = pd.read_csv(extra_args.dataset)
    weather_data = torch.tensor(weather_data.values, dtype=extra_args.dtype, device=extra_args.device)

    num_samples, num_features = weather_data.shape
    assert num_features == 6
    start_index = 0
    for _ in range(max_num_batches):
        x, y = get_batch(sequence_length, batch_size, extra_args,start_index)
        with ctx:
            outputs = model(x, targets=y)  # Forward pass

        val_loss = outputs['loss']
        loss_list_val.append(val_loss)

        # Compute accuracy by comparing predicted clusters
        predicted_clusters = outputs['logits'].argmax(-1)
        target_clusters = y.argmax(-1)  # Assuming y is cluster-based
        acc_list.append((predicted_clusters == target_clusters).float().mean())

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = torch.exp(torch.tensor(val_loss)).item()  # Use torch.exp instead of hardcoded e

    return val_acc, val_loss, val_perplexity

    return val_acc, val_loss, val_perplexity


@torch.no_grad()
def eval_probs(model, P, type, order, sequence_length, windows, generator, extra_args, betas = None, input_seq=None, output_seq=None, ctx=nullcontext()):
    assert model.training == False
    assert P is not None
    if betas is None:
        betas = [1]
    
    if input_seq is not None and output_seq is not None:
        x = input_seq[:, :sequence_length]
        y = output_seq[:, :sequence_length]
    else:
        x, y = get_batch(P, type, order, sequence_length, 1, generator, extra_args)

    # Get model estimation
    with ctx:
        outputs = model(x, targets=y, save_weights=True)
    probs = F.softmax(outputs['logits'], dim=-1)
    xb = x[0].float()
    probsb = probs[0, order-1:]
    powers = torch.Tensor([2**i for i in reversed(range(order))]).to(extra_args.device)
    idx = F.conv1d(xb.view(1, -1), powers.view(1, 1, -1)).squeeze()
    prob_vec = []
    for i in range(2**order):
        vec = probsb[idx == i][:,1] # estimated p
        prob_vec.append(vec)

    # Get (windowed) empirical add-beta estimator
    if windows is None:
        windows = [0]
    est_vec = []
    if windows == [0]:
        est_vec.append(empirical_est(x, y, type, order))
        beta_vec = []
        for beta in betas:
            beta_est = empirical_est(x, y, type, order, beta=beta)
            err = 0
            for i in range(2**order):
                err += torch.linalg.norm(prob_vec[i] - beta_est[i], ord=1)
            beta_vec.append(err)
    else:
        beta_vec = None
        for w in windows:
            est_vec.append(empirical_est(x, y, type, order, window=w))
    
    return prob_vec, est_vec, beta_vec

@torch.no_grad()
def eval_conditions(model, extra_args, ctx=nullcontext()):
    assert model.training == False

    x0 = torch.Tensor([[0,0,1,1,0]])
    x1 = torch.zeros(1,251)
    x = torch.cat((x0, x1), dim=1).to(int).to(extra_args.device)
    with ctx:
        outputs = model(x, targets=x, check_conditions=True)

    return None


def save_checkpoint(model, opt, scheduler, itr, ckpt_path, **extra_args):

    checkpoint = dict({
        'model': model.state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
        'itr': itr,
    }, **extra_args)

    torch.save(checkpoint, ckpt_path)
