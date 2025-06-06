from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import time
import pandas as pd

from .utils import eval, eval_probs, eval_conditions, get_batch, optimal_est, save_checkpoint


def train_base(model, opt, P, scheduler, iterations, acc_steps, batch_size, sequence_length, eval_freq, ckpt_path, extra_args):
    device_type = 'cuda' if 'cuda' in str(extra_args.device) else 'cpu'
    type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=torch.float16)
    itr, substep, best_val_loss, text_table = 0, 0, float('inf'), None
    
    print(f"Compiling model ...")
    model = torch.compile(model) # requires pytorch 2.0+

    if P is not None:
        P_test = P
        print("Markov transition matrix:")
        print(P)
    
    # Optimal test loss
    opt_loss = optimal_est(P_test,sequence_length,extra_args)
    if extra_args.wandb:
        wandb.log({
            "val/opt_loss": opt_loss,
        })

    # Freeze convolution kernels if requested
    if extra_args.fix_conv:
        for mn, m in model.named_modules():
            for pn, p in m.named_parameters():
                if pn.endswith('conv1d.weight') or pn.endswith('conv1d.bias') or "ker" in pn:
                    p.requires_grad = False

    model.train()
    t0 = time.time()
    start_index = 0
    while itr < iterations:
        for microstep_idx in range(acc_steps):  # gradient accumulation
            x, y,start_index = get_batch(sequence_length, batch_size, extra_args,start_index)
            with type_ctx:
                outputs = model(x, targets=y)
            loss = outputs['loss'] / acc_steps
            loss.backward()
            substep += 1

        if extra_args.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), extra_args.grad_clip)

        opt.step()
        scheduler.step()
        opt.zero_grad(set_to_none=True)
        itr += 1

        if itr % eval_freq == 0 or itr == iterations: # from here it's only evaluation code, all the training is above
            t1 = time.time()
            dt = t1 - t0

            model.eval()
            train_loss = loss.detach().cpu().item()
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else extra_args.lr
            val_acc, val_loss, val_perplexity = eval(model, P_test, sequence_length, batch_size, extra_args, max_num_batches=10, ctx=type_ctx)

            print_string = f"{itr} [train] loss={train_loss:.3f} [val] loss={val_loss:.3f}, pp={val_perplexity:.2f}, acc={val_acc:3f}"
            print_string += f" [time per itr] {dt*1000/eval_freq:.2f}ms"
            if scheduler is not None:
                print_string += f" [lr] {current_lr:.5f}"
            print(print_string)

            if extra_args.wandb:
                wandb.log({
                    "iter": itr,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/loss_gap": val_loss - opt_loss,
                    "val/perplexity": val_perplexity,
                    "val/acc": val_acc,
                    "lr": current_lr,
                })
            
            if itr == iterations:
                windows = [0] # Windows for add-beta estimator
                '''x, y = get_batch(P_test, type, order, sequence_length, 1, generator, extra_args)
                print("Test sequence:")
                print(x[0,:30])
                torch.save(x, 'x_test.pt')
                torch.save(y, 'y_test.pt')
                if extra_args.wandb:
                    wandb.save('x_test.pt')
                    wandb.save('y_test.pt')
                x = torch.load('x_test.pt')
                y = torch.load('y_test.pt')'''
                x = None
                y = None

                betas = np.linspace(0.1, 3, 30)
                prob_vec, est_vec, beta_vec = eval_probs(model, P_test, sequence_length, windows, extra_args,
                                betas=betas, input_seq=x, output_seq=y, ctx=type_ctx)
                if extra_args.wandb:
                    num_features = x.shape[-1]  # Number of weather variables (e.g., 6)

                    for feature_idx in range(num_features):
                        for i in range(len(prob_vec[feature_idx])):
                             # Dictionary to store estimation logs
                                est_dict = {
                                    f"est/model_est_feature_{feature_idx}": prob_vec[feature_idx][i].detach().cpu().item()}
                                for j, w in enumerate(windows):
                                    est_dict[f"est/empirical_est_w{w}_feature_{feature_idx}"] = (
                                        est_vec[j][feature_idx][i].detach().cpu().item()
                                    )
                                wandb.log(est_dict)

                    # Log beta error estimates
                    if beta_vec is not None:
                        for b, beta_value in enumerate(betas):
                            wandb.log({f"est/beta_err_{beta_value:.2f}": beta_vec[b].detach().cpu().item()})
                eval_conditions(model, extra_args, ctx=type_ctx)

            model.train()
            t0 = time.time()

    print(f"saving checkpoint to {ckpt_path}")
    save_checkpoint(model=model,
                    opt=opt,
                    scheduler=scheduler,
                    itr=itr,
                    ckpt_path=ckpt_path)

