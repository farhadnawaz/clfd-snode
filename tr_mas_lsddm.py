import os
import logging
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from copy import deepcopy

#from tensorboardX import SummaryWriter

import torch
from torch.optim.lr_scheduler import LambdaLR

from imitation_cl.logging.utils import custom_logging_setup, read_dict, write_dict
from imitation_cl.data.utils import get_minibatch_extended as get_minibatch
from imitation_cl.data.lasa import LASAExtended
from imitation_cl.data.helloworld import HelloWorldExtended
from imitation_cl.data.robottasks import RobotTasksExtended
from imitation_cl.train.utils import check_cuda, set_seed, get_sequence 
from imitation_cl.plot.trajectories import streamplot
from imitation_cl.model.node import MASNODETaskEmbedding
from imitation_cl.model.lsddm_emb import configure
from imitation_cl.model.lsddm_emb_t import configure as configure_t
from imitation_cl.metrics.traj_metrics import mean_swept_error, mean_frechet_error_fast as mean_frechet_error, dtw_distance_fast as dtw_distance

import os
import numpy as np
import time
from tqdm import tqdm, trange

import torch
import torch.optim as optim

# PyTorch bug: https://github.com/pytorch/pytorch/issues/49285
import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")


def parse_args(return_parser=False):
    parser = ArgumentParser()

    parser.add_argument('--data_dir', type=str, required=True, help='Location of dataset')
    parser.add_argument('--num_iter', type=int, required=True, help='Number of training iterations')
    parser.add_argument('--tsub', type=int, default=20, help='Length of trajectory subsequences for training')
    parser.add_argument('--replicate_num', type=int, default=0, help='Number of times the final point of the trajectories should be replicated for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--tnet_dim', type=int, default=2, help='Dimension of target network input and output')
    parser.add_argument('--fhat_layers', type=int, required=True, help='Number of hidden layers in the fhat of target network')
    parser.add_argument('--task_emb_dim', type=int, default=5, help='Dimension of the task embedding vector')
    parser.add_argument('--explicit_time', type=int, default=0, help='1: Use time as an explicit network input, 0: Do not use time')

    parser.add_argument('--mas_lambda', type=float, default=0.1, help='MAS regularization strength')

    parser.add_argument('--lr_change_iter', type=int, default=-1, help='-1 or 0: No LR scheduler, >0: Number of iterations after which initial LR is divided by 10')

    parser.add_argument('--lsddm_a', type=float, default=0.5)
    parser.add_argument('--lsddm_projfn', type=str, default='PSD-REHU', help='LSDDM projection function')
    parser.add_argument('--lsddm_projfn_eps', type=float, default=0.0001)
    parser.add_argument('--lsddm_smooth_v', type=int, default=0)
    parser.add_argument('--lsddm_hp', type=int, default=60)
    parser.add_argument('--lsddm_h', type=int, default=1000)
    parser.add_argument('--lsddm_rehu', type=float, default=0.01)

    parser.add_argument('--dummy_run', type=int, default=0, help='1: Dummy run, no evaluation, 0: Actual training run')

    parser.add_argument('--data_class', type=str, required=True, help='Dataset class for training')
    parser.add_argument('--eval_during_train', type=int, default=0, help='0: net for a task is evaluated immediately after training, 1: eval for all nets is done after training of all tasks')
    parser.add_argument('--seed', type=int, required=True, help='Seed for reproducability')
    parser.add_argument('--seq_file', type=str, required=True, help='Name of file containing sequence of demonstration files')
    parser.add_argument('--log_dir', type=str, default='logs/', help='Main directory for saving logs')
    parser.add_argument('--description', type=str, required=True, help='String identifier for experiment')

    # Plot traj or not
    parser.add_argument('--plot_traj', type=int, default=1, help='1: Plot the traj plots, 0: Dont plot traj_plots')

    # Plot vectorfield or not
    parser.add_argument('--plot_vectorfield', type=int, default=1, help='1: Plot vector field in the traj plots, 0: Dont plot vector field')

    # Args for plot formatting
    parser.add_argument('--plot_fs', type=int, default=10, help='Fontsize to be used in the plots')
    parser.add_argument('--figw', type=float, default=16.0, help='Plot width')
    parser.add_argument('--figh', type=float, default=3.3, help='Plot height')

    if return_parser:
        # This is used by the slurm creator script
        # When running this script directly, this has no effect
        return parser
    else:
        args = parser.parse_args()
        return args

def train_task(args, task_id, tnet, node, device, pbar=trange, writer=None):

    starttime = time.time()

    filenames = get_sequence(args.seq_file)

    dataset = None
    if args.data_class == 'LASA':
        datafile = os.path.join(args.data_dir, filenames[task_id])
        dataset = LASAExtended(datafile, seq_len=args.tsub, norm=True, device=device)

        # Goal position at origin
        dataset.zero_center()
    elif args.data_class == 'HelloWorld':
        dataset = HelloWorldExtended(data_dir=args.data_dir, filename=filenames[task_id], device=device)

        # Goal position at origin
        dataset.zero_center()
    elif args.data_class == 'RobotTasks':
        dataset = RobotTasksExtended(data_dir=args.data_dir, datafile=filenames[task_id], device=device)

        # Goal position at origin
        dataset.zero_center()
    else:
        raise NotImplementedError(f'Unknown dataset class {args.data_class}')

    node.set_target_network(tnet)

    node.set_task_id(task_id)

    tnet.train()
    node.train()

    # Create a new task embedding for this task
    node.gen_new_task_emb()
    node = node.to(device)

    # For optimizing the weights and biases of the NODE
    theta_optimizer = optim.Adam(node.target_network.parameters(), lr=args.lr)

    # For optimizing the task embedding for the current task.
    # We only optimize the task embedding corresponding to the current task,
    # the remaining ones stay constant.

    emb_optimizer = optim.Adam([node.get_task_emb(task_id)], lr=args.lr)

    # Whether the regularizer will be computed during training?
    calc_reg = task_id > 0 and args.mas_lambda > 0

    # Apply learning scheduler if needed
    if args.lr_change_iter > 0:
        theta_lambda = lambda epoch: 1.0 if (epoch < args.lr_change_iter) else 0.1
        theta_scheduler = LambdaLR(theta_optimizer, lr_lambda=theta_lambda)
        emb_lambda = lambda epoch: 1.0 if (epoch < args.lr_change_iter) else 0.1
        emb_scheduler = LambdaLR(emb_optimizer, lr_lambda=emb_lambda)

    best_loss = np.inf
    best_iter = 0

    # Start training iterations
    for iteration in pbar(args.num_iter):

        ### Train theta and task embedding.
        theta_optimizer.zero_grad()
        emb_optimizer.zero_grad()

        # Train using the translated trajectory (with goal at the origin)
        t, y_all = get_minibatch(dataset.t[0], dataset.pos_goal_origin, nsub=None, tsub=args.tsub, dtype=torch.float)

        # We use the timesteps associated with the first sequence
        # Starting points
        y_start = y_all[:,0].float()
        y_start.requires_grad = True

        # Predicted trajectories - forward simulation
        y_hat = node(t.float(), y_start) 
        
        # MSE
        loss = ((y_hat-y_all)**2).mean()

        # Log the loss in tensorboard
        if writer is not None:
            writer.add_scalar(f'task_loss/task_{task_id}', loss.item(), iteration)

        # Calling loss_task.backward computes the gradients w.r.t. the loss for the 
        # current task. 
        loss.backward(retain_graph=calc_reg, create_graph=False)

        # The task embedding is only trained on the task-specific loss.
        emb_optimizer.step()

        if calc_reg:

            loss_reg = node.regularization_loss()
            loss_reg *= args.mas_lambda

            # Backpropagate the regularization loss
            loss_reg.backward()

        # Update the NODE params using the current task loss and the regularization loss
        theta_optimizer.step()

        if args.lr_change_iter > 0:
            theta_scheduler.step()
            emb_scheduler.step()

        if loss.item() <= best_loss:
            best_node = deepcopy(node)
            best_loss = loss.item()
            best_iter = int(iteration)        


    if calc_reg:

        # Update omega on finishing the training for a task
        best_node.update_omega(args=args, 
                               data=dataset,
                               get_minibatch_fn=get_minibatch,
                               device=device)

        # Save the parameters
        best_node.update_theta_prev()

        # Merge with previous omega
        best_node.merge_omega()

    endtime = time.time()
    duration = endtime - starttime

    return best_node, duration, best_loss, best_iter

def eval_task(args, task_id, node, device, ax=None):

    node = node.to(device)

    filenames = get_sequence(args.seq_file)

    if args.data_class == 'LASA':
        datafile = os.path.join(args.data_dir, filenames[task_id])
        dataset = LASAExtended(datafile, seq_len=args.tsub, norm=True, device=device)

        # Goal position at origin
        dataset.zero_center()
    elif args.data_class == 'HelloWorld':
        dataset = HelloWorldExtended(data_dir=args.data_dir, filename=filenames[task_id], device=device)

        # Goal position at origin
        dataset.zero_center()
    elif args.data_class == 'RobotTasks':
        dataset = RobotTasksExtended(data_dir=args.data_dir, datafile=filenames[task_id], device=device)

        # Goal position at origin
        dataset.zero_center()
    else:
        raise NotImplementedError(f'Unknown dataset class {args.data_class}')

    # Set the target network in the NODE
    #node.set_target_network(tnet)
    node = node.float()
    node.eval()

    # Choose the task_id for the NODE
    node.set_task_id(task_id)

    # The time steps
    t = dataset.t[0].float()

    # The starting position 
    # (n,d-dimensional, where n is the num of demos and 
    # d is the dimension of each point)
    #y_start = torch.unsqueeze(dataset.pos[0,0], dim=0)
    # Use the translated trajectory (goal at origin)
    y_start = dataset.pos_goal_origin[:,0]
    y_start = y_start.float()
    y_start.requires_grad = True    

    # The entire demonstration trajectory
    y_all = dataset.pos.float()

    # The predicted trajectory is computed in a piecemeal fashion
    # Predicted trajectory
    t_step = 20
    t_start = 0
    t_end = t_start + t_step
    y_start = y_start
    y_hats = list()
    i = 0
    
    while t_end <= y_all.shape[1]:
        i += 1
        y_hat = node(t[t_start:t_end], y_start)
        y_hats.append(y_hat)
        y_start = y_hat[:,-1,:].detach().clone()
        y_start.requires_grad = True
        t_start = t_end
        t_end = t_start + t_step

    y_hat_zeroed = torch.cat(y_hats, 1)
    y_hat = dataset.unzero_center(y_hat_zeroed)
    y_hat_np = y_hat.cpu().detach().numpy()

    #y_hats_np = [yy.cpu().detach().numpy() for yy in y_hats]
    #y_hat_np = np.concatenate(y_hats_np, axis=1)

    # Translate goal to away from the origin
    #y_hat_np = dataset.unzero_center(y_hat_np)

    # Compute trajectory metrics
    y_all_np = y_all.cpu().detach().numpy()

    # De-normalize the data before computing trajectories
    y_all_np_denorm = dataset.denormalize(y_all_np)
    y_hat_np_denorm = dataset.denormalize(y_hat_np)

    # Compute the error metric (array of metrics for each trajectory in the ground truth)
    metric_dtw_err, metric_dtw_errs = dtw_distance(y_all_np_denorm, y_hat_np_denorm)
    metric_frechet_err, metric_frechet_errs = mean_frechet_error(y_all_np_denorm, y_hat_np_denorm)
    metric_swept_err, metric_swept_errs = mean_swept_error(y_all_np_denorm, y_hat_np_denorm)

    eval_traj_metrics = {'swept': metric_swept_err, 
                         'frechet': metric_frechet_err, 
                         'dtw': metric_dtw_err}

    # Store the metric errors
    # Convert np arrays to list so that these can be written to JSON
    eval_traj_metric_errors = {'swept': metric_swept_errs.tolist(), 
                               'frechet': metric_frechet_errs.tolist(), 
                               'dtw': metric_dtw_errs.tolist()}

    plot_data = {'t': t.detach().cpu().numpy(),
                 'y_all': dataset.pos_goal_origin.cpu().detach().numpy(),
                 'y_hat': y_hat_zeroed.cpu().detach().numpy()}

    return eval_traj_metrics, eval_traj_metric_errors, plot_data


def train_all(args):

    # Create logging folder and set up console logging
    save_dir, identifier = custom_logging_setup(args)

    # Tensorboard logging setup
    # writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tb', args.description, identifier))

    # Check if cuda is available
    cuda_available, device = check_cuda()
    logging.info(f'cuda_available: {cuda_available}')

    properties = {"latent_space_dim":args.tnet_dim,
                  "explicit_time": args.explicit_time,
                  "a":args.lsddm_a,
                  "projfn":args.lsddm_projfn,
                  "projfn_eps":args.lsddm_projfn_eps,
                  "smooth_v":args.lsddm_smooth_v,
                  "hp":args.lsddm_hp,
                  "h":args.lsddm_h,
                  "rehu":args.lsddm_rehu,
                  "task_emb_dim": args.task_emb_dim,
                  "device": device,
                  "fhat_layers": args.fhat_layers}        

    # The NODE uses the target network as the RHS of its
    # differential equation
    if args.explicit_time==1:
        properties["explicit_time"] = args.explicit_time
        target_network = configure_t(properties)
    elif args.explicit_time==0:
        target_network = configure(properties)

    node = MASNODETaskEmbedding(target_network=target_network, 
                                te_dim=args.task_emb_dim, 
                                mas_lambda=args.mas_lambda, 
                                method='euler', 
                                explicit_time=args.explicit_time,
                                verbose=True).to(device)

    # Extract the list of demonstrations from the text file 
    # containing the sequence of demonstrations
    seq = get_sequence(args.seq_file)

    num_tasks = len(seq)

    eval_resuts=None

    for task_id in range(num_tasks):

        logging.info(f'#### Training started for task_id: {task_id} (task {task_id+1} out of {num_tasks}) ###')

        # Train on the current task_id
        node, duration, best_loss, best_iter = train_task(args=args, task_id=task_id, tnet=target_network, node=node, device=device, writer=None)

        logging.info(f'task_id: {task_id}, best_loss: {best_loss:.3E}, best_iter: {best_iter}')
        
        # At the end of every task store the latest networks
        logging.info('Saving models')
        torch.save(node, os.path.join(save_dir, 'models', f'node_{task_id}.pth'))

        if args.eval_during_train == 0:
            # Evaluate the latest network immediately after training
            # is complete for a task
            eval_resuts = eval_during_train(args, save_dir, task_id, eval_resuts)
        elif args.eval_during_train == 1:
            # Evaluation is done after training is finished for all tasks
            pass
        elif args.eval_during_train == 2:
            # No evaluation is performed, this is a trail run
            pass
        else:
            raise NotImplementedError(f'Unknown arg eval_during_train: {args.eval_during_train}')

    logging.info('Training done')

    # writer.close()

    return save_dir

def eval_during_train(args, save_dir, train_task_id, eval_results=None, writer=None):
    """
    Evaluates one saved model after training for 
    that task is complete.

    This avoids the need to save the networks for each task 
    for the purpose of evaluation.
    """

    # Check if cuda is available
    cuda_available, device = check_cuda()
    logging.info(f'cuda_available: {cuda_available}')

    # Dict for storing evaluation results
    # This will be written to a json file in the log folder
    # Create this if this is the first time eval is run
    if eval_results is None:
        eval_results = dict()

        # For storing command line arguments for this run
        eval_results['args'] = read_dict(os.path.join(save_dir, 'commandline_args.json'))

        # For storing the evaluation results
        eval_results['data'] = {'metrics': dict(), 'metric_errors': dict()}

    # Create a target network without parameters
    # Parameters are overwritten during the forward pass of the hypernetwork
    properties = {"latent_space_dim":args.tnet_dim,
                  "explicit_time": args.explicit_time,
                  "a":args.lsddm_a,
                  "projfn":args.lsddm_projfn,
                  "projfn_eps":args.lsddm_projfn_eps,
                  "smooth_v":args.lsddm_smooth_v,
                  "hp":args.lsddm_hp,
                  "h":args.lsddm_h,
                  "rehu":args.lsddm_rehu,
                  "device": device,
                  "task_emb_dim": args.task_emb_dim,
                  "fhat_layers": args.fhat_layers}

    # Create a LSDDM network 
    # Parameters are supplied during the forward pass of the hypernetwork
    # Load the network for the current task_id
    if args.explicit_time==1:
        target_network = configure_t(properties)
    elif args.explicit_time==0:
        target_network = configure(properties)    
        
    target_network = target_network.to(device)

    # Shapes of the target network parameters
    param_names, param_shapes = target_network.get_param_shapes()
        
    # The NODE uses the target network as the RHS of its
    # differential equation
    # Apart from this, the NODE has no other trainable parameters
    node = MASNODETaskEmbedding(target_network=target_network, 
                                te_dim=args.task_emb_dim,
                                mas_lambda=args.mas_lambda, 
                                explicit_time=args.explicit_time,
                                method='euler').to(device)

    node = torch.load(os.path.join(save_dir, 'models', f'node_{train_task_id}.pth'))

    # Extract the list of demonstrations from the text file 
    # containing the sequence of demonstrations
    seq = get_sequence(args.seq_file)

    num_tasks = len(seq)

    # After the last task has been trained, we create a plot
    # showing the performance on all the tasks
    if train_task_id == (num_tasks - 1) and args.plot_traj==1:
        figw, figh = args.figw, args.figh
        plt.subplots_adjust(left=1/figw, right=1-1/figw, bottom=1/figh, top=1-1/figh)
        fig, axes = plt.subplots(figsize=(figw, figh), 
                                 sharey=False, 
                                 sharex=False,
                                 ncols=num_tasks if num_tasks<=10 else (num_tasks//2), 
                                 nrows=1 if num_tasks<=10 else 2,
                                 subplot_kw={'aspect': 1 if args.plot_vectorfield==1 else 'auto',
                                             'projection': 'rectilinear' if args.plot_vectorfield==1 else '3d'})

        # Row column for plot with trajectories
        r, c = 0, 0

    logging.info(f'#### Evaluation started for task_id: {train_task_id} (task {train_task_id+1} out of {num_tasks}) ###')

    eval_results['data']['metrics'][f'train_task_{train_task_id}'] = dict()
    eval_results['data']['metric_errors'][f'train_task_{train_task_id}'] = dict()

    # Evaluate on all the past and current task_ids
    for eval_task_id in range(train_task_id+1):
        logging.info(f'Loaded network trained on task {train_task_id}, evaluating on task {eval_task_id}')

        # Figure is plotted only for the last task
        
        eval_traj_metrics, eval_traj_metric_errors, plot_data = eval_task(args, eval_task_id, node, device)

        # Plot the trajectories for the last trained model
        if train_task_id == (num_tasks-1) and args.plot_traj==1:

            r = 1 if num_tasks<=10 else eval_task_id//(num_tasks//2)
            c = eval_task_id if num_tasks<=10 else eval_task_id%(num_tasks//2)

            if num_tasks == 1:
                ax = axes
            elif num_tasks<=10:
                ax = axes[c] 
            else:
                ax = axes[r][c]            

            streamplot(t=plot_data['t'],
                       y_all=plot_data['y_all'],
                       y_hat=plot_data['y_hat'],
                       ode_rhs=node.ode_rhs,
                       V=None,
                       L=1,
                       ax=ax,
                       fontsize=10,
                       device=device,
                       limit=4.0,
                       alpha=0.6,
                       explicit_time=args.explicit_time,
                       plot_vectorfield=args.plot_vectorfield,
                       extra_t=True
                       )

            ax.set_title(eval_task_id, fontsize=args.plot_fs)
            
            # Remove axis labels and ticks
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.xaxis.get_label().set_visible(False)
            ax.yaxis.get_label().set_visible(False)
            fig.legend(loc='lower center', fontsize=args.plot_fs, ncol=4)
        
        logging.info(f'Evaluated trajectory metrics: {eval_traj_metrics}')

        # Store the evaluated metrics
        eval_results['data']['metrics'][f'train_task_{train_task_id}'][f'eval_task_{eval_task_id}'] = eval_traj_metrics
        eval_results['data']['metric_errors'][f'train_task_{train_task_id}'][f'eval_task_{eval_task_id}'] = eval_traj_metric_errors


    if train_task_id == (num_tasks-1) and args.plot_traj==1:
        fig.subplots_adjust(hspace=-0.2, wspace=0.1)

        # Save the evaluation plot
        if args.plot_vectorfield == 1:
            plt.savefig(os.path.join(save_dir, f'plot_trajectories_{args.description}.pdf'), bbox_inches='tight')
        else:
            plt.savefig(os.path.join(save_dir, f'plot_trajectories_{args.description}.pdf'))

    # (Over)write the evaluation results to a file in the log dir
    write_dict(os.path.join(save_dir, 'eval_results.json'), eval_results)

    # Remove the networks that have been evaluated (except for the network of the last task)
    if train_task_id < (num_tasks-1):
        os.remove(os.path.join(save_dir, 'models', f'node_{train_task_id}.pth'))

    logging.info('Current task evaluation done')

    return eval_results


if __name__ == '__main__':

    # Parse commandline arguments
    args = parse_args()

    # Set the seed for reproducability
    set_seed(args.seed)

    # Training
    save_dir = train_all(args)

    # Evaluation can be run in a standalone manner if needed
    if args.eval_during_train == 1:
        raise NotImplementedError('eval_during_train=1 is not supported')

    logging.info('Completed')