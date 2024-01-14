from tqdm import trange
from argparse import ArgumentParser
import logging
import os
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from iflow.dataset import lasa_dataset
from iflow import model
from iflow.trainers import goto_dynamics_train
from iflow.utils import to_numpy, to_torch

from imitation_cl.train.utils import check_cuda, set_seed, get_sequence
from imitation_cl.metrics.traj_metrics import mean_swept_error, mean_frechet_error_fast as mean_frechet_error, dtw_distance_fast as dtw_distance
from imitation_cl.logging.utils import custom_logging_setup, write_dict, read_dict, Dictobject


def parse_args(return_parser=False):
    parser = ArgumentParser()

    # parser.add_argument('--data_dir', type=str, required=True, help='Location of dataset')
    parser.add_argument('--seed', type=int, required=True, help='Seed for reproducability')
    parser.add_argument('--seq_file', type=str, required=True, help='Name of file containing sequence of demonstration files')
    parser.add_argument('--log_dir', type=str, default='logs_iFlow/', help='Main directory for saving logs')
    parser.add_argument('--description', type=str, required=True, help='String identifier for experiment')

    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--depth', type=int, default=10, help='Depth of network')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--nr_epochs', type=int, default=1000, help='Number of training epochs')

    if return_parser:
        # This is used by the slurm creator script
        # When running this script directly, this has no effect
        return parser
    else:
        args = parser.parse_args()
        return args

#### Invertible Flow model #####
def main_layer(dim):
    return  model.CouplingLayer(dim)

def create_flow_seq(dim, depth):
    chain = []
    for i in range(depth):
        chain.append(main_layer(dim))
        chain.append(model.RandomPermutation(dim))
        chain.append(model.LULinear(dim))
    chain.append(main_layer(dim))
    return model.SequentialFlow(chain)

def eval_task(args, task_id, iflow, device):

    filenames = get_sequence(args.seq_file)
    filename=filenames[task_id].replace('.mat','')

    data = lasa_dataset.LASA(filename=filename)
    val_trajs = data.train_data

    with torch.no_grad():
        iflow.eval()

        ### Generate Predicted Trajectories ###
        predicted_trajs = []
        for trj in val_trajs:
            n_trj = trj.shape[0]
            y0 = trj[0, :]
            y0 = torch.from_numpy(y0[None, :]).float().to(device)
            traj_pred = iflow.generate_trj(y0, T=n_trj)
            traj_pred = traj_pred.detach().cpu().numpy()
            predicted_trajs.append(traj_pred)

        predicted_trajs = data.unormalize(np.array(predicted_trajs))

        val_trajs = data.unormalize(np.array(val_trajs))
        metric_frechet_err, metric_frechet_errs = mean_frechet_error(val_trajs, predicted_trajs)
        metric_dtw_err, metric_dtw_errs = dtw_distance(val_trajs, predicted_trajs)
        metric_swept_err, metric_swept_errs = mean_swept_error(val_trajs, predicted_trajs)

        eval_traj_metrics = {'swept': metric_swept_err, 
                            'frechet': metric_frechet_err, 
                            'dtw': metric_dtw_err}

        # Store the metric errors
        # Convert np arrays to list so that these can be written to JSON
        eval_traj_metric_errors = {'swept': metric_swept_errs.tolist(), 
                                'frechet': metric_frechet_errs.tolist(), 
                                'dtw': metric_dtw_errs.tolist()}

        return eval_traj_metrics, eval_traj_metric_errors

def train_task(args, task_id, device):

    filenames = get_sequence(args.seq_file)
    filename=filenames[task_id].replace('.mat','')

    data = lasa_dataset.LASA(filename=filename)
    dim = data.dim
    params = {'batch_size': args.batch_size, 'shuffle': True}
    dataloader = DataLoader(data.dataset, **params)

    ######### Model #########
    dynamics = model.TanhStochasticDynamics(dim, device=device, dt=0.01, T_to_stable=2.5)
    #dynamics = model.LinearStochasticDynamics(dim, dt=0.01, T_to_stable=2.5)
    flow = create_flow_seq(dim, args.depth).to(device)
    iflow = model.ContinuousDynamicFlow(dynamics=dynamics, device=device, model=flow, dim=dim)

    # TODO REMOVE
    num_params = 0
    shapes = list()

    for n,p in iflow.named_parameters():
        shape = p.shape
        num_params += np.prod(list(shape))
        shapes.append(list(p.shape))

    print('###################################')
    print('num_params = ', num_params)
    print('num param tensors = ',len(shapes))
    print('param shapes = ',shapes)
    print('###################################')

    ######### Optimization #########
    params = list(flow.parameters()) + list(dynamics.parameters())
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    #print(f'Dataloader: {len(dataloader)}')
    #exit(0)

    for i in trange(args.nr_epochs):
        ## Training ##
        for local_x, local_y in dataloader:
            local_x = local_x.to(device)
            local_y = [l.to(device) for l in local_y]
            dataloader.dataset.set_step()
            optimizer.zero_grad()
            loss = goto_dynamics_train(iflow, local_x, local_y)
            loss.backward(retain_graph=True)
            optimizer.step()

    return iflow

def train_eval(args):

    # Create logging folder and set up console logging
    save_dir, identifier = custom_logging_setup(args)

    # Check if cuda is available
    cuda_available, device = check_cuda()
    logging.info(f'cuda_available: {cuda_available}')

    # Extract the list of demonstrations from the text file 
    # containing the sequence of demonstrations
    seq = get_sequence(args.seq_file)

    num_tasks = len(seq)

    # Dict for storing evaluation results
    # This will be written to a json file in the log folder
    eval_results = dict()

    # For storing command line arguments for this run
    eval_results['args'] = read_dict(os.path.join(save_dir, 'commandline_args.json'))

    # For storing the evaluation results
    eval_results['data'] = {'metrics': dict(), 'metric_errors': dict()}

    for task_id in range(num_tasks):

        logging.info(f'#### Training started for task_id: {task_id} (task {task_id+1} out of {num_tasks}) ###')

        # Train on the current task_id
        iflow = train_task(args, task_id, device)

        # Calculating parameter size
        # param_size = 0
        # for n,p in iflow.named_parameters():
        #     param_size += np.prod(list(p.shape))
        # print(f'task_id={task_id} param_size={param_size}')

        logging.info(f'#### Evaluation started for task_id: {task_id} (task {task_id+1} out of {num_tasks}) ###')

        eval_results['data']['metrics'][f'train_task_{task_id}'] = dict()
        eval_results['data']['metric_errors'][f'train_task_{task_id}'] = dict()        

        eval_traj_metrics, eval_traj_metric_errors = eval_task(args, task_id, iflow, device)
      
        logging.info(f'Evaluated trajectory metrics: {eval_traj_metrics}')

        # Store the evaluated metrics
        eval_results['data']['metrics'][f'train_task_{task_id}'][f'eval_task_{task_id}'] = eval_traj_metrics
        eval_results['data']['metric_errors'][f'train_task_{task_id}'][f'eval_task_{task_id}'] = eval_traj_metric_errors

    # Write the evaluation results to a file in the log dir
    write_dict(os.path.join(save_dir, 'eval_results.json'), eval_results)

    logging.info('Done')

    return save_dir

if __name__ == '__main__':

    # Parse commandline arguments
    args = parse_args()

    # Set the seed for reproducability
    set_seed(args.seed)

    # Training and evaluation
    save_dir = train_eval(args)

    logging.info('Completed')