import argparse
import torch
import torch.utils.data
from n_body_system.dataset_nbody import NBodyMStickSeqDataset as NBodyMStickDataset
from motion.dataset import MotionSeqDataset as MotionDataset
from md17.dataset import MD17SeqDataset as MD17Dataset

from n_body_system.model import GNN, Baseline, Linear, EGNN_vel, Linear_dynamics, RF_vel, GMN
import os
from torch import nn, optim
import json

import random, time
import numpy as np


from models.ncgnn import NCGNN
from util import save_main_results

parser = argparse.ArgumentParser(description='Graph Mechanics Networks')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N', help='experiment_name')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=5, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='n_body_system/logs', metavar='N',
                    help='folder to output the json log file')
parser.add_argument('--lr', type=float, default=5e-4, metavar='N',
                    help='learning rate')
parser.add_argument('--nf', type=int, default=64, metavar='N',
                    help='hidden dim')
parser.add_argument('--attention', type=int, default=0, metavar='N',
                    help='attention in the ae model')
parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--max_training_samples', type=int, metavar='N',
                    help='maximum amount of training samples')

parser.add_argument('--weight_decay', type=float, default=1e-12, metavar='N',
                    help='timing experiment')
parser.add_argument('--norm_diff', type=eval, default=False, metavar='N',
                    help='normalize_diff')
parser.add_argument('--tanh', type=eval, default=False, metavar='N',
                    help='use tanh')
parser.add_argument('--data_dir', type=str, default='spatial_graph/n_body_system/new_dataset/data',
                    help='Data directory.')
parser.add_argument('--learnable', type=eval, default=False, metavar='N',
                    help='Use learnable FK.')

parser.add_argument("--config_by_file", default=False, action="store_true", )


parser.add_argument('--dataset', type=str, default='motion', metavar='N',
                    help='available datasets: md17, motion, nbody')
parser.add_argument('--n_isolated', type=int, default=1,
                    help='Number of isolated balls.')
parser.add_argument('--n_stick', type=int, default=2,
                    help='Number of sticks.')
parser.add_argument('--n_hinge', type=int, default=0,
                    help='Number of hinges.')
parser.add_argument('--mol', type=str, default='aspirin',
                    help='Name of the molecule.')
parser.add_argument('--delta_frame', type=int, default=50,
                    help='Number of frames delta.')

parser.add_argument('--model', type=str, default='gmn', metavar='N',
                    help='available models: gmn, egnn_vel, rf_vel')
parser.add_argument("--n_step", type=int, default=2, help='Number of steps for NCGNN')
parser.add_argument("--use_extra_data", type=int, default=1, help='use additional datapoints for training, i.e., NCGNN+')
parser.add_argument("--alignment_loss_weight", type=float, default=0.1, help='The weight of alignment regularization loss for NCGNN+')
parser.add_argument("--l2_penalty", type=int, default=1, help='apply adaptive l2 penalty')
parser.add_argument("--l2_penalty_decay", type=float, default=0.99, help='The decay of adaptive l2 penalty loss')



parser.add_argument('--filename', type=str, default='main_results', metavar='N',
                    help='save the results into a csv file')


args = parser.parse_args()
if args.config_by_file:

    job_param_path = 'configs/simple_config_%s.json' % args.dataset

    with open(job_param_path, 'r') as f:
        hyper_params = json.load(f)
        args.exp_name = hyper_params["exp_name"]
        args.batch_size = hyper_params["batch_size"]
        args.epochs = hyper_params["epochs"]
        
        if not args.max_training_samples:
            args.max_training_samples = hyper_params["max_training_samples"]
        args.data_dir = hyper_params["data_dir"]

        args.alignment_loss_weight = hyper_params["alignment_loss_weight"]

        if 'delta_frame' in hyper_params:
            args.delta_frame = hyper_params["delta_frame"]
        # args.model = hyper_params["model"]
        # args.n_isolated = hyper_params["n_isolated"]
        # args.n_stick = hyper_params["n_stick"]
        # args.n_hinge = hyper_params["n_hinge"]

args.cuda = torch.cuda.is_available()


device = torch.device("cuda" if args.cuda else "cpu")
loss_mse = nn.MSELoss()

# print(args)
try:
    os.makedirs(args.outf)
except OSError:
    pass

try:
    os.makedirs(args.outf + "/" + args.exp_name)
except OSError:
    pass

# torch.autograd.set_detect_anomaly(True)


def get_velocity_attr(loc, vel, rows, cols):

    diff = loc[cols] - loc[rows]
    norm = torch.norm(diff, p=2, dim=1).unsqueeze(1)
    u = diff/norm
    va, vb = vel[rows] * u, vel[cols] * u
    va, vb = torch.sum(va, dim=1).unsqueeze(1), torch.sum(vb, dim=1).unsqueeze(1)
    return va

def get_dataloaders(args):
    if args.dataset == 'md17':
        dataset_train = MD17Dataset(partition='train', max_samples=args.max_training_samples, data_dir=args.data_dir,
                                molecule_type=args.mol, delta_frame=args.delta_frame, n_step=args.n_step)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                                num_workers=8)

        dataset_val = MD17Dataset(partition='val', max_samples=2000, data_dir=args.data_dir,
                                    molecule_type=args.mol, delta_frame=args.delta_frame, n_step=args.n_step)
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                                num_workers=8)

        dataset_test = MD17Dataset(partition='test', max_samples=2000, data_dir=args.data_dir,
                                    molecule_type=args.mol, delta_frame=args.delta_frame, n_step=args.n_step)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                                num_workers=8)
    elif args.dataset == 'motion':
        dataset_train = MotionDataset(partition='train', max_samples=args.max_training_samples, data_dir=args.data_dir,
                                  delta_frame=args.delta_frame, n_step=args.n_step)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                                num_workers=8)

        dataset_val = MotionDataset(partition='val', max_samples=600, data_dir=args.data_dir,
                                    delta_frame=args.delta_frame, n_step=args.n_step)
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                                num_workers=8)

        dataset_test = MotionDataset(partition='test', max_samples=600, data_dir=args.data_dir,
                                    delta_frame=args.delta_frame, n_step=args.n_step)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                                num_workers=8)

    elif args.dataset == 'nbody':
        n_isolated, n_stick, n_hinge = args.n_isolated, args.n_stick, args.n_hinge

        dataset_train = NBodyMStickDataset(partition='train',
                                            max_samples=args.max_training_samples, n_isolated=n_isolated,
                                            n_stick=n_stick, n_hinge=n_hinge, data_dir=args.data_dir, n_step=args.n_step)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                                num_workers=8)

        dataset_val = NBodyMStickDataset(partition='val', n_isolated=n_isolated,
                                            n_stick=n_stick, n_hinge=n_hinge, data_dir=args.data_dir, n_step=args.n_step)
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                                num_workers=8)

        dataset_test = NBodyMStickDataset(partition='test', n_isolated=n_isolated,
                                            n_stick=n_stick, n_hinge=n_hinge, data_dir=args.data_dir, n_step=args.n_step)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                                num_workers=8)

    return train_loader, test_loader, val_loader



def main():
    # fix seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    
    train_loader, test_loader, val_loader = get_dataloaders(args)
    model = NCGNN(args, device)

    return train(model, train_loader, test_loader, val_loader)



def train(model, train_loader, test_loader, val_loader):
    results = {'epochs': [], 'loss': [], 'train loss': []}
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    best_train_loss = 1e8
    
    t = time.time()
    epoch_time = 0
    training_time = t
    for epoch in range(args.epochs):
        model.args.epoch = epoch

        et = time.time()
        train_loss = train_one_epoch(model, epoch, train_loader)
        epoch_time += time.time() - et

        results['train loss'].append(train_loss)
        if epoch % args.test_interval == 0:
            val_loss = train_one_epoch(model, epoch, val_loader, training=False)
            test_loss = train_one_epoch(model, epoch, test_loader, training=False)
            results['epochs'].append(epoch)
            results['loss'].append(test_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_train_loss = train_loss
                best_epoch = epoch
            print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best apoch %d"
                  % (best_val_loss, best_test_loss, best_epoch))

        json_object = json.dumps(results, indent=4)
        with open(args.outf + "/" + args.exp_name + "/loss.json", "w") as outfile:
            outfile.write(json_object)
    
    training_time = time.time() - t
    avg_epoch_time = epoch_time / args.epochs

    dataset_name = args.dataset
    if args.dataset == 'md17':
        dataset_name += '_%s' % args.mol
    elif args.dataset == 'nbody':
        dataset_name += '_%s_%s_%s' % (args.n_isolated, args.n_stick, args.n_hinge)

    args.training_time = training_time
    args.avg_epoch_time = avg_epoch_time
    save_main_results(args.model, dataset_name, args.n_step, args.use_extra_data, best_test_loss, filename=args.filename, full_args=args)

    return best_train_loss, best_val_loss, best_test_loss, best_epoch


def train_one_epoch(model, epoch, loader, training=True):
    if training:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'coord_reg': 0, 'counter': 0, 'loss_stick': 0, 'loss_vel': 0, 'alignment_loss': 0, 'l2_penalty_loss': 0}

    from torch import autograd
    
    for batch_idx, data in enumerate(loader):
        batch_size, n_nodes, n_steps, n_dim = data[0].size()

        data, cfg = data[:-1], data[-1]
        data = [d.to(device) for d in data]
        sloc, svel = data[0].reshape((-1, n_steps, n_dim)), data[1].reshape((-1, n_steps, n_dim))
        data = [d.view(-1, d.shape[-1]) for d in data[2:]]  # construct mini-batch graphs
        loc, vel, edge_attr, charges, loc_end, vel_end, Z = data


        edges = loader.dataset.get_edges(batch_size, n_nodes)
        edges = [edges[0].to(device), edges[1].to(device)]

        cfg = loader.dataset.get_cfg(batch_size, n_nodes, cfg)
        cfg = {_: cfg[_].to(device) for _ in cfg}
        
        
        model.optimizer.zero_grad()

        loc_pred, loc_preds, vel_preds = model(loc, vel, cfg, edges, edge_attr, Z)

        main_loss, alignment_loss, l2_penalty_loss = model.compute_loss(loc_pred, loc_end, loc_preds, vel_preds, sloc, svel, training=training)
        
        loss = 0.
        loss += main_loss
        if training:
            loss += alignment_loss
            if args.l2_penalty==1:
                loss += l2_penalty_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            # torch.nn.utils.clip_grad_value_(model.parameters(), 0.01)
            model.optimizer.step()
        res['loss'] += main_loss.item()*batch_size
        try:
            res['alignment_loss'] += alignment_loss.item()*batch_size
            res['l2_penalty_loss'] += l2_penalty_loss.item()*batch_size
        except:
            print(alignment_loss)
        res['counter'] += batch_size

    if not training:
        prefix = "==> "
    else:
        prefix = ""
    print('%s epoch %d avg main loss: %.5f alignment loss: %.5f l2 penalty loss: %.5f'
          % (prefix+loader.dataset.partition, epoch,
             res['loss'] / res['counter'], res['alignment_loss'] / res['counter'], res['l2_penalty_loss'] / res['counter']))

    return res['loss'] / res['counter']



if __name__ == "__main__":
    print(args)
    best_train_loss, best_val_loss, best_test_loss, best_epoch = main()
    print("best_train = %.6f" % best_train_loss)
    print("best_val = %.6f" % best_val_loss)
    print("best_test = %.6f" % best_test_loss)
    print("best_epoch = %d" % best_epoch)





