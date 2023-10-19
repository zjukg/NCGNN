from numpy import dtype
import numpy as np
import torch
import torch.utils.data
from n_body_system.model import GNN, Baseline, Linear, EGNN_vel, Linear_dynamics, RF_vel, GMN
from torch import nn, optim


class NCGNN(nn.Module):
    def __init__(self, args, device):
        super(NCGNN, self).__init__()
        self.device = device
        
        model = self.init_base_model(args, device)



        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        mse_loss = nn.MSELoss(reduction='none')

        self._builtincoeffs = self.init_newton_cotes_rules()

        self.n_step = args.n_step
        self.use_extra_data = args.use_extra_data
        self.base_model = model
        self.optimizer = optimizer
        self.mse_loss = mse_loss

        self.prev_loss = 100000.

        self.l2_penalty_ratio = 1.

        self.args = args
        self.to(self.device)

    def loss(self, pred, target):
        mse_loss = self.mse_loss(pred, target)
        return torch.mean(mse_loss)

    def adaptive_l2_penalty(self,):
        penalty =  sum([self.l2_penalty_ratio * torch.sum(w.pow(2)) / 2 for w in self.parameters()])
        self.l2_penalty_ratio *= self.args.l2_penalty_decay
        return penalty
    
    def init_newton_cotes_rules(self):
        _builtincoeffs = {
        0: (1,1,[1,],-1,1),
        1: (1,2,[1,1],-1,12),
        2: (1,3,[1,4,1],-1,90),
        3: (3,8,[1,3,3,1],-3,80),
        4: (2,45,[7,32,12,32,7],-8,945),
        5: (5,288,[19,75,50,50,75,19],-275,12096),
        6: (1,140,[41,216,27,272,27,216,41],-9,1400),
        7: (7,17280,[751,3577,1323,2989,2989,1323,3577,751],-8183,518400),
        8: (4,14175,[989,5888,-928,10496,-4540,10496,-928,5888,989],
            -2368,467775),
        9: (9,89600,[2857,15741,1080,19344,5778,5778,19344,1080,
                    15741,2857], -4671, 394240),
        10: (5,299376,[16067,106300,-48525,272400,-260550,427368,
                    -260550,272400,-48525,106300,16067],
            -673175, 163459296),
        11: (11,87091200,[2171465,13486539,-3237113, 25226685,-9595542,
                        15493566,15493566,-9595542,25226685,-3237113,
                        13486539,2171465], -2224234463, 237758976000),
        12: (1, 5255250, [1364651,9903168,-7587864,35725120,-51491295,
                        87516288,-87797136,87516288,-51491295,35725120,
                        -7587864,9903168,1364651], -3012, 875875),
        13: (13, 402361344000,[8181904909, 56280729661, -31268252574,
                            156074417954,-151659573325,206683437987,
                            -43111992612,-43111992612,206683437987,
                            -151659573325,156074417954,-31268252574,
                            56280729661,8181904909], -2639651053,
            344881152000),
        14: (7, 2501928000, [90241897,710986864,-770720657,3501442784,
                            -6625093363,12630121616,-16802270373,19534438464,
                            -16802270373,12630121616,-6625093363,3501442784,
                            -770720657,710986864,90241897], -3740727473,
            1275983280000)
        }

        return _builtincoeffs


    def newton_cotes_rules(self, n_step):
        
        na, da, vi, nb, db = self._builtincoeffs[n_step]

        weights = np.array(vi, dtype=float) 
        error_coeff = nb / db

        return torch.Tensor(weights).to(device=self.device), error_coeff

    def init_base_model(self, args, device):
        if args.dataset == 'md17':
            if args.model == 'gnn':
                model = GNN(input_dim=6, hidden_nf=args.nf, n_layers=args.n_layers, device=device, recurrent=True)
            elif args.model == 'egnn_vel':
                model = EGNN_vel(in_node_nf=2, in_edge_nf=2 + 3, hidden_nf=args.nf, device=device, n_layers=args.n_layers,
                                recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh)
            elif args.model == 'egnn_vel_cons':
                model = EGNN_vel(in_node_nf=2, in_edge_nf=2 + 3, hidden_nf=args.nf, device=device, n_layers=args.n_layers,
                                recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh)
            elif args.model == 'gmn':
                model = GMN(in_node_nf=2, in_edge_nf=2 + 3, hidden_nf=args.nf, device=device, n_layers=args.n_layers,
                            recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh, learnable=args.learnable)
            elif args.model == 'baseline':
                model = Baseline()
            elif args.model == 'linear_vel':
                model = Linear_dynamics(device=device)
            elif args.model == 'linear':
                model = Linear(6, 3, device=device)
            elif args.model == 'rf_vel':
                model = RF_vel(hidden_nf=args.nf, edge_attr_nf=2 + 3, device=device, act_fn=nn.SiLU(), n_layers=args.n_layers)
            else:
                ### add your model here
                raise Exception("Wrong model specified")

        if args.dataset == 'motion':
            if args.model == 'gnn':
                model = GNN(input_dim=6, hidden_nf=args.nf, n_layers=args.n_layers, device=device, recurrent=True)
            elif args.model == 'egnn_vel':
                model = EGNN_vel(in_node_nf=2, in_edge_nf=2 + 1, hidden_nf=args.nf, device=device, n_layers=args.n_layers,
                                    recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh)
            elif args.model == 'egnn_vel_cons':
                model = EGNN_vel(in_node_nf=2, in_edge_nf=2 + 1, hidden_nf=args.nf, device=device, n_layers=args.n_layers,
                                    recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh)
            elif args.model == 'gmn':
                model = GMN(in_node_nf=2, in_edge_nf=2 + 1, hidden_nf=args.nf, device=device, n_layers=args.n_layers,
                            recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh, learnable=args.learnable)
            elif args.model == 'baseline':
                model = Baseline()
            elif args.model == 'linear_vel':
                model = Linear_dynamics(device=device)
            elif args.model == 'linear':
                model = Linear(6, 3, device=device)
            elif args.model == 'rf_vel':
                model = RF_vel(hidden_nf=args.nf, edge_attr_nf=2 + 1, device=device, act_fn=nn.SiLU(), n_layers=args.n_layers)
            else:
                raise Exception("Wrong model specified")    
        
        if args.dataset == 'nbody':
            if args.model == 'gnn':
                model = GNN(input_dim=6, hidden_nf=args.nf, n_layers=args.n_layers, device=device, recurrent=True)
            elif args.model == 'egnn_vel':
                model = EGNN_vel(in_node_nf=1, in_edge_nf=2 + 1, hidden_nf=args.nf, device=device, n_layers=args.n_layers,
                                recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh)
            elif args.model == 'egnn_vel_cons':
                model = EGNN_vel(in_node_nf=1, in_edge_nf=2 + 1, hidden_nf=args.nf, device=device, n_layers=args.n_layers,
                                recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh)
            elif args.model == 'gmn':
                model = GMN(in_node_nf=1, in_edge_nf=2 + 1, hidden_nf=args.nf, device=device, n_layers=args.n_layers,
                            recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh, learnable=args.learnable)
            elif args.model == 'baseline':
                model = Baseline()
            elif args.model == 'linear_vel':
                model = Linear_dynamics(device=device)
            elif args.model == 'linear':
                model = Linear(6, 3, device=device)
            elif args.model == 'rf_vel':
                model = RF_vel(hidden_nf=args.nf, edge_attr_nf=2 + 1, device=device, act_fn=nn.SiLU(), n_layers=args.n_layers)
            else:
                raise Exception("Wrong model specified")

        
        print(model)

        return model


    def forward(self, loc, vel, cfg, edges, edge_attr, Z):
        n_step = self.n_step

        loc_pred, vel_pred = loc.detach(), vel.detach()

        loc_preds, vel_preds = [loc_pred, ] , [vel_pred, ]
        for step in range(n_step+1):
            concated_feature = self.concat_node_feature(loc_pred, vel_pred, edges, edge_attr, Z, cfg)
            loc_pred, vel_pred = self.base_model(*concated_feature)
            loc_preds.append(torch.clone(loc_pred))
            vel_preds.append(torch.clone(vel_pred))
        loc_preds.pop(0)
        vel_preds.pop(0)

        loc_preds, vel_preds = torch.stack(loc_preds, 1), torch.stack(vel_preds, 1) # [N, n_step, 3]

        weights, error_coeff = self.newton_cotes_rules(n_step)
        quad = torch.sum(vel_preds * weights.reshape([-1, 1]), 1) / weights.sum()
        loc_pred = loc_preds[:, 0] + quad
        

        return loc_pred, loc_preds, vel_preds

    

    
    def compute_loss(self, loc_pred, loc_end, loc_preds, vel_preds, sloc, svel, training=True):
        main_loss = self.loss(loc_pred, loc_end)
        weights, error_coeff = self.newton_cotes_rules(self.n_step)
        alignment_losses = [self.loss(vel_preds[:, i], svel[:, i])*weights[i] for i in range(len(weights))]

        alignment_loss = alignment_losses[-1]
        if self.use_extra_data:
            alignment_loss += sum(alignment_losses[1:-1])

        alignment_loss *= self.args.alignment_loss_weight

        l2_penalty_loss = self.adaptive_l2_penalty()

        return main_loss, alignment_loss, l2_penalty_loss

    def concat_node_feature(self, loc, vel, edges, edge_attr, Z, cfg):
        v_norm = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
        if self.args.dataset != 'nbody':
            node_feat = torch.cat((v_norm, Z / Z.max()), dim=-1)
        else:
            node_feat = v_norm
        rows, cols = edges
        loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
        edge_feat = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties

        

        if self.args.model == 'gmn':
            return (node_feat, loc, edges, vel, cfg, edge_feat)
        elif self.args.model == 'egnn_vel':
            return (node_feat, loc, edges, vel, edge_feat) 
        elif self.args.model == 'gnn':
            return (torch.cat([loc, vel], dim=1), edges, edge_attr)
        elif self.args.model == 'baseline':
            return (loc,)
        elif self.args.model == 'linear_vel':
            return (loc, vel)
        elif self.args.model == 'rf_vel':
            return (v_norm, loc.detach(), edges, vel, edge_feat)