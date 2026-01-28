#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import argparse
import math
import time
import sys
import os

import torch
import torch.nn as nn
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from utils_ModelTrainEval import *
import Optim

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

parser = argparse.ArgumentParser(description='SEIR-PINN: Physics-Informed Neural Network for Epidemic Forecasting')

# --- Data options
parser.add_argument('--data', type=str, required=True, help='location of the data file')
parser.add_argument('--train', type=float, default=0.6, help='how much data used for training')
parser.add_argument('--valid', type=float, default=0.2, help='how much data used for validation')
parser.add_argument('--model', type=str, default='SEIRPINNmodel', help='model to select')

# --- Model options
parser.add_argument('--hidRNN', type=int, default=50, help='number of hidden units in MLP')
parser.add_argument('--output_fun', type=str, default=None, help='the output function of neural net')

# --- PINN-specific options
parser.add_argument('--lambda_data', type=float, default=1.0, help='weight for data loss')
parser.add_argument('--lambda_physics', type=float, default=1.0, help='weight for physics loss')
parser.add_argument('--lambda_ic', type=float, default=0.1, help='weight for initial condition loss')
parser.add_argument('--lambda_conserve', type=float, default=0.1, help='weight for conservation law loss')
parser.add_argument('--collocation_points', type=int, default=1000, help='number of collocation points for physics loss')
parser.add_argument('--population', type=float, default=1e6, help='population per region (for normalization)')

# --- Logging options
parser.add_argument('--save_dir', type=str, default='./save', help='dir path to save the final model')
parser.add_argument('--save_name', type=str, default='seir_pinn', help='filename to save the final model')

# --- Optimization options
parser.add_argument('--optim', type=str, default='adam', help='optimization method')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit')
parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay (L2 regularization)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')

# --- Prediction options
parser.add_argument('--horizon', type=int, default=12, help='predict horizon')
parser.add_argument('--window', type=int, default=24 * 7, help='window size')
parser.add_argument('--metric', type=int, default=1, help='whether (1) or not (0) normalize rse and rae with global variance/deviation')
parser.add_argument('--normalize', type=int, default=0, help='the normalized method used, detail in the utils.py')

# --- Hardware options
parser.add_argument('--seed', type=int, default=54321, help='random seed')
parser.add_argument('--gpu', type=int, default=None, help='GPU number to use')
parser.add_argument('--cuda', type=str, default=None, help='use gpu or not')

# --- Misc
parser.add_argument('--sim_mat', type=str, default=None, help='file of similarity measurement (optional for spatial coupling)')

args = parser.parse_args()
print(args)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

args.cuda = args.gpu is not None
if args.cuda:
    torch.cuda.set_device(args.gpu)

# Set the random seed manually for reproducibility
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

# Load data
Data = Data_utility(args)

# Dynamically import the model module
model_name = args.model
if model_name == 'SEIRPINN':
    model_name = 'SEIRPINNmodel'

model_module = __import__('models.' + model_name, fromlist=[model_name])
model = model_module.Model(args, Data)
print('model:', model)

if args.cuda:
    model.cuda()

nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)

# Loss functions
criterion = nn.MSELoss(reduction='sum')
evaluateL2 = nn.MSELoss(reduction='sum')
evaluateL1 = nn.L1Loss(reduction='sum')

if args.cuda:
    criterion = criterion.cuda()
    evaluateL1 = evaluateL1.cuda()
    evaluateL2 = evaluateL2.cuda()

best_val = 10000000

# Optimizer
optim = Optim.Optim(
    model.parameters(), args.optim, args.lr, args.clip, model.named_parameters(), weight_decay=args.weight_decay,
)

ifPlot = 0

# Training loop
try:
    print('begin training')
    print('=' * 89)
    print('SEIR-PINN: Physics-Informed Neural Network with Latent E Learning')
    print('=' * 89)

    # Plot convergence
    x_epoch = []
    y_train_loss = []
    y_validate_loss = []

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        # Train with physics loss
        train_loss = train(Data, Data.train, model, criterion, optim, args.batch_size, args.model, args)

        # Validate
        val_loss, val_rae, val_rel_error, val_corr, val_r2 = evaluate(
            Data, Data.valid, model, evaluateL2, evaluateL1, args.batch_size, args.model
        )

        print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.8f} | '
              'valid rse {:5.4f} | valid rae {:5.4f} | valid relative error: {:5.2f}% | valid corr {:5.4f} | valid r2 {:5.4f}'
              .format(epoch, (time.time() - epoch_start_time),
                      train_loss, val_loss, val_rae, val_rel_error, val_corr, val_r2))

        if math.isnan(train_loss):
            print('ERROR: Training loss is NaN. Try reducing learning rate or adjusting loss weights.')
            sys.exit()

        # Plot the convergence
        x_epoch.append(epoch)
        y_train_loss.append(train_loss)
        y_validate_loss.append(val_loss)

        # Save the model if the validation loss is the best we've seen so far
        if val_loss < best_val:
            best_val = val_loss
            model_path = '%s/%s.pt' % (args.save_dir, args.save_name)

            with open(model_path, 'wb') as f:
                torch.save(model.state_dict(), f)

            print('best validation')
            test_acc, test_rae, test_relative_error, test_corr, test_r2 = evaluate(
                Data, Data.test, model, evaluateL2, evaluateL1, args.batch_size, args.model
            )
            print("test rse {:5.4f} | test rae {:5.4f} | test relative error {:5.2f}% | test corr {:5.4f} | test r2 {:5.4f}"
                  .format(test_acc, test_rae, test_relative_error, test_corr, test_r2))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model
model_path = '%s/%s.pt' % (args.save_dir, args.save_name)
with open(model_path, 'rb') as f:
    model.load_state_dict(torch.load(f))

print("=" * 89)
print("Final Test Results")
print("=" * 89)
test_acc, test_rae, test_relative_error, test_corr, test_r2 = evaluate(
    Data, Data.test, model, evaluateL2, evaluateL1, args.batch_size, args.model
)
print("test rse {:5.4f} | test rae {:5.4f} | test relative error {:5.2f}% | test corr {:5.4f} | test r2 {:5.4f}"
      .format(test_acc, test_rae, test_relative_error, test_corr, test_r2))

print("=" * 89)
print("SEIR-PINN Training Complete!")
print("Model saved to: %s" % model_path)
print("=" * 89)
print("\nKey Outputs:")
print("- Learned latent E(t) compartment (Exposed)")
print("- Time-varying β(t) (transmission rate)")
print("- Time-varying σ(t) (incubation rate)")
print("- Time-varying γ(t) (recovery rate)")
print("\nUse GetPrediction() to extract learned variables for visualization.")
