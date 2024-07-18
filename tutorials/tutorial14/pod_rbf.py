import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import torch
from pina.solvers import ReducedOrderModelSolver as ROMSolver
from pina.model.layers import PODBlock, RBFLayer
from pina.model import FeedForward
from pina.geometry import CartesianDomain
from pina.problem import AbstractProblem, ParametricProblem
from pina import Condition, LabelTensor
from pina.callbacks import MetricTracker
from smithers.dataset import NavierStokesDataset
import matplotlib.pyplot as plt

torch.manual_seed(20)

def err(snap, snap_pred):
    # relative errors
    errs = torch.linalg.norm(snap - snap_pred, dim=1)/torch.linalg.norm(snap, dim=1)
    err = float(torch.mean(errs))
    return err

class PODRBF(torch.nn.Module):
    """
    Non-intrusive ROM using POD as reduction and RBF as approximation.
    """
    def __init__(self, pod_rank, rbf_kernel):
        super().__init__()
        self.pod = PODBlock(pod_rank)
        self.rbf = RBFLayer(kernel=rbf_kernel)

    def fit(self, params, snaps):
        self.pod.fit(snaps)
        self.rbf.fit(params, self.pod.reduce(snaps))
        self.snapshots = snaps
        self.params = params

    def forward(self, param_test):
        snaps_pred_test = self.pod.expand(self.rbf(param_test))
        return snaps_pred_test

class POD(PODBlock, torch.nn.Module):
    def __init__(self, pod_rank):
        super().__init__(pod_rank)

    def encode(self, X):
        return self.reduce(X)

    def decode(self, X):
        return self.expand(X)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Reduced Order Model Example')
    parser.add_argument('--reddim', type=int, default=3, help='Reduced dimension')
    parser.add_argument('--field', type=str, default='mag(v)', help='Field to reduce')

    args = parser.parse_args()
    field = args.field
    reddim = args.reddim

    # Import dataset
    data = NavierStokesDataset()
    snapshots = data.snapshots[field]
    params = data.params
    Ndof = snapshots.shape[1]
    Nparams = params.shape[1]

    # Divide dataset into training and testing
    params_train, params_test, snapshots_train, snapshots_test = train_test_split(
            params, snapshots, test_size=0.2, shuffle=False, random_state=42)

    # From numpy to LabelTensor
    params_train = LabelTensor(torch.tensor(params_train, dtype=torch.float32),
            labels=['mu'])
    params_test = LabelTensor(torch.tensor(params_test, dtype=torch.float32),
            labels=['mu'])
    snapshots_train = LabelTensor(torch.tensor(snapshots_train, dtype=torch.float32),
            labels=[f's{i}' for i in range(snapshots_train.shape[1])])
    snapshots_test = LabelTensor(torch.tensor(snapshots_test, dtype=torch.float32),
            labels=[f's{i}' for i in range(snapshots_test.shape[1])])

    # Define ROM problem with only data
    class SnapshotProblem(ParametricProblem):
        input_variables = [f'mu']
        output_variables = [f's{i}' for i in range(Ndof)]
        parameter_domain = CartesianDomain({'mu':[0, 100]})
        conditions = {'data': Condition(input_points=params_train,
            output_points=snapshots_train)}

    problem = SnapshotProblem()
    print(snapshots_train.shape, snapshots_test.shape)

    # POD model
    rom = POD(reddim)
    rom.fit(snapshots_train)
    predicted_snaps_train = rom.decode(rom.encode(snapshots_train))
    predicted_snaps_test = rom.decode(rom.encode(snapshots_test))

    error_train = err(snapshots_train, predicted_snaps_train)
    error_test = err(snapshots_test, predicted_snaps_test)
    print('POD model')
    print('Train relative error:', error_train)
    print('Test relative error:', error_test)

    # POD-RBF model
    rom_rbf = PODRBF(pod_rank=reddim, rbf_kernel='thin_plate_spline')
    rom_rbf.fit(params_train, snapshots_train)
    predicted_snaps_test_rbf = rom_rbf(params_test)
    predicted_snaps_train_rbf = rom_rbf(params_train)

    error_train_rbf = err(snapshots_train, predicted_snaps_train_rbf)
    error_test_rbf = err(snapshots_test, predicted_snaps_test_rbf)
    print('POD-RBF')
    print('Train relative error:', error_train_rbf)
    print('Test relative error:', error_test_rbf)

    # Plot the results
    fig, axs = plt.subplots(1, 3, figsize=(15, 3))
    ind_test = 2
    snap = snapshots_train[ind_test].detach().numpy().reshape(-1)
    pred_snap = predicted_snaps_train[ind_test].detach().numpy().reshape(-1)
    a0 = axs[0].tricontourf(data.triang, snap, levels=16,
            cmap='viridis')
    axs[0].set_title('Truth (mu test={})'.format(params_train[ind_test].detach().numpy()[0]))
    a1 = axs[1].tricontourf(data.triang, pred_snap, levels=16,
            cmap='viridis')
    axs[1].set_title('Prediction (mu test={})'.format(params_train[ind_test].detach().numpy()[0]))
    a2 = axs[2].tricontourf(data.triang, snap - pred_snap, levels=16,
            cmap='viridis')
    axs[2].set_title('Error')
    fig.colorbar(a0, ax=axs[0])
    fig.colorbar(a1, ax=axs[1])
    fig.colorbar(a2, ax=axs[2])
    plt.show()


