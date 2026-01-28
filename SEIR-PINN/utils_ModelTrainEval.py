import torch
import torch.nn as nn
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


eps = 1e-8


def compute_seir_residuals(model, t_colloc, N):
    """
    Compute SEIR physics residuals at collocation points.

    Args:
        model: SEIR-PINN model
        t_colloc: Collocation time points [n_colloc, 1] with requires_grad=True
        N: Population per region [m]

    Returns:
        residuals: Dict with 'S', 'E', 'I', 'R' residuals
    """
    # Forward pass to get compartments and parameters
    S, E, I, R, beta, sigma, gamma = model.physics_forward(t_colloc)

    # Compute time derivatives using autograd
    dS_dt = model.compute_derivatives(S, t_colloc)
    dE_dt = model.compute_derivatives(E, t_colloc)
    dI_dt = model.compute_derivatives(I, t_colloc)
    dR_dt = model.compute_derivatives(R, t_colloc)

    # SEIR differential equations
    # dS/dt = -beta * S * I / N
    dS_dt_physics = -beta * S * I / N.unsqueeze(0)

    # dE/dt = beta * S * I / N - sigma * E  (This constrains latent E!)
    dE_dt_physics = beta * S * I / N.unsqueeze(0) - sigma * E

    # dI/dt = sigma * E - gamma * I
    dI_dt_physics = sigma * E - gamma * I

    # dR/dt = gamma * I
    dR_dt_physics = gamma * I

    # Compute residuals (should be zero if physics is satisfied)
    residual_S = dS_dt - dS_dt_physics
    residual_E = dE_dt - dE_dt_physics
    residual_I = dI_dt - dI_dt_physics
    residual_R = dR_dt - dR_dt_physics

    residuals = {
        'S': residual_S,
        'E': residual_E,
        'I': residual_I,
        'R': residual_R
    }

    return residuals


def train(loader, data, model, criterion, optim, batch_size, modelName, args):
    """
    PINN training with composite loss: data + physics + IC + conservation.

    Args:
        loader: Data loader
        data: Training data
        model: SEIR-PINN model
        criterion: Loss function (MSE)
        optim: Optimizer
        batch_size: Batch size
        modelName: Model name
        args: Arguments with lambda weights

    Returns:
        avg_loss: Average training loss
    """
    model.train()
    total_loss = 0
    n_samples = 0
    counter = 0

    # Get collocation points for physics loss
    n_colloc = args.collocation_points
    t_colloc = torch.linspace(0, 1, n_colloc, requires_grad=True).unsqueeze(1)
    if args.cuda:
        t_colloc = t_colloc.cuda()

    # Population
    N = model.N

    for inputs in loader.get_batches(data, batch_size, True):
        counter += 1
        X, Y = inputs[0], inputs[1]  # X: [batch, window, m], Y: [batch, m]

        model.zero_grad()

        # ===== 1. Data Loss (on observed I) =====
        # Extract time points from data
        batch_len = X.size(0)

        # For simplicity, predict I at the next time step
        # Use the model's I network at appropriate time points
        # Normalize time indices
        window_size = X.size(1)
        t_data = torch.ones(batch_len, 1) * (window_size / (loader.n - 1))
        if args.cuda:
            t_data = t_data.cuda()

        # Get predictions
        _, _, I_pred, _ = model.forward_compartments(t_data)

        scale = loader.scale.expand(I_pred.size(0), loader.m)
        L_data = criterion(I_pred * scale, Y * scale)

        # ===== 2. Physics Loss (SEIR ODEs at collocation points) =====
        residuals = compute_seir_residuals(model, t_colloc, N)

        L_physics = (criterion(residuals['S'], torch.zeros_like(residuals['S'])) +
                     criterion(residuals['E'], torch.zeros_like(residuals['E'])) +
                     criterion(residuals['I'], torch.zeros_like(residuals['I'])) +
                     criterion(residuals['R'], torch.zeros_like(residuals['R'])))

        # ===== 3. Initial Condition Loss =====
        t0 = torch.zeros(1, 1, requires_grad=True)
        if args.cuda:
            t0 = t0.cuda()

        S0, E0, I0, R0 = model.forward_compartments(t0)

        # Get initial I from data
        I_init = X[0, 0, :]  # First sample, first time step

        # IC constraints: S(0) ≈ N - I(0), E(0) ≈ 0, R(0) ≈ 0
        L_ic = (criterion(S0, (N - I_init).unsqueeze(0)) +
                criterion(E0, torch.zeros_like(E0)) +
                criterion(I0, I_init.unsqueeze(0)) +
                criterion(R0, torch.zeros_like(R0)))

        # ===== 4. Conservation Law =====
        # S + E + I + R ≈ N (approximately, accounting for deaths/births)
        S_c, E_c, I_c, R_c = model.forward_compartments(t_colloc)
        total_pop = S_c + E_c + I_c + R_c
        L_conserve = criterion(total_pop, N.unsqueeze(0).expand(n_colloc, -1))

        # ===== Total Loss =====
        loss = (args.lambda_data * L_data +
                args.lambda_physics * L_physics +
                args.lambda_ic * L_ic +
                args.lambda_conserve * L_conserve)

        # Backward and optimize
        loss.backward()
        optim.step()

        if torch.__version__ < '0.4.0':
            total_loss += loss.data[0]
        else:
            total_loss += loss.item()
        n_samples += (I_pred.size(0) * loader.m)

    return total_loss / n_samples


def evaluate(loader, data, model, evaluateL2, evaluateL1, batch_size, modelName):
    """
    Evaluate SEIR-PINN model on test data.

    Returns standard metrics: RSE, RAE, relative error, correlation, R²
    """
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    counter = 0
    for inputs in loader.get_batches(data, batch_size, False):
        X, Y = inputs[0], inputs[1]

        # Get predictions
        batch_len = X.size(0)
        window_size = X.size(1)
        t_data = torch.ones(batch_len, 1) * (window_size / (loader.n - 1))
        if model.use_cuda:
            t_data = t_data.cuda()

        with torch.no_grad():
            _, _, I_pred, _ = model.forward_compartments(t_data)

        if predict is None:
            predict = I_pred.cpu()
            test = Y.cpu()
        else:
            predict = torch.cat((predict, I_pred.cpu()))
            test = torch.cat((test, Y.cpu()))

        scale = loader.scale.expand(I_pred.size(0), loader.m)

        counter = counter + 1

        if torch.__version__ < '0.4.0':
            total_loss += evaluateL2(I_pred * scale, Y * scale).data[0]
            total_loss_l1 += evaluateL1(I_pred * scale, Y * scale).data[0]
        else:
            total_loss += evaluateL2(I_pred * scale, Y * scale).item()
            total_loss_l1 += evaluateL1(I_pred * scale, Y * scale).item()

        n_samples += (I_pred.size(0) * loader.m)

    rse = math.sqrt(total_loss / n_samples)
    rae = (total_loss_l1 / n_samples)

    predict = predict.data.numpy()
    Ytest = test.data.numpy()

    # Relative Error (%)
    rel_error_list = []
    for i in range(Ytest.shape[1]):  # over regions
        denom = np.abs(Ytest[:, i]) + eps
        rel_err = np.mean(np.abs(predict[:, i] - Ytest[:, i]) / denom)
        rel_error_list.append(rel_err)

    relative_error = np.mean(rel_error_list) * 100

    # R² (Coefficient of Determination)
    r2_list = []
    for i in range(Ytest.shape[1]):  # over regions
        if np.var(Ytest[:, i]) > 0:
            r2_list.append(r2_score(Ytest[:, i], predict[:, i]))

    r2 = np.mean(r2_list) if len(r2_list) > 0 else 0.0

    # Correlation
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)

    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()

    return rse, rae, relative_error, correlation, r2


def GetPrediction(loader, data, model, evaluateL2, evaluateL1, batch_size, modelName, args):
    """
    Get predictions and extract learned latent variables and parameters.

    Returns:
        X_true: Input windows
        I_predict: Predicted infections
        I_true: True infections
        E_latent: Learned exposed compartment (LATENT!)
        S_inferred: Inferred susceptible
        R_inferred: Inferred recovered
        BetaList: Transmission rate β(t)
        SigmaList: Incubation rate σ(t)
        GammaList: Recovery rate γ(t)
    """
    model.eval()
    I_predict = None
    I_true = None
    X_true = None

    E_latent = None
    S_inferred = None
    R_inferred = None
    BetaList = None
    SigmaList = None
    GammaList = None

    counter = 0
    for inputs in loader.get_batches(data, batch_size, False):
        X, Y = inputs[0], inputs[1]

        batch_len = X.size(0)
        window_size = X.size(1)
        t_data = torch.ones(batch_len, 1) * (window_size / (loader.n - 1))
        if model.use_cuda:
            t_data = t_data.cuda()

        with torch.no_grad():
            S, E, I, R, beta, sigma, gamma = model.physics_forward(t_data)

        counter = counter + 1

        if I_predict is None:
            I_predict = I.cpu()
            I_true = Y.cpu()
            X_true = X.cpu()

            E_latent = E.cpu()
            S_inferred = S.cpu()
            R_inferred = R.cpu()
            BetaList = beta.cpu()
            SigmaList = sigma.cpu()
            GammaList = gamma.cpu()
        else:
            I_predict = torch.cat((I_predict, I.cpu()))
            I_true = torch.cat((I_true, Y.cpu()))
            X_true = torch.cat((X_true, X.cpu()))

            E_latent = torch.cat((E_latent, E.cpu()))
            S_inferred = torch.cat((S_inferred, S.cpu()))
            R_inferred = torch.cat((R_inferred, R.cpu()))
            BetaList = torch.cat((BetaList, beta.cpu()))
            SigmaList = torch.cat((SigmaList, sigma.cpu()))
            GammaList = torch.cat((GammaList, gamma.cpu()))

    scale = loader.scale

    # Scale back to original units
    I_predict = (I_predict * scale).detach().numpy()
    I_true = (I_true * scale).detach().numpy()
    X_true = (X_true * scale).detach().numpy()

    E_latent = (E_latent * scale).detach().numpy()
    S_inferred = (S_inferred * scale).detach().numpy()
    R_inferred = (R_inferred * scale).detach().numpy()

    BetaList = BetaList.detach().numpy()
    SigmaList = SigmaList.detach().numpy()
    GammaList = GammaList.detach().numpy()

    return X_true, I_predict, I_true, E_latent, S_inferred, R_inferred, BetaList, SigmaList, GammaList
