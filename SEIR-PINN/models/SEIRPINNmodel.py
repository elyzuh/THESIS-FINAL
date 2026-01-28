import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class MLP(nn.Module):
    """Multi-layer perceptron with tanh activation"""
    def __init__(self, input_dim, output_dim, hidden_dim=50, num_layers=4):
        super(MLP, self).__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)


class Model(nn.Module):
    """
    SEIR-PINN: Physics-Informed Neural Network for SEIR epidemic modeling
    with latent E (Exposed) compartment learning.

    Key Innovation: E is never observed, only inferred from physics constraints.

    Compartments:
        - S(t): Susceptible (inferred from conservation S+E+I+R=N)
        - E(t): Exposed (LATENT - learned via physics)
        - I(t): Infected (OBSERVED from data)
        - R(t): Recovered (inferred from cumulative infections)

    Parameters (time-varying):
        - β(t): Transmission rate [0,1]
        - σ(t): Incubation rate [0,1]
        - γ(t): Recovery rate [0,1]

    Physics Constraints (SEIR ODEs):
        dS/dt = -β*S*I/N
        dE/dt = β*S*I/N - σ*E  # Constrains latent E!
        dI/dt = σ*E - γ*I
        dR/dt = γ*I
    """

    def __init__(self, args, data):
        super(Model, self).__init__()

        self.use_cuda = args.cuda
        self.m = data.m  # Number of regions
        self.hidR = args.hidRNN  # Hidden dimension
        self.num_layers = 4  # MLP depth

        # Population per region (assume constant for now)
        self.register_buffer('N', torch.ones(self.m) * args.population)

        # ===== Compartment Networks =====
        # Each network: time (1D) → compartment values (m regions)
        self.net_S = MLP(1, self.m, self.hidR, self.num_layers)
        self.net_E = MLP(1, self.m, self.hidR, self.num_layers)  # LATENT!
        self.net_I = MLP(1, self.m, self.hidR, self.num_layers)
        self.net_R = MLP(1, self.m, self.hidR, self.num_layers)

        # ===== Parameter Networks =====
        # Each network: time (1D) → parameters (m regions)
        # With sigmoid activation to constrain [0,1]
        self.net_beta_hidden = MLP(1, self.hidR, self.hidR, self.num_layers)
        self.fc_beta = nn.Linear(self.hidR, self.m)

        self.net_sigma_hidden = MLP(1, self.hidR, self.hidR, self.num_layers)
        self.fc_sigma = nn.Linear(self.hidR, self.m)

        self.net_gamma_hidden = MLP(1, self.hidR, self.hidR, self.num_layers)
        self.fc_gamma = nn.Linear(self.hidR, self.m)

        # Optional: Spatial coupling via adjacency matrix
        if hasattr(data, 'adj'):
            self.adj = data.adj
        else:
            self.adj = None

    def forward_compartments(self, t):
        """
        Forward pass for compartments.

        Args:
            t: Time points [batch, 1] or [1] (normalized to [0,1])

        Returns:
            S, E, I, R: Compartment values [batch, m]
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)

        S = self.net_S(t)
        E = self.net_E(t)  # LATENT - learned from physics!
        I = self.net_I(t)
        R = self.net_R(t)

        # Apply softplus to ensure positivity
        S = F.softplus(S)
        E = F.softplus(E)
        I = F.softplus(I)
        R = F.softplus(R)

        return S, E, I, R

    def forward_parameters(self, t):
        """
        Forward pass for time-varying parameters.

        Args:
            t: Time points [batch, 1]

        Returns:
            beta, sigma, gamma: Parameters [batch, m] in range [0,1]
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)

        # Beta (transmission rate)
        h_beta = self.net_beta_hidden(t)
        beta = torch.sigmoid(self.fc_beta(h_beta))

        # Sigma (incubation rate)
        h_sigma = self.net_sigma_hidden(t)
        sigma = torch.sigmoid(self.fc_sigma(h_sigma))

        # Gamma (recovery rate)
        h_gamma = self.net_gamma_hidden(t)
        gamma = torch.sigmoid(self.fc_gamma(h_gamma))

        return beta, sigma, gamma

    def forward(self, x):
        """
        Forward pass for prediction (used during training with data batches).

        This is different from physics_forward which operates on time directly.
        Here we extract the last time step from window and predict next step.

        Args:
            x: Input window [batch, window, m]

        Returns:
            I_pred: Predicted infections [batch, m]
        """
        batch_size = x.size(0)

        # For simplicity, we predict based on the last observed time step
        # In a full implementation, this could use the window history
        # For now, return I prediction at next time step

        # This is a placeholder - actual prediction uses physics-informed approach
        # During training, we use physics loss at collocation points
        # Here we just return current I values for compatibility

        # Use last time step as proxy (will be refined by physics loss)
        I_current = x[:, -1, :]  # [batch, m]

        return I_current

    def physics_forward(self, t):
        """
        Physics-informed forward pass.

        Args:
            t: Time points [n_colloc, 1] with requires_grad=True

        Returns:
            S, E, I, R, beta, sigma, gamma, derivatives
        """
        # Compartments
        S, E, I, R = self.forward_compartments(t)

        # Parameters
        beta, sigma, gamma = self.forward_parameters(t)

        return S, E, I, R, beta, sigma, gamma

    def compute_derivatives(self, compartment, t):
        """
        Compute time derivative using automatic differentiation.

        Args:
            compartment: Tensor [batch, m] that depends on t
            t: Time tensor [batch, 1] with requires_grad=True

        Returns:
            d_compartment_dt: Time derivative [batch, m]
        """
        # Sum over regions to get scalar for each batch
        # This allows grad to compute derivatives
        grad_outputs = torch.ones_like(compartment)

        d_compartment_dt = torch.autograd.grad(
            outputs=compartment,
            inputs=t,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]

        # Expand back to [batch, m] if needed
        if d_compartment_dt.size(1) == 1:
            d_compartment_dt = d_compartment_dt.expand(-1, self.m)

        return d_compartment_dt
