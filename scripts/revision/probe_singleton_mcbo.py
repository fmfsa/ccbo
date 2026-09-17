"""Report singleton GP differences against pinned MCBO; never modifies models."""
import json
from pathlib import Path
import torch
from ccbo.qmcbo.quotient import ensure_mcbo_on_path
ensure_mcbo_on_path()
from mcbo.utils.dag import DAG
from mcbo.models.gp_network import GaussianProcessNetwork
from ccbo.qmcbo.qgp_network import JointQuotientGPNetwork

torch.set_default_dtype(torch.float64)
torch.manual_seed(7)
n = 40
x0 = torch.randn(n)
x1 = 0.8*x0 + 0.3*torch.randn(n)
y = x1 + 0.1*torch.randn(n)
train_y = torch.stack([x0, x1, y], dim=-1)
train_x = torch.zeros(n, 6)
profile = {
    "dag": DAG([[], [0], [1]]), "interventional": True,
    "valid_targets": [torch.tensor([0, 0, 0])],
    "additive_noise_dists": [torch.distributions.Normal(0., s) for s in (1., .3, .1)],
    "input_dim": 6, "do_map": lambda x: x, "active_input_indices": [[], [], []],
}
algo = {"algo": "MCBO", "beta": 10.}
torch.manual_seed(123)
base = GaussianProcessNetwork(train_x, train_y, algo, profile)
torch.manual_seed(123)
quotient = JointQuotientGPNetwork(train_x, train_y, algo, profile, [[0], [1], [2]])
rows = []
for k, (a, b) in enumerate(zip(base.node_GPs, quotient.cluster_GPs)):
    query = torch.zeros(7, 1) if k == 0 else torch.linspace(0., 1., 7).reshape(-1, 1)
    with torch.no_grad():
        pa, pb = a.posterior(query), b.posterior(query)
    assert torch.isfinite(pa.mean).all() and torch.isfinite(pb.mean).all()
    rows.append({"node": k, "upstream_model": type(a).__name__,
                 "quotient_model": type(b).__name__,
                 "upstream_likelihood": type(a.likelihood).__name__,
                 "quotient_likelihood": type(b.likelihood).__name__,
                 "max_abs_mean_difference": float((pa.mean-pb.mean).abs().max()),
                 "max_abs_variance_difference": float((pa.variance-pb.variance).abs().max())})
report = {"scope": "One synthetic fit; structural identity is not numerical identity to pinned upstream MCBO",
          "seed_data": 7, "seed_fit": 123, "n": n, "nodes": rows}
out = Path("validation/singleton_mcbo_probe.json")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
