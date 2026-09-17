"""Scientific-contract gates for the corrected MCBO adapter.

Run with the pinned MCBO runtime; test collection explicitly skips absent stack.
"""
import json
import pytest

from ccbo.qmcbo.corrected import keyed_seed


def test_semantic_seeds_stable_and_distinct():
    assert keyed_seed(3, "init", [0, 1]) == keyed_seed(3, "init", [0, 1])
    assert len({keyed_seed(3, phase, [0, 1]) for phase in ("init", "score", "measure")}) == 3


@pytest.fixture
def stack():
    from ccbo.qmcbo.quotient import ensure_mcbo_on_path
    try:
        ensure_mcbo_on_path()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    torch = pytest.importorskip("torch")
    pytest.importorskip("botorch.sampling.samplers")
    import functions
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield torch, functions
    torch.set_default_dtype(previous)


def test_mapping_clamps_physical_values_once_and_preserves_eta(stack):
    torch, functions = stack
    from mcbo.models import gp_network as g
    from ccbo.qmcbo.corrected import corrected_backend
    env = functions.ToyGraph(noise_scales=0.0)
    X = torch.tensor([[[0.2, .4, .5, .1, .7, .9]]])
    original = g.HallucinatedGaussianProcessNetwork.__init__
    with corrected_backend():
        p = g.HallucinatedGaussianProcessNetwork([None]*3, X, {"beta":10},
                    env.get_env_profile(), ([[], [], []], [[], [], []]), torch.ones(3))
        assert torch.equal(p.X, env.do_map(X)[..., :3])
        assert torch.equal(p.eta, (X[..., 3:] - .5)*2)
        draw = p.rsample(torch.Size([2]))
        assert torch.equal(draw[0], p.X)
    assert g.HallucinatedGaussianProcessNetwork.__init__ is original
    assert torch.equal(X[..., :2], torch.tensor([[[.2, .4]]]))


def test_backend_and_rng_restore_on_failure(stack):
    torch, _ = stack
    import numpy as np
    from mcbo.models import gp_network as g
    from ccbo.qmcbo.corrected import corrected_backend, isolated_rng
    original = g.HallucinatedGaussianProcessNetwork.rsample
    torch_state, np_state = torch.get_rng_state().clone(), np.random.get_state()
    with pytest.raises(RuntimeError):
        with corrected_backend(), isolated_rng(99):
            torch.randn(9); np.random.rand(8)
            raise RuntimeError("probe")
    assert g.HallucinatedGaussianProcessNetwork.rsample is original
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert np.array_equal(np.random.get_state()[1], np_state[1])


def test_shared_target_initialization_survives_menu_reorder(stack):
    torch, functions = stack
    from ccbo.qmcbo.corrected import initial_design, action_menu
    p = functions.ToyGraph(noise_scales=0.).get_env_profile()
    p["valid_targets"] = action_menu(p, [[0,1],[2]], "full")
    a = initial_design(p, 7)
    p["valid_targets"] = p["valid_targets"][::-1]
    b = initial_design(p, 7)
    for target in p["valid_targets"]:
        am = (a[:, :3] == target).all(-1); bm = (b[:, :3] == target).all(-1)
        assert torch.equal(a[am], b[bm])
    coarse = dict(p, valid_targets=action_menu(p, [[0,1],[2]], "coarse"))
    c = initial_design(coarse, 7)
    for target in coarse["valid_targets"]:
        assert torch.equal(a[(a[:,:3]==target).all(-1)], c[(c[:,:3]==target).all(-1)])


def test_scalar_models_equal_and_constant_normalization_finite(stack):
    torch, functions = stack
    from mcbo.models.gp_network import GaussianProcessNetwork
    from ccbo.qmcbo.qgp_network import JointQuotientGPNetwork
    from ccbo.qmcbo.corrected import corrected_backend, isolated_rng
    profile = functions.ToyGraph(noise_scales=0.).get_env_profile()
    # Constant root parent; tests the previously unguarded zero span too.
    Y = torch.stack([torch.zeros(12), torch.linspace(-1,1,12), torch.linspace(-1,1,12)**2], -1)
    X = torch.zeros(12,6)
    cfg = {"algo":"MCBO", "beta":10., "scalar_fit_policy":"upstream-fixed"}
    with corrected_backend(), isolated_rng(88):
        a = GaussianProcessNetwork(X, Y, cfg, profile)
    with corrected_backend(), isolated_rng(88):
        b = JointQuotientGPNetwork(X, Y, cfg, profile, [[0],[1],[2]])
    for ga, gb in zip(a.node_GPs, b.cluster_GPs):
        assert type(ga) is type(gb)
        assert torch.equal(ga.train_inputs[0], gb.train_inputs[0])
        assert torch.isfinite(ga.train_inputs[0]).all()
        query = torch.tensor([[0.], [.5], [1.]])
        pa, pb = ga.posterior(query), gb.posterior(query)
        torch.testing.assert_close(pa.mean, pb.mean, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(pa.variance, pb.variance, rtol=1e-9, atol=1e-9)
    a.set_target(torch.tensor([1,0,0])); b.set_target(torch.tensor([1,0,0]))
    candidate = torch.tensor([[[.2,.4,.5,.5,.5,.5]]])
    with corrected_backend(77):
        sa = a.posterior(candidate).rsample(torch.Size([4]))
        sb = b.posterior(candidate).rsample(torch.Size([4]))
    torch.testing.assert_close(sa, sb, rtol=1e-9, atol=1e-9)


def test_acquisition_noise_repeatable_without_advancing_rng(stack):
    torch, functions = stack
    from types import SimpleNamespace
    from mcbo.models.gp_network import HallucinatedGaussianProcessNetwork
    from ccbo.qmcbo.corrected import corrected_backend
    class FakeGP:
        def posterior(self, X):
            return SimpleNamespace(mean=X[..., :1] * .1,
                                   variance=torch.ones_like(X[..., :1]) * .01)
    env = functions.ToyGraph(noise_scales=.3)
    X = torch.full((1,1,6), .4, requires_grad=True)
    norm = ([[],[torch.tensor(-5.)],[torch.tensor(-5.)]],
            [[],[torch.tensor(5.)],[torch.tensor(20.)]])
    with corrected_backend(99):
        p = HallucinatedGaussianProcessNetwork([FakeGP()]*3, X, {"beta":10},
                 env.get_env_profile(), norm, torch.tensor([1,0,0]))
        state = torch.get_rng_state().clone()
        a = p.rsample(torch.Size([4])); ga = torch.autograd.grad(a.sum(), X, retain_graph=True)[0]
        b = p.rsample(torch.Size([4])); gb = torch.autograd.grad(b.sum(), X)[0]
        assert torch.equal(a,b) and torch.equal(ga,gb)
        assert torch.equal(torch.get_rng_state(),state)


def test_acquisition_candidate_permutation_and_batch_size_invariance(stack):
    torch, functions = stack
    from types import SimpleNamespace
    from mcbo.models.gp_network import HallucinatedGaussianProcessNetwork
    from ccbo.qmcbo.corrected import corrected_backend
    class FakeGP:
        def posterior(self, X):
            return SimpleNamespace(mean=X[..., :1] * .1,
                                   variance=torch.ones_like(X[..., :1]) * .01)
    env = functions.ToyGraph(noise_scales=.3)
    X = torch.tensor([[[.2,.3,.4,.5,.6,.7]], [[.6,.7,.2,.8,.4,.1]], [[.4,.9,.7,.1,.2,.3]]])
    norm = ([[],[torch.tensor(-5.)],[torch.tensor(-5.)]],
            [[],[torch.tensor(5.)],[torch.tensor(20.)]])
    def draw(x):
        p = HallucinatedGaussianProcessNetwork([FakeGP()]*3, x, {"beta":10},
                    env.get_env_profile(), norm, torch.tensor([1,0,0]))
        return p.rsample(torch.Size([8]))
    with corrected_backend(55):
        a = draw(X)
        perm = torch.tensor([2,0,1])
        assert torch.equal(a[:,perm], draw(X[perm]))
        assert torch.equal(a[:, :1], draw(X[:1]))
        assert torch.equal(a[:, 1:2], draw(X[1:2]))


def test_joint_acquisition_candidate_permutation_and_batch_size_invariance(stack):
    torch, functions = stack
    from types import SimpleNamespace
    from ccbo.qmcbo.qgp_network import JointHallucinatedNetwork
    from ccbo.qmcbo.corrected import corrected_backend
    class FakeGP:
        likelihood = SimpleNamespace(task_noise_covar=torch.tensor([[.2,.1],[.1,.3]]), has_global_noise=False)
        def posterior(self, X):
            return SimpleNamespace(mean=X[..., :1].expand(*X.shape[:-1], 2) * .1,
                                   variance=torch.ones(*X.shape[:-1], 2) * .01)
    env = functions.ToyGraph(noise_scales=.3)
    # Joint source block sampled, scalar target clamped: no fitted model needed.
    net = SimpleNamespace(algo_profile={"algo":"MCBO", "beta":10., "scalar_fit_policy":"upstream-fixed"},
            env_profile=env.get_env_profile(), cluster_GPs=[FakeGP(),None],
            target=[0,0,1], norm_lo=[None,None], norm_hi=[None,None],
            cluster_inputs=[[],[0,1]], partition=[[0,1],[2]], cluster_order=[0,1])
    X = torch.tensor([[[.2,.3,.4,.5,.6,.7]], [[.6,.7,.2,.8,.4,.1]], [[.4,.9,.7,.1,.2,.3]]])
    def draw(x):
        return JointHallucinatedNetwork(net,x).rsample(torch.Size([8]))
    with corrected_backend(55):
        a = draw(X)
        perm = torch.tensor([2,0,1])
        assert torch.equal(a[:,perm], draw(X[perm]))
        assert torch.equal(a[:, :1], draw(X[:1]))


def test_scoring_sample_count_cannot_change_measured_data_or_next_action(stack, monkeypatch, tmp_path):
    torch, _ = stack
    from mcbo import mcbo_trial as trial
    from ccbo.qmcbo.corrected import run_corrected
    calls = []
    def cheap_policy(X, Y, obs, cfg, profile, *unused):
        calls.append((X.clone(),Y.clone()))
        # Depends on learning data and algorithm RNG, so either leakage shows.
        values = (torch.rand(1,6) + Y.mean().abs().remainder(1)).remainder(1)
        return torch.cat([profile["valid_targets"][0].unsqueeze(0), values], -1), None
    monkeypatch.setattr(trial, "get_new_suggested_point", cheap_policy)
    run_corrected("PSAGraph", "MCBO", 8, 3, str(tmp_path/"a"), score_samples=4)
    first = calls[:]; calls.clear()
    run_corrected("PSAGraph", "MCBO", 8, 3, str(tmp_path/"b"), score_samples=17)
    assert len(first) == len(calls) == 3
    for (xa,ya),(xb,yb) in zip(first,calls):
        assert torch.equal(xa,xb) and torch.equal(ya,yb)
    a = json.loads(next((tmp_path/"a").glob("*.decisions.json")).read_text())
    b = json.loads(next((tmp_path/"b").glob("*.decisions.json")).read_text())
    for ea, eb in zip(a["events"],b["events"]):
        assert ea["X"] == eb["X"]
        assert ea["network_observation"] == eb["network_observation"]
    assert a["unit"]["initial_rows"] == 11


def test_real_one_iteration_reaches_measurement_without_wandb_init(stack, monkeypatch, tmp_path):
    """Actual upstream optimizer chain, including its separately imported log."""
    import importlib
    import wandb
    from mcbo import mcbo_trial as trial
    from ccbo.qmcbo.corrected import run_corrected
    optimizer = importlib.import_module("mcbo.acquisition_function_optimization.optimize_acqf")
    original_trial_binding, original_optimizer_binding = trial.wandb, optimizer.wandb
    def forbidden(*args, **kwargs):
        raise AssertionError("Real wandb initialization/logging is forbidden")
    monkeypatch.setattr(wandb, "init", forbidden)
    monkeypatch.setattr(wandb, "log", forbidden)
    info = run_corrected("ToyGraph", "MCBO", 1000, 1, str(tmp_path), menu="full")
    result = json.loads(next(tmp_path.glob("*.decisions.json")).read_text())
    assert sum(e["phase"] == "sequential" for e in result["events"]) == 1
    assert result["events"][-1]["acquisition_diagnostics"]
    assert info["num_trials"] == 1
    assert trial.wandb is original_trial_binding
    assert optimizer.wandb is original_optimizer_binding
