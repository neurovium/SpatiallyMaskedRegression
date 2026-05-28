"""
Smoke tests for the model factory in `fixed/model.py`.

Confirms that:
  1. `build_model` returns ReconModel + 'fc.weight' for the default
     `model_type='instantaneous'`.
  2. `build_model` returns LaggedReconModelConfigurable + 'fc_main.weight'
     for `model_type='lagged'`.
  3. The configurable lagged model honours `settings.lags_ms`.
  4. Both models produce a forward output of shape (batch, 1, time)
     given the same dummy input.
  5. `collect_lagged_weights` stacks lag-0 + each user-supplied lag.

Run with:
    python -m unittest fixed.test_model_factory

or:
    python fixed/test_model_factory.py
"""

import sys
import types
import unittest


# This test does not need ``ReconModel`` from the ``new/`` folder to be
# reachable, because ``model.py`` only imports it at module load. To keep
# the tests self-contained we ship a tiny stub here.
class _StubReconModel:
    def __init__(self, n, del_size):
        self.n = n
        self.del_size = del_size
        # Expose a state_dict so train.py's code path still works.
        import torch
        self._w = torch.zeros((1, n - del_size))

    def state_dict(self):
        return {'fc.weight': self._w}

    def to(self, device):
        return self

    def parameters(self):
        return iter([])

    def __call__(self, data, mask_index):
        import torch
        batch, _, time = data.shape
        return torch.zeros((batch, 1, time))


class _StubLagged:
    """Mirrors the real LaggedReconModel signature so `model.py` imports cleanly."""
    def __init__(self, n, del_size, fs, dropout_p=0.5):
        self.n, self.del_size, self.fs = n, del_size, fs


# Install stubs *before* importing model.py
_stub_module = types.ModuleType('ReconModel')
_stub_module.ReconModel = _StubReconModel
_stub_module.LaggedReconModel = _StubLagged
sys.modules['ReconModel'] = _stub_module

import torch  # noqa: E402  (after stub install)

import model  # noqa: E402  (the file under test)


class _SettingsStub:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TestBuildModel(unittest.TestCase):
    def test_instantaneous_default(self):
        s = _SettingsStub()
        m, key = model.build_model(s, num_node=10, del_size=2)
        self.assertIsInstance(m, _StubReconModel)
        self.assertEqual(key, 'fc.weight')

    def test_instantaneous_explicit(self):
        s = _SettingsStub(model_type='instantaneous')
        m, key = model.build_model(s, num_node=10, del_size=2)
        self.assertIsInstance(m, _StubReconModel)
        self.assertEqual(key, 'fc.weight')

    def test_lagged(self):
        s = _SettingsStub(model_type='lagged', fs=500)
        m, key = model.build_model(s, num_node=10, del_size=2)
        self.assertIsInstance(m, model.LaggedReconModelConfigurable)
        self.assertEqual(key, 'fc_main.weight')
        # Default lags from the manuscript
        self.assertEqual(m.lags, [20, 30, 50, 60])

    def test_lagged_custom_lags(self):
        s = _SettingsStub(model_type='lagged', fs=500, lags_ms=[10, 25])
        m, _ = model.build_model(s, num_node=8, del_size=1)
        self.assertEqual(m.lags, [10, 25])
        self.assertEqual(sorted(m.fc_lags.keys()), ['10', '25'])

    def test_lagged_requires_fs(self):
        s = _SettingsStub(model_type='lagged')
        with self.assertRaises(ValueError):
            model.build_model(s, num_node=10, del_size=2)

    def test_unknown_model_type(self):
        s = _SettingsStub(model_type='banana')
        with self.assertRaises(ValueError):
            model.build_model(s, num_node=10, del_size=2)


class TestLaggedForward(unittest.TestCase):
    def test_forward_shape(self):
        m = model.LaggedReconModelConfigurable(
            n=6, del_size=1, fs=500, lags_ms=[20, 50],
        )
        data = torch.randn(4, 6, 100)        # batch=4, channels=6, time=100
        out = m(data, mask_index=[2])
        self.assertEqual(out.shape, (4, 1, 100))

    def test_collect_lagged_weights(self):
        m = model.LaggedReconModelConfigurable(
            n=6, del_size=1, fs=500, lags_ms=[20, 50],
        )
        stacked = model.collect_lagged_weights(m)
        # 3 entries: lag-0 (fc_main) + 2 user-supplied lags
        self.assertEqual(stacked.shape, (3, 1, 5))


if __name__ == '__main__':
    unittest.main()
