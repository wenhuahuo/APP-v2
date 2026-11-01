"""学习分块的求积加权、点序无关性与梯度。"""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))

from learned_tokenizer import LearnedTokenizer  # noqa: E402


class LearnedTokenizerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tokenizer = LearnedTokenizer(input_dim=3, d_model=8, num_tokens=5)
        self.features = torch.randn(20, 3)
        self.weights = torch.rand(20) + 0.1

    def test_quadrature_mass_and_weighted_tokens(self):
        tokens, mass, centres = self.tokenizer(self.features, self.weights)
        self.assertEqual(tokens.shape, (5, 3))
        self.assertEqual(centres.shape, (5, 2))
        self.assertTrue(torch.all(mass > 0))
        torch.testing.assert_close(mass.sum(), self.weights.sum())
        torch.testing.assert_close(centres, tokens[:, :2])

    def test_point_permutation_does_not_change_tokens(self):
        permutation = torch.randperm(len(self.features))
        reference = self.tokenizer(self.features, self.weights)
        permuted = self.tokenizer(self.features[permutation], self.weights[permutation])
        for expected, actual in zip(reference, permuted):
            torch.testing.assert_close(actual, expected)

    def test_assignment_parameters_receive_gradients(self):
        tokens, mass, _ = self.tokenizer(self.features, self.weights)
        (tokens.square().sum() + mass.sum()).backward()
        for parameter in self.tokenizer.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
