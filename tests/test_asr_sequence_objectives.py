import unittest

import torch

from therapist_harness.asr_sequence_objectives import (
    character_costs, mean_preference_loss, minimum_character_risk,
)


class SequenceObjectivesTest(unittest.TestCase):
    def test_ipo_finite_optimum_and_frozen_reference(self):
        ref = torch.zeros(2, dtype=torch.double, requires_grad=True)
        counts = torch.ones(2)
        gradients = []
        for margin in [0., .5, 1.]:
            policy = torch.tensor([margin, 0.], dtype=torch.double, requires_grad=True)
            loss, _ = mean_preference_loss(policy, ref, counts, objective="ipo", beta=1.)
            loss.backward()
            gradients.append(policy.grad[0].item())
        self.assertLess(gradients[0], 0)
        self.assertEqual(gradients[1], 0)
        self.assertGreater(gradients[2], 0)
        self.assertIsNone(ref.grad)

    def test_initial_gradients_match_with_unequal_lengths(self):
        gradients = []
        for objective in ["dpo", "ipo"]:
            policy = torch.tensor([-3., -7.], dtype=torch.double, requires_grad=True)
            loss, _ = mean_preference_loss(policy, policy.detach(), torch.tensor([3, 8]),
                                           objective=objective, beta=1.)
            gradients.append(torch.autograd.grad(loss, policy)[0])
        self.assertTrue(torch.equal(*gradients))

    def test_risk_gradient_favors_fewer_errors(self):
        logp = torch.tensor([-2., -3., -4.], dtype=torch.double, requires_grad=True)
        loss, expected, q = minimum_character_risk(logp, [1., 3., 5.])
        grad = torch.autograd.grad(loss, logp)[0]
        analytical = q.detach() * (torch.tensor([1., 3., 5.]) - expected.detach())
        self.assertTrue(torch.allclose(grad, analytical))
        self.assertLess(grad[0], 0)
        self.assertGreater(grad[-1], 0)
        self.assertTrue(torch.autograd.gradcheck(
            lambda x: minimum_character_risk(x, [1., 3., 5.])[0], (logp,)))
        shifted = minimum_character_risk(logp + 100, [1., 3., 5.])[1]
        self.assertTrue(torch.allclose(shifted, expected))

    def test_equal_cost_and_single_candidate_have_zero_gradient(self):
        for values in [[0.], [0., 0.], [3., 3., 3.]]:
            x = torch.arange(len(values), dtype=torch.double, requires_grad=True)
            loss = minimum_character_risk(x, values)[0]
            self.assertEqual(torch.count_nonzero(torch.autograd.grad(loss, x)[0]).item(), 0)

    def test_cantonese_character_cost_matches_evaluation(self):
        self.assertEqual(character_costs("幫我關門。", ["幫我關門", "幫我開門", ""], normalized=False), [0, 1, 4])
        self.assertEqual(character_costs("幫我關門。", ["幫我開門"], normalized=True), [.25])


if __name__ == "__main__":
    unittest.main()
