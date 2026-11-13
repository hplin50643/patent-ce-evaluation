"""Offline checks of the replacement head, exact loss, token limits and reload."""
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from transformers import ModernBertConfig

from model import ModernContrastiveModel
from protocol import loss_tensor, preference
from run_modernbert import encode, save_last


def tiny_config():
    config = ModernBertConfig(vocab_size=40, hidden_size=16, intermediate_size=24,
                             num_hidden_layers=1, num_attention_heads=2,
                             local_attention=8, global_attn_every_n_layers=1,
                             max_position_embeddings=64, pad_token_id=0,
                             cls_token_id=1, sep_token_id=2,
                             reference_compile=False, attention_dropout=0.,
                             embedding_dropout=0., mlp_dropout=0.)
    config._attn_implementation = 'eager'
    return config


class LengthTokenizer:
    def __call__(self, reference, candidate, **kwargs):
        if kwargs != {'truncation': False, 'return_token_type_ids': False}:
            raise AssertionError('Native pair encoding arguments changed')
        return {'input_ids': [1] + list(range(len(reference) + len(candidate))) + [2],
                'attention_mask': [1] * (len(reference) + len(candidate) + 2)}


class ReplacementTests(unittest.TestCase):
    def test_cls_pooler_tanh_fc_sigmoid_exact_formula(self):
        model = ModernContrastiveModel(tiny_config()).eval()
        hidden = torch.arange(2 * 3 * 16, dtype=torch.float32).reshape(2, 3, 16) / 50

        class Encoder(nn.Module):
            def forward(self, **kwargs):
                return SimpleNamespace(last_hidden_state=hidden)

        model.bert = Encoder()
        with torch.no_grad():
            model.pooler.weight.copy_(torch.eye(16))
            model.pooler.bias.fill_(.1)
            model.fc.weight.fill_(.2)
            model.fc.bias.fill_(-.3)
        actual = model.compute_score(torch.ones(2, 3, dtype=torch.long), torch.ones(2, 3))
        expected = torch.sigmoid(torch.tanh(hidden[:, 0] + .1).sum(-1) * .2 - .3)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        self.assertEqual(actual.shape, (2,))

    def test_tiny_modernbert_full_update_and_saved_scores(self):
        torch.manual_seed(42)
        model = ModernContrastiveModel(tiny_config()).train()
        self.assertFalse(model.config.reference_compile)
        inputs = dict(input_ids=torch.tensor([[1, 7, 8, 2, 0, 0], [1, 4, 5, 6, 2, 0]]),
                      attention_mask=torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]]))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        before = model.bert.embeddings.tok_embeddings.weight.detach().clone()
        loss = loss_tensor(model(**inputs)[:1], model(**inputs)[1:], torch.tensor([1]), .5, .01)
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.bert.embeddings.tok_embeddings.weight))
        model.eval()
        with torch.no_grad():
            expected = model(**inputs)
        with tempfile.TemporaryDirectory(prefix='synthetic_modernbert_export_') as directory:
            model.save_pretrained(directory, safe_serialization=True)
            restored = ModernContrastiveModel.from_pretrained(directory, local_files_only=True,
                        attn_implementation='eager', reference_compile=False).float().eval()
            with torch.no_grad():
                actual = restored(**inputs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertTrue((Path(directory) / 'model.safetensors').is_file())

    def test_equation5_gradient_direction_and_tie_tolerance_unchanged(self):
        a = torch.tensor([.4, .7, .505, .52], requires_grad=True)
        b = torch.tensor([.5, .5, .5, .5])
        labels = torch.tensor([1, -1, 0, 0])
        actual = loss_tensor(a, b, labels, .1, .01)
        torch.testing.assert_close(actual, torch.tensor((.2 + .3 + 0 + .01) / 4))
        actual.backward()
        self.assertLess(float(a.grad[0]), 0)
        self.assertGreater(float(a.grad[1]), 0)
        self.assertEqual(float(a.grad[2]), 0)
        self.assertGreater(float(a.grad[3]), 0)
        self.assertEqual(preference(1e-4), 1)
        self.assertEqual(preference(-1e-4), -1)

    def test_native_pair_lengths_keep_original_rows_and_never_truncate(self):
        row = dict(comparison_id='test_000000', source_split='test', source_row_index=0,
                   gold_id='g', gold_claim='gold', A='aaa', B='bbbb')
        duplicate = dict(row, comparison_id='test_000001', source_row_index=1)
        cache, records = encode([row, duplicate], LengthTokenizer(), max_length=10)
        self.assertEqual(len(cache), 2)
        self.assertEqual(len(records['rows']), 2)
        self.assertEqual(records['rows'][0]['tokens_a'], 9)
        self.assertEqual(records['rows'][0]['tokens_b'], 10)
        self.assertFalse(records['rows'][0]['truncated_a'])
        with self.assertRaisesRegex(ValueError, 'No truncation authorized'):
            encode([row], LengthTokenizer(), max_length=9)

    def test_checkpoint_preserves_optimizer_progress_and_all_rng(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(3, 2), nn.Dropout(.2), nn.Linear(2, 1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)

        def step():
            optimizer.zero_grad(set_to_none=True)
            (model(torch.randn(4, 3)) - random.random() - float(np.random.random())).square().mean().backward()
            optimizer.step()

        step()
        with tempfile.TemporaryDirectory(prefix='synthetic_modernbert_resume_') as directory:
            path = Path(directory) / 'last.pt'
            with patch('torch.cuda.get_rng_state_all', return_value=[]):
                save_last(path, model, optimizer, dict(fingerprint='fixed', epoch=2, next_row=4))
            step()
            expected = {name: value.clone() for name, value in model.state_dict().items()}
            restored = torch.load(path, map_location='cpu', weights_only=False)
            model.load_state_dict(restored['model'])
            optimizer.load_state_dict(restored['optimizer'])
            random.setstate(restored['rng_python'])
            np.random.set_state(restored['rng_numpy'])
            torch.set_rng_state(restored['rng_torch'])
            step()
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
            self.assertEqual(restored['next_row'], 4)
            self.assertEqual(restored['fingerprint'], 'fixed')


if __name__ == '__main__':
    unittest.main()
