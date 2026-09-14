import unittest

import torch
from torch import nn

from .halton_sparse2d_adapter import HaltonFullContextSparse2DAdapter
from .unified_fullctx_maskgit import (
    OneDConditionedHaltonFullContextSparse2DAdapter,
    UnifiedHaltonMixMaskGIT,
)


class TinyOneD(nn.Module):
    titok_vocab_size = 17
    titok_num_tokens = 32
    mask_token_1d = 17

    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim
        self.embedding_1d = nn.Embedding(self.titok_vocab_size + 1, dim)
        self.pos_embedding_1d = nn.Embedding(self.titok_num_tokens, dim)
        self.class_embedding = nn.Embedding(8, dim)
        self.class_pos_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        self.modality_embedding = nn.Embedding(2, dim)
        self.output_1d = nn.Linear(dim, self.titok_vocab_size)

    def _one_d_embeddings(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        return self.embedding_1d(tokens) + self.pos_embedding_1d(positions)[None] + self.modality_embedding.weight[0][None, None]

    def _class_tokens(self, labels, force_drop_ids=None):
        if force_drop_ids is not None:
            labels = torch.where(force_drop_ids.bool(), torch.zeros_like(labels), labels)
        return self.class_embedding(labels)[:, None] + self.class_pos_embedding + self.modality_embedding.weight[0][None, None]

    def _run_backbone(self, hidden, freqs, valid):
        if freqs is not None or hidden.shape[:2] != valid.shape:
            raise ValueError("bad tiny 1D inputs")
        return hidden * valid[..., None]

    def forward_1d(self, input_tokens, labels, force_drop_ids=None):
        return self.output_1d(self._one_d_embeddings(input_tokens))


def tiny_2d(cls):
    return cls(dim=32, depth=1, heads=4, dropout=0.0).eval()


class UnifiedHaltonMixTests(unittest.TestCase):
    def batch(self):
        z1 = torch.randint(0, 17, (2, 32))
        z2 = torch.randint(0, 128, (2, 5))
        index = torch.tensor([[0, 17, 85, 170, 255], [1, 2, 3, 4, 5]])
        valid = torch.ones_like(index, dtype=torch.bool)
        labels = torch.tensor([1, 2])
        base = torch.randn(2, 256, 16, 16)
        return z1, z2, index, valid, labels, base

    def test_zero_init_context_preserves_halton_logits(self):
        torch.manual_seed(7)
        source = tiny_2d(HaltonFullContextSparse2DAdapter)
        target = tiny_2d(OneDConditionedHaltonFullContextSparse2DAdapter)
        result = target.load_state_dict(source.state_dict(), strict=False)
        self.assertEqual(set(result.missing_keys), {
            "one_d_context_attn.in_proj_weight",
            "one_d_context_attn.in_proj_bias",
            "one_d_context_attn.out_proj.weight",
            "one_d_context_attn.out_proj.bias",
            "one_d_context_out.weight",
        })
        self.assertEqual(result.unexpected_keys, [])
        z1, z2, index, valid, labels, base = self.batch()
        context = torch.randn(2, 5, 32)
        with torch.no_grad():
            a = source.forward_2d(z1, z2, index, valid, labels, base_features=base)
            b = target.forward_2d(z1, z2, index, valid, labels, base_features=base, one_d_context=context)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(target.one_d_context_out.weight.count_nonzero().item(), 0)

    def test_unified_wrapper_routes_1d_context_and_grads(self):
        torch.manual_seed(11)
        one_d = TinyOneD()
        two_d = tiny_2d(OneDConditionedHaltonFullContextSparse2DAdapter)
        model = UnifiedHaltonMixMaskGIT(one_d, two_d).train()
        z1, z2, index, valid, labels, base = self.batch()
        logits = model.forward_2d(z1, z2, index, valid, labels, base_features=base)
        self.assertEqual(logits.shape, (2, 5, two_d.codebook_size))
        loss = logits[..., :17].mean()
        loss.backward()
        self.assertIsNotNone(model.two_d.one_d_context_out.weight.grad)
        self.assertGreater(model.two_d.one_d_context_out.weight.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
