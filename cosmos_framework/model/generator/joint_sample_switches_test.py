"""CPU tests for JOINT_ATTN_BLOCK / JOINT_GEOM_LEAD / JOINT_SAMPLER_STEPS_OVERRIDE.

    cd <repo> && PYTHONPATH=. python cosmos_framework/model/generator/joint_sample_switches_test.py
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from cosmos_framework.data.generator.sequence_packing.modality import ModalityData
from cosmos_framework.data.generator.sequence_packing.runtime import from_all_seq
from cosmos_framework.model.generator import joint_sample_switches as sw
from cosmos_framework.model.generator.fk_sequence import FKTokenPayload
from cosmos_framework.model.generator.mot.attention import SplitInfo, build_packed_sequence


def idx(x):
    return torch.tensor(x, dtype=torch.long)


def synthetic_sequence():
    """15 tokens: text 0-1 | vision 2-5 (2 frames x 2 tokens, frame0 = condition) |
    action 6-8 (row 0 = condition) | pointflow 9-11 (anchor 9, noisy 10-11) | fk 12-14 (anchor 12, noisy 13-14)."""
    vision = ModalityData(
        sequence_indexes=idx([2, 3, 4, 5]),
        token_shapes=[(2, 1, 2)],
        noisy_frame_indexes=[idx([1])],
        tokens=[torch.zeros(1)],
    )
    action = ModalityData(
        sequence_indexes=idx([6, 7, 8]), token_shapes=[(3,)], noisy_frame_indexes=[idx([1, 2])], tokens=[torch.zeros(1)]
    )
    return SimpleNamespace(
        sequence_length=15,
        sample_lens=[15],
        split_lens=[2, 13],
        attn_modes=["causal", "full"],
        text_indexes=idx([0, 1]),
        vision=vision,
        action=action,
        point=FKTokenPayload(torch.zeros(3, 1), idx([9, 10, 11]), idx([[10, 11]])),
        fk=FKTokenPayload(torch.zeros(3, 1), idx([12, 13, 14]), idx([[13, 14]])),
    )


class ParsingTests(unittest.TestCase):
    def test_attn_block_parse(self):
        self.assertEqual(sw.parse_joint_attn_block(""), ())
        self.assertEqual(sw.parse_joint_attn_block(None), ())
        self.assertEqual(
            sw.parse_joint_attn_block("action<-pointflow, action<-fk,action<-fk"),
            (("action", "pointflow"), ("action", "fk")),
        )

    def test_attn_block_errors(self):
        for bad in ("action<-pointflo", "foo<-fk", "action", "action<-fk<-vision", "action<-action", "<-fk", "a->b"):
            with self.assertRaises(ValueError, msg=bad):
                sw.parse_joint_attn_block(bad)

    def test_env_helpers_default_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            for name in ("JOINT_ATTN_BLOCK", "JOINT_GEOM_LEAD", "JOINT_SAMPLER_STEPS_OVERRIDE"):
                os.environ.pop(name, None)
            self.assertEqual(sw.joint_attn_block_from_env(), ())
            self.assertEqual(sw.joint_geom_lead_from_env(), 0)
            self.assertIsNone(sw.joint_steps_override_from_env())
        with mock.patch.dict(os.environ, {"JOINT_GEOM_LEAD": "3", "JOINT_SAMPLER_STEPS_OVERRIDE": "12"}):
            self.assertEqual(sw.joint_geom_lead_from_env(), 3)
            self.assertEqual(sw.joint_steps_override_from_env(), 12)

    def test_lead_and_steps_errors(self):
        for bad in ("-1", "x", "1.5"):
            with self.assertRaises(ValueError):
                sw.parse_joint_geom_lead(bad)
        for bad in ("0", "-2", "x"):
            with self.assertRaises(ValueError):
                sw.parse_joint_steps_override(bad)

    def test_scope_default_and_restore(self):
        self.assertEqual(sw.get_active_joint_attn_block(), ())
        with sw.active_joint_attn_block((("action", "fk"),)):
            self.assertEqual(sw.get_active_joint_attn_block(), (("action", "fk"),))
        self.assertEqual(sw.get_active_joint_attn_block(), ())
        with self.assertRaises(RuntimeError):
            with sw.active_joint_attn_block((("action", "fk"),)):
                raise RuntimeError
        self.assertEqual(sw.get_active_joint_attn_block(), ())


class MaskTests(unittest.TestCase):
    def test_layout(self):
        ids, noisy = sw.build_modality_layout(synthetic_sequence())
        self.assertEqual(ids.tolist(), [-1, -1, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        self.assertEqual(
            noisy.tolist(), [False] * 4 + [True, True] + [False, True, True] + [False, True, True] * 2
        )

    def test_empty_rules_is_none(self):
        self.assertIsNone(sw.build_joint_attn_allowed(synthetic_sequence(), ()))
        ids, noisy = sw.build_modality_layout(synthetic_sequence())
        self.assertIsNone(sw.build_blocked_matrix(ids, noisy, ()))

    def test_blocked_pairs(self):
        seq = synthetic_sequence()
        allowed = sw.build_joint_attn_allowed(seq, (("action", "pointflow"), ("action", "fk")))
        self.assertEqual(tuple(allowed.shape), (13, 15))  # gen rows 2..14
        row = lambda token: allowed[token - 2]
        # noisy action rows (7, 8) lose noisy PF (10, 11) and noisy FK (13, 14), nothing else
        for q in (7, 8):
            self.assertEqual([k for k in range(15) if not row(q)[k]], [10, 11, 13, 14])
        # action row 0 (condition/state) sees everything
        self.assertTrue(row(6).all())
        # PF/FK anchors stay visible to noisy action; text visible
        self.assertTrue(row(7)[[0, 1, 9, 12]].all())
        # other query modalities untouched (direction matters: PF/FK rows still see action)
        for q in [2, 3, 4, 5, 9, 10, 11, 12, 13, 14]:
            self.assertTrue(row(q).all(), q)

    def test_each_modality_pair_and_conditions(self):
        seq = synthetic_sequence()
        allowed = sw.build_joint_attn_allowed(seq, (("pointflow", "vision"), ("vision", "action")))
        row = lambda token: allowed[token - 2]
        self.assertEqual([k for k in range(15) if not row(10)[k]], [4, 5])  # first-frame vision tokens 2,3 stay
        self.assertEqual([k for k in range(15) if not row(11)[k]], [4, 5])
        self.assertTrue(row(9).all())  # PF anchor query unaffected
        self.assertEqual([k for k in range(15) if not row(4)[k]], [7, 8])  # noisy vision rows lose noisy action
        self.assertTrue(row(2).all() and row(3).all())  # first-frame (condition) vision rows unaffected

    def test_absent_modality_raises(self):
        seq = synthetic_sequence()
        seq.fk = None
        with self.assertRaises(ValueError):
            sw.build_joint_attn_allowed(seq, (("action", "fk"),))

    def test_multi_sample_unsupported(self):
        seq = synthetic_sequence()
        seq.sample_lens = [7, 8]
        seq.split_lens = [1, 6, 1, 7]
        seq.attn_modes = ["causal", "full"] * 2
        with self.assertRaises(NotImplementedError):
            sw.build_joint_attn_allowed(seq, (("action", "fk"),))


def _pack(n, text_len, heads=4, kv_heads=2, dim=8):
    hidden = torch.randn(n, 16)
    pack, meta, _ = build_packed_sequence(
        "two_way",
        packed_sequence=hidden,
        attn_modes=["causal", "full"],
        split_lens=[text_len, n - text_len],
        sample_lens=[n],
        packed_und_token_indexes=torch.arange(text_len),
        packed_gen_token_indexes=torch.arange(text_len, n),
        num_heads=heads,
        head_dim=dim,
        num_layers=1,
    )
    return pack


class MaskedAttentionTests(unittest.TestCase):
    def reference(self, q, k, v, kn, allowed, text_len):
        groups = q.shape[1] // k.shape[1]
        k, v, kn = (x.repeat_interleave(groups, dim=1) for x in (k, v, kn))
        out = torch.zeros_like(q)
        scale = q.shape[-1] ** -0.5
        n = q.shape[0]
        for h in range(q.shape[1]):
            for i in range(n):
                if i < text_len:
                    keys, kk, ok = torch.arange(i + 1), k, None
                else:
                    keys, kk = torch.arange(n), kn
                    keys = keys[allowed[i - text_len]]
                w = ((q[i, h] @ kk[keys, h].T) * scale).softmax(-1)
                out[i, h] = w @ v[keys, h]
        return out

    def test_matches_naive_reference(self):
        torch.manual_seed(0)
        seq = synthetic_sequence()
        allowed = sw.build_joint_attn_allowed(seq, (("action", "pointflow"), ("action", "fk"), ("vision", "action")))
        pack = _pack(15, 2)
        q, k, v, kn = torch.randn(15, 4, 8), torch.randn(15, 2, 8), torch.randn(15, 2, 8), torch.randn(15, 2, 8)
        with torch.no_grad():
            out = sw.masked_two_way_attention(
                from_all_seq(q, pack), from_all_seq(k, pack), from_all_seq(v, pack), allowed, 2,
                normalized_k=from_all_seq(kn, pack),
            )
            from cosmos_framework.data.generator.sequence_packing.runtime import get_all_seq

            got = get_all_seq(out).view(15, 4, 8)
        torch.testing.assert_close(got, self.reference(q, k, v, kn, allowed, 2), atol=1e-5, rtol=1e-5)

    def test_all_allowed_equals_unmasked(self):
        torch.manual_seed(1)
        pack = _pack(15, 2)
        q, k, v = torch.randn(15, 4, 8), torch.randn(15, 2, 8), torch.randn(15, 2, 8)
        from cosmos_framework.data.generator.sequence_packing.runtime import get_all_seq

        with torch.no_grad():
            a = sw.masked_two_way_attention(
                from_all_seq(q, pack), from_all_seq(k, pack), from_all_seq(v, pack), torch.ones(13, 15, dtype=torch.bool), 2
            )
            # gen rows with an all-True mask == plain softmax over all keys
            kk, vv = k.repeat_interleave(2, 1), v.repeat_interleave(2, 1)
            full = torch.nn.functional.scaled_dot_product_attention(
                q[2:].transpose(0, 1), kk.transpose(0, 1), vv.transpose(0, 1)
            ).transpose(0, 1)
        torch.testing.assert_close(get_all_seq(a).view(15, 4, 8)[2:], full, atol=1e-5, rtol=1e-5)

    def test_grad_enabled_raises(self):
        pack = _pack(15, 2)
        x = torch.randn(15, 4, 8)
        with self.assertRaises(RuntimeError):
            sw.masked_two_way_attention(
                from_all_seq(x, pack), from_all_seq(x[:, :2], pack), from_all_seq(x[:, :2], pack),
                torch.ones(13, 15, dtype=torch.bool), 2,
            )

    def test_split_info_default_is_unmasked(self):
        meta = SplitInfo([2, 13], ["causal", "full"], [15], 15)
        self.assertIsNone(meta.joint_attn_allowed)


class GeomLeadTests(unittest.TestCase):
    def test_schedule_indices(self):
        self.assertEqual(sw.geom_lead_schedule(5, 0), [(i, i, i + 1) for i in range(5)])
        # geometry grid has N-k = 3 steps; reaches its last point (sigma 0) at outer step 3 and stays
        self.assertEqual(sw.geom_lead_schedule(5, 2), [(0, 0, 1), (1, 1, 2), (2, 2, 3), (3, 3, 3), (4, 3, 3)])
        for n, k in ((3, 3), (3, 10), (3, -1), (0, 0)):
            with self.assertRaises(ValueError):
                sw.geom_lead_schedule(n, k)

    def test_sigma_grid_matches_sampler(self):
        from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler

        sampler = UniPCSampler(tensor_kwargs={"device": "cpu"})
        sigmas, timesteps = sw.unipc_sigma_grid(sampler, 5, 5.0)
        self.assertEqual(len(sigmas), 6)
        self.assertEqual(float(sigmas[-1]), 0.0)
        self.assertEqual(timesteps.tolist(), [999, 952, 882, 768, 555])
        g_sigmas, g_timesteps = sw.unipc_sigma_grid(sampler, 3, 5.0)
        self.assertEqual(len(g_sigmas), 4)
        self.assertEqual(int(g_timesteps[0]), 999)  # geometry starts at the same max sigma
        self.assertEqual(float(g_sigmas[-1]), 0.0)

    def _grids(self, n, k):
        from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler

        sampler = UniPCSampler(tensor_kwargs={"device": "cpu"})
        return sw.unipc_sigma_grid(sampler, n, 5.0), sw.unipc_sigma_grid(sampler, n - k, 5.0)

    def test_geometry_timestep_sequence(self):
        n, k = 6, 2
        (sig, ts), (gsig, gts) = self._grids(n, k)
        calls = []

        def fn(state, t, gt):
            calls.append((int(t), int(gt)))
            return [torch.zeros_like(x) for x in state]

        sw.euler_geom_lead_sample(fn, [torch.zeros(5)], [2], sig, ts, k, gsig, gts)
        geom_t = [c[1] for c in calls]
        self.assertEqual(geom_t[0], int(ts[0]))  # starts at the max, same as vision
        self.assertEqual(geom_t[: n - k], gts.tolist())
        self.assertEqual(geom_t[n - k :], [0] * k)  # sigma 0 from outer step N-k on
        self.assertEqual([c[0] for c in calls], ts.tolist())  # vision/action: normal grid

    def test_k0_is_plain_euler(self):
        n = 5
        (sig, ts), _ = self._grids(n, 0)
        calls = []

        def fn(state, t, gt):
            calls.append((int(t), int(gt)))
            return [torch.ones_like(x) for x in state]

        out = sw.euler_geom_lead_sample(fn, [torch.zeros(5)], [2], sig, ts, 0)
        self.assertTrue(all(a == b for a, b in calls))
        self.assertEqual([a for a, _ in calls], ts.tolist())
        torch.testing.assert_close(out[0], -float(sig[0]) * torch.ones(5))  # sum dt = 0 - sigma[0]

    def test_toy_linear_field_exact_x0_for_geometry(self):
        # x_sigma = sigma*eps + (1-sigma)*x0 -> v = eps - x0 is constant along the path: Euler is exact.
        torch.manual_seed(0)
        n, k = 8, 3
        (sig, ts), (gsig, gts) = self._grids(n, k)
        x0, eps = torch.randn(7), torch.randn(7)
        glen = 4
        state0 = [eps.clone()]
        sig0 = float(sig[0])  # ~0.9998: initial state is x_sigma0, built consistently
        state0 = [sig0 * eps + (1 - sig0) * x0]
        # the toy field is defined on the same (eps, x0) path for head and geometry alike
        v_const = eps - x0
        fn = lambda state, t, gt: [v_const.clone()]
        out = sw.euler_geom_lead_sample(fn, state0, [glen], sig, ts, k, gsig, gts)[0]
        # geometry: N-k Euler steps from gsig[0] to 0, exact => x0 (given start on the path at gsig[0] == sig[0])
        torch.testing.assert_close(out[-glen:], x0[-glen:], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out[:-glen], x0[:-glen], atol=1e-5, rtol=1e-5)
        # and geometry is fixed after it reaches 0: more outer steps change nothing
        (sig2, ts2), (gsig2, gts2) = self._grids(n + 4, k + 4)
        out2 = sw.euler_geom_lead_sample(fn, state0, [glen], sig2, ts2, k + 4, gsig2, gts2)[0]
        torch.testing.assert_close(out2[-glen:], out[-glen:], atol=1e-5, rtol=1e-5)

    def test_lead_requires_geometry_grid(self):
        (sig, ts), _ = self._grids(5, 2)
        with self.assertRaises(ValueError):
            sw.euler_geom_lead_sample(lambda s, t, g: s, [torch.zeros(3)], [1], sig, ts, 2)
        with self.assertRaises(ValueError):
            sw.euler_geom_lead_sample(lambda s, t, g: s, [torch.zeros(3)], [1], sig, ts, 5)



def two_sample_sequence():
    """synthetic_sequence() twice: tokens 0-14 (sample 0) and 15-29 (sample 1)."""
    a = synthetic_sequence()
    shift = lambda x: torch.cat([x, x + 15])
    vision = ModalityData(
        sequence_indexes=shift(a.vision.sequence_indexes),
        token_shapes=a.vision.token_shapes * 2,
        noisy_frame_indexes=a.vision.noisy_frame_indexes * 2,
        tokens=a.vision.tokens * 2,
    )
    action = ModalityData(
        sequence_indexes=shift(a.action.sequence_indexes),
        token_shapes=a.action.token_shapes * 2,
        noisy_frame_indexes=a.action.noisy_frame_indexes * 2,
        tokens=a.action.tokens * 2,
    )
    payload = lambda p: FKTokenPayload(torch.zeros(6, 1), shift(p.sequence_indexes), torch.cat([p.noisy_indexes, p.noisy_indexes + 15]))
    return SimpleNamespace(
        sequence_length=30,
        sample_lens=[15, 15],
        split_lens=[2, 13, 2, 13],
        attn_modes=["causal", "full"] * 2,
        text_indexes=idx([0, 1, 15, 16]),
        vision=vision,
        action=action,
        point=payload(a.point),
        fk=payload(a.fk),
        control_weights=None,
    )


def _varlen_reference(q, k, v, is_causal=False, causal_type=None, cumulative_seqlen_Q=None,
                      cumulative_seqlen_KV=None, max_seqlen_Q=None, max_seqlen_KV=None):
    """CPU stand-in for the flash varlen kernel: SDPA per segment, BSHD in/out, GQA by repetition."""
    q, k, v = q[0], k[0], v[0]
    groups = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
    cu_q = [0, q.shape[0]] if cumulative_seqlen_Q is None else cumulative_seqlen_Q.tolist()
    cu_k = [0, k.shape[0]] if cumulative_seqlen_KV is None else cumulative_seqlen_KV.tolist()
    outs = []
    for (a, b), (c, d) in zip(zip(cu_q[:-1], cu_q[1:]), zip(cu_k[:-1], cu_k[1:])):
        if b == a:
            continue
        o = torch.nn.functional.scaled_dot_product_attention(
            q[a:b].transpose(0, 1), k[c:d].transpose(0, 1), v[c:d].transpose(0, 1), is_causal=is_causal
        )
        outs.append(o.transpose(0, 1))
    return torch.cat(outs)[None]


def _dense_masked_reference(q, k, v, kn, ids, rules, sample_lens, text_lens):
    """Per sample: text rows causal over text keys; generation rows over all sample keys (normalised K),
    minus every key of a blocked modality (all tokens, text never blocked)."""
    groups = q.shape[1] // k.shape[1]
    k, v, kn = (x.repeat_interleave(groups, dim=1) for x in (k, v, kn))
    outs, start = [], 0
    for n, t in zip(sample_lens, text_lens):
        s = slice(start, start + n)
        sid = ids[s]
        allowed = torch.ones(n - t, n, dtype=torch.bool, device=sid.device)
        for query, key in rules:
            qm, km = sw._MODALITY_ID[query], sw._MODALITY_ID[key]
            allowed &= ~((sid[t:] == qm)[:, None] & (sid == km)[None, :])
        sdpa = lambda qq, kk, vv, **kw: torch.nn.functional.scaled_dot_product_attention(
            qq.transpose(0, 1), kk.transpose(0, 1), vv.transpose(0, 1), **kw
        ).transpose(0, 1)
        qs, ks, vs, kns = q[s], k[s], v[s], kn[s]
        outs.append(sdpa(qs[:t], ks[:t], vs[:t], is_causal=True))
        outs.append(sdpa(qs[t:], kns, vs, attn_mask=allowed[None]))
        start += n
    return torch.cat(outs)


class FixedMaskTests(unittest.TestCase):
    def test_presets_resolve_and_keep_wam_path(self):
        self.assertEqual(sw.resolve_joint_attn_mask(""), ())
        self.assertEqual(sw.resolve_joint_attn_mask(None), ())
        self.assertEqual(
            sw.resolve_joint_attn_mask("action_no_geom"), (("action", "pointflow"), ("action", "fk"))
        )
        self.assertEqual(sw.resolve_joint_attn_mask("action<-fk;vision<-fk"), (("action", "fk"), ("vision", "fk")))
        for name, spec in sw.JOINT_ATTN_MASK_PRESETS.items():
            rules = sw.resolve_joint_attn_mask(name)
            self.assertTrue(rules, name)
            self.assertNotIn(("vision", "action"), rules, name)
            self.assertNotIn(("action", "vision"), rules, name)
        self.assertEqual(len(sw.resolve_joint_attn_mask("geom_readonly")), 4)
        self.assertEqual(len(sw.resolve_joint_attn_mask("physical_split")), 6)
        with self.assertRaises(ValueError):
            sw.resolve_joint_attn_mask("geom_readonl")

    def test_modality_ids_cover_conditioning_tokens(self):
        self.assertEqual(
            sw.build_modality_ids(synthetic_sequence()).tolist(),
            [-1, -1, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        )

    def test_gather_plan(self):
        seq = two_sample_sequence()
        self.assertIsNone(sw.build_modality_gather(seq, ()))
        plan = sw.build_modality_gather(seq, sw.resolve_joint_attn_mask("geom_readonly"))
        # every generation token appears exactly once
        self.assertEqual(sorted(plan.q_index.tolist()), list(range(26)))
        self.assertEqual(plan.cu_seqlens_q.tolist()[-1], 26)
        self.assertEqual(plan.cu_seqlens_kv.tolist()[-1], plan.kv_index.numel())
        # sample 0 segments: vision, action (lose PF 9-11 and FK 12-14), pointflow, fk (see everything)
        cu_k = plan.cu_seqlens_kv.tolist()
        segments = [plan.kv_index[a:b].tolist() for a, b in zip(cu_k[:-1], cu_k[1:])]
        self.assertEqual(segments[0], list(range(9)))  # vision: text + vision + action
        self.assertEqual(segments[1], list(range(9)))  # action
        self.assertEqual(segments[2], list(range(15)))  # pointflow reads all
        self.assertEqual(segments[5], list(range(15, 24)))  # sample 1 action, offset by 15
        self.assertEqual(plan.max_seqlen_q, 4)
        self.assertEqual(plan.max_seqlen_kv, 15)

    def test_gather_errors(self):
        seq = two_sample_sequence()
        seq.fk = None
        with self.assertRaises(ValueError):
            sw.build_modality_gather(seq, (("action", "fk"),))
        seq = two_sample_sequence()
        seq.attn_modes = ["causal", "full", "full", "full"]
        with self.assertRaises(NotImplementedError):
            sw.build_modality_gather(seq, (("action", "fk"),))

    def _pack2(self, heads=4, kv_heads=2, dim=8):
        pack, meta, _ = build_packed_sequence(
            "two_way",
            packed_sequence=torch.randn(30, 16),
            attn_modes=["causal", "full"] * 2,
            split_lens=[2, 13, 2, 13],
            sample_lens=[15, 15],
            packed_und_token_indexes=idx([0, 1, 15, 16]),
            packed_gen_token_indexes=torch.cat([torch.arange(2, 15), torch.arange(17, 30)]),
            num_heads=heads,
            head_dim=dim,
            num_layers=1,
        )
        return pack

    def _run(self, preset, grad):
        from cosmos_framework.data.generator.sequence_packing.runtime import get_all_seq
        from cosmos_framework.model.generator.mot import attention as attn_mod

        torch.manual_seed(0)
        seq = two_sample_sequence()
        rules = sw.resolve_joint_attn_mask(preset)
        plan = sw.build_modality_gather(seq, rules)
        pack = self._pack2()
        q, k, v, kn = (torch.randn(30, h, 8, requires_grad=grad) for h in (4, 2, 2, 2))
        with mock.patch.object(attn_mod, "attention", _varlen_reference):
            out = attn_mod.two_way_attention(
                from_all_seq(q, pack), from_all_seq(k, pack), from_all_seq(v, pack),
                packed_key_states_normalized=from_all_seq(kn, pack), modality_gather=plan,
            )
        got = get_all_seq(out).view(30, 4, 8)
        ref = _dense_masked_reference(q, k, v, kn, sw.build_modality_ids(seq), rules, [15, 15], [2, 2])
        return got, ref, (q, k, v, kn)

    def test_gathered_attention_matches_dense_mask(self):
        for preset in sw.JOINT_ATTN_MASK_PRESETS:
            got, ref, _ = self._run(preset, grad=False)
            torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5, msg=preset)

    def test_gathered_attention_gradients(self):
        got, ref, leaves = self._run("physical_split", grad=True)
        w = torch.randn_like(got)
        g_got = torch.autograd.grad((got * w).sum(), leaves)
        g_ref = torch.autograd.grad((ref * w).sum(), leaves)
        for a, b in zip(g_got, g_ref):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)

    def test_mask_changes_output(self):
        got, _, inputs = self._run("action_no_geom", grad=False)
        plain = _dense_masked_reference(*inputs, sw.build_modality_ids(two_sample_sequence()), (), [15, 15], [2, 2])
        # action rows differ from the unmasked attention, video rows do not
        self.assertFalse(torch.allclose(got[6:9], plain[6:9]))
        torch.testing.assert_close(got[2:6], plain[2:6])


if __name__ == "__main__":
    unittest.main()
