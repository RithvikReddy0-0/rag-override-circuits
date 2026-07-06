"""Activation patching for context-override behavior (Phase 1).

Conventions (Heimersheim & Nanda 2024, "How to use and interpret activation
patching"):

- CLEAN run  = conflicting-context prompt: the context states the answer that
  contradicts the model's measured prior, and the model should defer to it.
  This is the behavior we want to explain.
- CORRUPTED run = supportive-context prompt: identical token sequence except
  the single critical-number token, which instead agrees with the prior.

- **Denoising** (restoration): run the corrupted prompt, patch in clean
  activations component-by-component, measure recovery of the clean logit
  diff. Finds components *sufficient* to restore the override.
- **Noising**: run the clean prompt, patch in corrupted activations, measure
  degradation. Finds components *necessary* for the override.
  Both directions are reported because they answer different questions.

- Metric: LD = logit(answer_context_token) - logit(answer_prior_token) at the
  final position. Normalized recovery r = (LD_patched - LD_corr) /
  (LD_clean - LD_corr): 0 = corrupted baseline, 1 = full clean recovery.

Because the two prompts differ at exactly ONE token position (enforced at
dataset build time and re-asserted here), activations at positions before the
critical number are identical between runs and patching them is a no-op —
kept as the "ctx_pre" sanity class.

Batched patching: on CPU, many single forward passes dominate wall-clock, so
configs are grouped into batches; each batch element is the same destination
prompt with a different patch applied via per-row hooks. Results are exactly
equal to running each patch separately (hooks touch only their own row).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from pairs import Phase1Pair, build_prompt

CACHE_SUFFIXES = ("hook_resid_pre", "hook_attn_out", "hook_mlp_out", "attn.hook_z")


def _cache_filter(name: str) -> bool:
    return name.endswith(CACHE_SUFFIXES)


@dataclass(frozen=True)
class PairRun:
    """Tokenized, landmark-annotated, baseline-measured pair for one model."""

    pair_id: str
    is_control: bool
    prior_absdiff: float
    tokens_clean: torch.Tensor  # [1, seq] conflicting-context prompt
    tokens_corr: torch.Tensor  # [1, seq] supportive-context prompt
    diff_pos: int  # the critical number position (only differing token)
    q_start: int  # position of the "Question" token
    ans_start: int  # position of the "Answer" token
    final_pos: int  # last position (next-token logits read here)
    id_context: int  # token id of the conflicting-context answer
    id_prior: int  # token id of the measured-prior answer
    ld_clean: float
    ld_corr: float

    @property
    def denominator(self) -> float:
        return self.ld_clean - self.ld_corr


def _find_landmark(str_tokens: list[str], word: str) -> int:
    exact = [i for i, s in enumerate(str_tokens) if s.strip() == word]
    if len(exact) == 1:
        return exact[0]
    contains = [i for i, s in enumerate(str_tokens) if word in s]
    if len(contains) == 1:
        return contains[0]
    raise ValueError(
        f"landmark {word!r} not uniquely found in prompt tokens "
        f"(exact={exact}, contains={contains})"
    )


def _logit_diff(final_logits: torch.Tensor, id_context: int, id_prior: int) -> float:
    return float(final_logits[id_context] - final_logits[id_prior])


def prepare_pair(model, pair: Phase1Pair) -> PairRun:
    """Tokenize both prompts, verify the one-token-diff structure, locate
    landmarks, and measure the clean/corrupted logit-diff baselines."""
    prompt_clean = build_prompt(pair.context_conflicting, pair.question)
    prompt_corr = build_prompt(pair.context_supportive, pair.question)
    tokens_clean = model.to_tokens(prompt_clean)
    tokens_corr = model.to_tokens(prompt_corr)
    if tokens_clean.shape != tokens_corr.shape:
        raise ValueError(f"{pair.pair_id}: prompts tokenize to different lengths")
    diff_positions = (tokens_clean != tokens_corr).nonzero()
    if len(diff_positions) != 1:
        raise ValueError(
            f"{pair.pair_id}: prompts differ at {len(diff_positions)} positions, need 1"
        )
    diff_pos = int(diff_positions[0, 1])

    str_tokens = model.to_str_tokens(prompt_clean)
    q_start = _find_landmark(str_tokens, "Question")
    ans_start = _find_landmark(str_tokens, "Answer")
    final_pos = tokens_clean.shape[1] - 1
    if not (0 < diff_pos < q_start < ans_start <= final_pos):
        raise ValueError(f"{pair.pair_id}: landmark ordering violated")

    id_context = model.to_single_token(" " + pair.answer_context_token)
    id_prior = model.to_single_token(" " + pair.answer_prior_token)

    with torch.no_grad():
        ld_clean = _logit_diff(model(tokens_clean)[0, -1, :], id_context, id_prior)
        ld_corr = _logit_diff(model(tokens_corr)[0, -1, :], id_context, id_prior)

    return PairRun(
        pair_id=pair.pair_id,
        is_control=pair.is_control,
        prior_absdiff=abs(pair.measured_prior_logit_diff),
        tokens_clean=tokens_clean,
        tokens_corr=tokens_corr,
        diff_pos=diff_pos,
        q_start=q_start,
        ans_start=ans_start,
        final_pos=final_pos,
        id_context=id_context,
        id_prior=id_prior,
        ld_clean=ld_clean,
        ld_corr=ld_corr,
    )


def position_classes(run: PairRun) -> dict[str, tuple[int, ...]]:
    """Semantic position groups for the coarse residual map. ctx_pre is a
    guaranteed no-op (identical activations) kept as a sanity check."""
    return {
        "ctx_pre": tuple(range(0, run.diff_pos)),
        "number": (run.diff_pos,),
        "ctx_post": tuple(range(run.diff_pos + 1, run.q_start)),
        "question": tuple(range(run.q_start, run.ans_start)),
        "answer_end": tuple(range(run.ans_start, run.final_pos + 1)),
    }


@dataclass(frozen=True)
class PatchConfig:
    """One patch: a component (and optionally positions/head) whose
    activations are copied from the source run into the destination run."""

    kind: str  # "resid" | "attn_out" | "mlp_out" | "z"
    layer: int
    positions: tuple[int, ...] | None = None  # None = all positions
    head: int | None = None
    label: str = ""

    @property
    def hook_name(self) -> str:
        if self.kind == "resid":
            return f"blocks.{self.layer}.hook_resid_pre"
        if self.kind == "attn_out":
            return f"blocks.{self.layer}.hook_attn_out"
        if self.kind == "mlp_out":
            return f"blocks.{self.layer}.hook_mlp_out"
        if self.kind == "z":
            return f"blocks.{self.layer}.attn.hook_z"
        raise ValueError(f"unknown kind {self.kind}")


def cache_source_run(model, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    """Forward the source prompt once, caching every hook point we may patch
    from (resid_pre, attn_out, mlp_out, z at all layers)."""
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=_cache_filter)
    return {name: cache[name].detach() for name in cache.keys()}


def _build_hooks(
    chunk: list[PatchConfig], src_cache: dict[str, torch.Tensor]
) -> list[tuple[str, object]]:
    """One hook per distinct hook point; each hook patches only the batch rows
    assigned to it, so a batch of B configs equals B independent patched runs."""
    by_name: dict[str, list[tuple[int, PatchConfig]]] = {}
    for row, cfg in enumerate(chunk):
        by_name.setdefault(cfg.hook_name, []).append((row, cfg))

    hooks = []
    for name, entries in by_name.items():
        src = src_cache[name]  # [1, pos, ...]

        def hook_fn(value, hook, entries=entries, src=src):
            for row, cfg in entries:
                if cfg.head is not None:  # z: [batch, pos, head, d_head]
                    if cfg.positions is None:
                        value[row, :, cfg.head, :] = src[0, :, cfg.head, :]
                    else:
                        pos = list(cfg.positions)
                        value[row, pos, cfg.head, :] = src[0, pos, cfg.head, :]
                elif cfg.positions is None:
                    value[row] = src[0]
                else:
                    pos = list(cfg.positions)
                    value[row, pos] = src[0, pos]
            return value

        hooks.append((name, hook_fn))
    return hooks


def run_patched(
    model,
    dest_tokens: torch.Tensor,
    src_cache: dict[str, torch.Tensor],
    configs: list[PatchConfig],
    id_context: int,
    id_prior: int,
    batch_size: int = 16,
) -> list[float]:
    """Run the destination prompt once per config (batched), each time patching
    in the source activations named by that config. Returns the patched logit
    diff per config, in order.

    Denoising: dest = corrupted tokens, src_cache = clean run's cache.
    Noising:   dest = clean tokens,     src_cache = corrupted run's cache.
    """
    results: list[float] = []
    for start in range(0, len(configs), batch_size):
        chunk = configs[start : start + batch_size]
        batch = dest_tokens.repeat(len(chunk), 1)
        hooks = _build_hooks(chunk, src_cache)
        with torch.no_grad():
            logits = model.run_with_hooks(batch, fwd_hooks=hooks)
        final = logits[:, -1, :]
        for i in range(len(chunk)):
            results.append(_logit_diff(final[i], id_context, id_prior))
    return results


def normalize(ld_patched: float, run: PairRun) -> float:
    """Normalized logit-diff recovery: 0 = corrupted baseline, 1 = clean."""
    return (ld_patched - run.ld_corr) / run.denominator


def head_configs(n_layers: int, n_heads: int, layers: Iterable[int] | None = None) -> list[PatchConfig]:
    layer_list = list(layers) if layers is not None else list(range(n_layers))
    return [
        PatchConfig(kind="z", layer=layer, head=head, label=f"L{layer}H{head}")
        for layer in layer_list
        for head in range(n_heads)
    ]


def attn_mlp_configs(n_layers: int) -> list[PatchConfig]:
    return [
        PatchConfig(kind=kind, layer=layer, label=f"L{layer}.{kind}")
        for layer in range(n_layers)
        for kind in ("attn_out", "mlp_out")
    ]


def resid_class_configs(run: PairRun, n_layers: int) -> list[PatchConfig]:
    return [
        PatchConfig(kind="resid", layer=layer, positions=positions, label=class_name)
        for layer in range(n_layers)
        for class_name, positions in position_classes(run).items()
        if positions  # skip empty classes
    ]


def resid_position_configs(run: PairRun, n_layers: int) -> list[PatchConfig]:
    """Full position resolution, restricted to positions >= diff_pos (earlier
    positions have identical activations by construction)."""
    return [
        PatchConfig(kind="resid", layer=layer, positions=(pos,), label=f"pos{pos}")
        for layer in range(n_layers)
        for pos in range(run.diff_pos, run.final_pos + 1)
    ]
