"""Run configuration. One dataclass; `mode` fixes (alpha, lam) for the named experimental arms."""
from dataclasses import dataclass, asdict, field
from typing import Optional

import yaml

# mode -> (use_cast_processor, alpha, lam)
MODES = {
    "golden":       (False, None, None),   # original Auffusion, stock processors (reference only)
    "cast_vanilla": (True, 1.5, 0.0),
    "cast_sparse":  (True, 1.5, 1.0),
    "cast_pladis":  (True, 1.5, 2.0),
    "cast":         (True, 1.5, None),   # generic CAST arm: alpha 1.5, lambda must be given explicitly
}


@dataclass
class CastConfig:
    checkpoint: str = "auffusion/auffusion"
    prompt: str = "A dog barking while rain falls in the background."
    seed: int = 42
    num_inference_steps: int = 100
    guidance_scale: float = 7.5
    height: int = 256
    width: int = 1024
    dtype: str = "float16"
    autocast: bool = True
    device: str = "cuda"
    mode: str = "cast_vanilla"
    alpha: Optional[float] = None       # resolved from mode unless given
    lam: Optional[float] = None
    apply_to_uncond: bool = True
    collect_stats: bool = True
    deterministic_self_attn: bool = False   # True: force the math SDPA backend (bitwise-repeatable self-attention, ~4x slower U-Net)
    output_dir: str = "outputs"

    def resolved(self) -> "CastConfig":
        use_cast, a, l = MODES[self.mode]
        c = CastConfig(**asdict(self))
        if use_cast:
            c.alpha = a if self.alpha is None else self.alpha
            c.lam = l if self.lam is None else self.lam
            if c.lam is None:
                raise ValueError(f"mode {self.mode!r} requires an explicit `lam`")
        return c

    @property
    def use_cast(self) -> bool:
        return MODES[self.mode][0]

    @staticmethod
    def from_yaml(path: str, **overrides) -> "CastConfig":
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        d.update(overrides)
        return CastConfig(**d)

    def to_dict(self):
        return asdict(self)
