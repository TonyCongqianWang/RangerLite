# RangerLite

RangerLite is a streamlined, bug-fixed derivative of the Ranger21 optimizer. It retains the high-performance core (PNM + Lookahead + NormLoss) while stripping away ~1,000 lines of experimental features and addressing a significant architectural flaw in the original implementation.

### The "Rolling Scope" Bug Fix

RangerLite resolves a critical Python variable scoping leak found in the original Ranger21. In the original code, Weight Decay and NormLoss were applied outside the inner parameter loop, accidentally targeting whichever tensor the namespace left "hanging" from the previous iteration.

* **Original Behavior:** Regularization was applied to a single, effectively random parameter per group, leaving 99.9% of the model unregularized.
* **RangerLite Behavior:** Regularization is mathematically stable and applied uniformly across all parameters.

### Key Changes
* **De-bloated:** Removed MadGrad/AdaBelief cores, internal schedulers, AGC, GC, and softplus smoothing.
* **Deterministic Parity:** Includes a `use_legacy_scoping_bug` flag. When set to `True`, RangerLite produces bit-for-bit identical updates to Ranger21 by intentionally replicating the rolling scope leak.

### Quick Start

You can convince yourself that the optimizer behaves the same way given a restricted initialization with

```
pip install ranger21
python rangerliter_test.py
```

Usage:

```
from ranger_lite import RangerLite

# For historical bit-for-bit compatibility
optimizer = RangerLite(
    model.parameters(),
    lr=1.0,
    use_legacy_scoping_bug=True
)

# For the corrected, stable implementation
optimizer = RangerLite(
    model.parameters(),
    lr=1.0,
    weight_decay=1e-4,
    use_legacy_scoping_bug=False
)
```

### Ranger21Fix

You can also init the submodule and run

```
pip install ranger21
python ranger21fix_test.py
```

to convince yourself that Ranger21Fix contains a fix for the ranger21 leaked variable bug.