"""Patch audiocraft to make xformers an optional import.

Verbatim port of deploy_fish.sh STEP 5b (lines 309-380). audiocraft 1.3.0
hard-imports `from xformers import ops` at module load time. There is no
stable xformers wheel for torch 2.8 + Python 3.12, so we make that import
optional and fall back to torch attention. Idempotent — safe to re-run.

Two files patched:
  * audiocraft/modules/transformer.py — the eager `from xformers import ops`
    import made try/except.
  * audiocraft/models/loaders.py — two call sites that set efficient
    attention backend to xformers, both wrapped in try/except so that the
    backend falls back to torch attention if xformers is unavailable.

Without these patches, `from audiocraft.models import MusicGen` raises
ModuleNotFoundError immediately and the whole pod is unusable.
"""

import pathlib
import sys

# audiocraft installs at /usr/local/lib/python3.12/dist-packages/audiocraft
# under the runpod base. Search the dist-packages dirs Python knows about
# so this works regardless of which interpreter (system vs venv) is used.
audiocraft_dir = None
for p in sys.path:
    candidate = pathlib.Path(p) / "audiocraft"
    if candidate.exists() and candidate.is_dir():
        audiocraft_dir = candidate.parent
        break

if audiocraft_dir is None:
    # Fall back to a glob across common locations.
    import glob
    for g in glob.glob("/usr/local/lib/python*/dist-packages") + glob.glob(
        "/usr/lib/python*/dist-packages"
    ):
        if pathlib.Path(g, "audiocraft").exists():
            audiocraft_dir = pathlib.Path(g)
            break

if audiocraft_dir is None:
    print("FATAL: audiocraft not found in any site-packages on sys.path", file=sys.stderr)
    sys.exit(1)

print(f"[patch] audiocraft dir: {audiocraft_dir}/audiocraft")

# --- Patch 1: transformer.py ---
transformer_py = audiocraft_dir / "audiocraft/modules/transformer.py"
if transformer_py.exists():
    txt = transformer_py.read_text()
    OLD = "from xformers import ops"
    NEW = "try:\n    from xformers import ops\nexcept Exception:\n    ops = None"
    if OLD in txt and NEW not in txt:
        transformer_py.write_text(txt.replace(OLD, NEW, 1))
        print("[patch] transformer.py: xformers import made optional")
    elif NEW in txt:
        print("[patch] transformer.py: already patched")
    else:
        print("[patch] transformer.py: pattern absent — audiocraft version drifted?")
else:
    print(f"[patch] transformer.py NOT FOUND at {transformer_py}", file=sys.stderr)
    sys.exit(1)

# --- Patch 2: loaders.py — two call sites ---
loaders_py = audiocraft_dir / "audiocraft/models/loaders.py"
if loaders_py.exists():
    txt = loaders_py.read_text()
    changed = False

    OLD1 = (
        "    _delete_param(cfg, 'conditioners.args.merge_text_conditions_p')\n"
        "    _delete_param(cfg, 'conditioners.args.drop_desc_p')\n"
        "    model = builders.get_lm_model(cfg)"
    )
    NEW1 = (
        "    _delete_param(cfg, 'conditioners.args.merge_text_conditions_p')\n"
        "    _delete_param(cfg, 'conditioners.args.drop_desc_p')\n"
        "    try:\n        from xformers import ops as _x  # noqa: F401\n"
        "    except Exception:\n"
        "        if hasattr(cfg, 'transformer_lm') and hasattr(cfg.transformer_lm, 'memory_efficient'):\n"
        "            cfg.transformer_lm.memory_efficient = False\n"
        "    model = builders.get_lm_model(cfg)"
    )
    if OLD1 in txt and NEW1 not in txt:
        txt = txt.replace(OLD1, NEW1, 1)
        print("[patch] loaders.py load_lm_model: xformers fallback added")
        changed = True
    elif NEW1 in txt:
        print("[patch] loaders.py load_lm_model: already patched")

    OLD2 = (
        "    if cfg.transformer_lm.memory_efficient:\n"
        "        set_efficient_attention_backend(\"xformers\")"
    )
    NEW2 = (
        "    if cfg.transformer_lm.memory_efficient:\n"
        "        try:\n            from xformers import ops as _xops_check  # noqa: F401\n"
        "            set_efficient_attention_backend(\"xformers\")\n"
        "        except Exception:\n"
        "            cfg.transformer_lm.memory_efficient = False"
    )
    if OLD2 in txt and NEW2 not in txt:
        txt = txt.replace(OLD2, NEW2, 1)
        print("[patch] loaders.py load_lm_model_magnet: xformers fallback added")
        changed = True
    elif NEW2 in txt:
        print("[patch] loaders.py load_lm_model_magnet: already patched")

    if changed:
        loaders_py.write_text(txt)
else:
    print(f"[patch] loaders.py NOT FOUND at {loaders_py}", file=sys.stderr)

print("[patch] audiocraft xformers-optional patches complete")
