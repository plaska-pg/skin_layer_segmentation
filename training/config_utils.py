"""YAML option defaults for the training scripts' argparse sub-commands.

Each trainer reads the YAML next to it (e.g. semseg/semseg.yaml) unless --config points elsewhere.
Top-level YAML sections are named after the sub-commands (train:, predict:, ...); keys are the
option names with dashes as underscores. Precedence: script defaults < YAML < command-line flags.
Paths in the YAMLs are relative to the repo root, so run the scripts from there.
"""

import argparse
from pathlib import Path

import yaml


def load_yaml(path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def yaml_defaults(parser: argparse.ArgumentParser, section: dict, where: str) -> dict:
    """section as parser defaults: lists -> comma strings for single-value options, mappings -> k=v lists."""
    actions = {a.dest: a for a in parser._actions}
    unknown = sorted(set(section) - set(actions))
    if unknown:
        raise SystemExit(f"{where}: unknown options {unknown}")
    out = {}
    for k, v in section.items():
        if isinstance(v, dict):
            v = ",".join(f"{kk}={vv}" for kk, vv in v.items())
        elif isinstance(v, list) and actions[k].nargs is None:
            v = ",".join(str(x) for x in v)
        out[k] = v
    return out


def subcommands(parser: argparse.ArgumentParser) -> dict:
    """{name: subparser} of a parser built with add_subparsers."""
    return next(a.choices for a in parser._actions if isinstance(a, argparse._SubParsersAction))


def apply_config(parser: argparse.ArgumentParser, default=None, argv=None) -> dict:
    """Add --config to each sub-command and set its defaults from the YAML section of the same name."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=default)
    known, _ = pre.parse_known_args(argv)
    subs = subcommands(parser)
    for sp in subs.values():
        sp.add_argument("--config", default=default, help=f"YAML of option defaults (default: {default})")
    if not known.config:
        return {}
    path = Path(known.config)
    if not path.exists():
        raise SystemExit(f"--config {path} not found")
    cfg = load_yaml(path)
    unknown = sorted(set(cfg) - set(subs))
    if unknown:
        raise SystemExit(f"{path}: unknown sections {unknown}; expected {sorted(subs)}")
    for name, sp in subs.items():
        section = cfg.get(name) or {}
        sp.set_defaults(**yaml_defaults(sp, section, f"{path} [{name}]"))
        for a in sp._actions:
            if a.dest in section:
                a.required = False  # a value from the config satisfies a required option
    return cfg
