"""Shell completion, printed rather than installed.

Printing the script and letting the user place it is deliberate: writing into
`~/.zshrc` or `/usr/local/share/zsh/site-functions` on someone's behalf is a
change to their shell they did not ask a dataset tool to make, and it is the
kind of thing that survives an uninstall.

The scripts are static. Completing *values* — dataset ids, model options —
would need a backend call on every Tab, which is a request per keystroke against
a server that may not be running, and a completion that hangs is worse than one
that only knows the verbs.
"""

import argparse

from app.cli import output
from app.cli.commands._common import add_common
from app.cli.errors import EXIT_OK

#: Kept in sync with `main.GROUPS` by construction — the script is generated
#: from it rather than being a second hand-written list that can drift.
GLOBAL_FLAGS = (
    "--backend --project --workspace --json --jsonl --no-header --no-color "
    "--quiet --verbose --timeout --wait-backend --help --version"
)

VERBS = {
    "dataset": "ls show ingest prep readiness export rm",
    "train": "run models ls show logs cancel",
    "test": "run datasets ls show per-item compare",
    "infer": "run ls show rm",
    "model": "ls show download rm",
    "project": "ls show create",
    "serve": "",
    "doctor": "",
    "completion": "bash zsh fish",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orinth completion",
        description="Print a shell completion script.",
    )
    parser.add_argument("shell", choices=["bash", "zsh", "fish"])
    add_common(parser)
    return parser


def run(argv: list[str], config, globals_) -> int:
    args = build_parser().parse_args(argv)
    groups = " ".join(VERBS)
    print({"bash": _bash, "zsh": _zsh, "fish": _fish}[args.shell](groups))
    output.note(_HINT[args.shell])
    return EXIT_OK


_HINT = {
    "bash": "Add to ~/.bashrc:  source <(orinth completion bash)",
    "zsh": "Save as _orinth on your $fpath, or add to ~/.zshrc:  source <(orinth completion zsh)",
    "fish": "Save to ~/.config/fish/completions/orinth.fish",
}


def _bash(groups: str) -> str:
    cases = "\n".join(
        f'      {group}) COMPREPLY=($(compgen -W "{verbs}" -- "$cur")); return;;'
        for group, verbs in VERBS.items()
        if verbs
    )
    return f'''_orinth() {{
  local cur prev
  cur="${{COMP_WORDS[COMP_CWORD]}}"
  prev="${{COMP_WORDS[COMP_CWORD-1]}}"

  if [[ "$cur" == -* ]]; then
    COMPREPLY=($(compgen -W "{GLOBAL_FLAGS}" -- "$cur"))
    return
  fi

  if [[ $COMP_CWORD -eq 1 ]]; then
    COMPREPLY=($(compgen -W "{groups}" -- "$cur"))
    return
  fi

  case "$prev" in
{cases}
  esac
}}
complete -F _orinth orinth'''


def _zsh(groups: str) -> str:
    cases = "\n".join(
        f"      {group}) compadd {verbs} ;;" for group, verbs in VERBS.items() if verbs
    )
    return f'''#compdef orinth
_orinth() {{
  if (( CURRENT == 2 )); then
    compadd {groups}
    return
  fi
  case "${{words[2]}}" in
{cases}
  esac
  compadd {GLOBAL_FLAGS}
}}
compdef _orinth orinth'''


def _fish(groups: str) -> str:
    lines = [
        f"complete -c orinth -n '__fish_use_subcommand' -a '{groups}'",
    ]
    for group, verbs in VERBS.items():
        if verbs:
            lines.append(
                f"complete -c orinth -n '__fish_seen_subcommand_from {group}' -a '{verbs}'"
            )
    for flag in GLOBAL_FLAGS.split():
        lines.append(f"complete -c orinth -l {flag.lstrip('-')}")
    return "\n".join(lines)
