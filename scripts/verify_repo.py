#!/usr/bin/env python
"""Repository verification for the computer-vision portfolio.

Checks lint, tests, the notebook standard, README sections, file sizes, licence/
attribution hygiene, absolute-path leaks and relative links, then prints either
``ALL CHECKS PASSED`` or the list of failures (exit code 1).

Usage::

    python scripts/verify_repo.py                       # everything (make verify)
    python scripts/verify_repo.py --static              # skip pytest (CI: make verify-static)
    python scripts/verify_repo.py --project 03_object_detection [--include PATH ...]
        # one project + global checks; untracked files outside the project are ignored
        # (--skip-shared-docs: do not yet require the root README / license / stack entries)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_NOTEBOOK_BYTES = 5 * 1024 * 1024
MAX_ASSET_BYTES = 1024 * 1024

# Search patterns are assembled at runtime so that this file never matches itself.
_ATTRIB_WORDS = ["cl" + "aude", "anthro" + "pic", "ai-" + "generated"]
ATTRIBUTION_RE = re.compile("|".join(_ATTRIB_WORDS), re.IGNORECASE)
COAUTHOR_RE = re.compile("co-" + "authored-by", re.IGNORECASE)
HOME_PATH_RE = re.compile("/" + "home/[A-Za-z0-9_.-]+/")
PRIVATE_NAMES = {"CL" + "AUDE.md", "." + "cl" + "aude"}

WEIGHT_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".engine", ".plan", ".bin",
    ".tflite", ".h5", ".npz", ".pkl", ".rrd", ".mp4", ".avi", ".mov", ".mkv",
}  # fmt: skip
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".ico", ".zip", ".gz",
                   ".tgz", ".tar", ".ply", ".bz2", ".xz", ".whl"} | WEIGHT_SUFFIXES  # fmt: skip
ASSET_SUFFIXES = {".png", ".gif", ".jpg", ".jpeg", ".svg", ".webp"}

REQUIRED_FILES = [
    "README.md", "LICENSE", "THIRD_PARTY_LICENSES.md", "pyproject.toml", "environment.yml",
    "Makefile", ".gitignore", ".gitattributes", ".pre-commit-config.yaml",
    ".github/workflows/ci.yml", "requirements/constraints.txt", "requirements/base.txt",
    "docs/stack.md", "src/cvlab/__init__.py", "scripts/check_env.py", "scripts/verify_repo.py",
]  # fmt: skip
GITIGNORE_REQUIRED = ["data/", "weights/", "runs/", "outputs/", "third_party/"]
MAKE_TARGETS = ["setup", "check-env", "lint", "test", "verify", "verify-static"]

PROJECT_README_SECTIONS = {
    "What you'll learn": r"what you(?:'|’)?ll learn|what you will learn",
    "Theory": r"theory",
    "How to run": r"how to run",
    "Results": r"results",
    "Limitations": r"limitations",
    "Licenses": r"licen[cs]e",
    "References": r"references",
}
ROOT_README_SECTIONS = {
    "Projects": r"projects",
    "Tested on": r"tested on",
    "Installation": r"install",
    "Quickstart": r"quick ?start",
    "Repository structure": r"(?:repo|repository) (?:structure|layout)",
    "Datasets": r"datasets?",
    "License": r"licen[cs]e",
    "Citation": r"citation|cite|citing",
}
PROJECT_DIR_RE = re.compile(r"^\d\d_[a-z0-9_]+$")


# ------------------------------------------------------------------------------ helpers
def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)


def git_lines(*args: str) -> list[str]:
    out = run(["git", *args])
    if out.returncode != 0:
        return []
    return [line for line in out.stdout.split("\0" if "-z" in args else "\n") if line]


def is_text(path: Path) -> bool:
    if path.suffix.lower() in BINARY_SUFFIXES:
        return False
    try:
        with open(path, "rb") as f:
            return b"\0" not in f.read(8192)
    except OSError:
        return False


def cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def headings(markdown: str) -> list[str]:
    return [
        line.lstrip("#").strip().lower()
        for line in markdown.splitlines()
        if re.match(r"^#{1,6}\s", line)
    ]


class Scope:
    """Files that the checks look at (tracked + untracked-not-ignored, optionally scoped)."""

    def __init__(
        self, project: str | None, include: list[str], skip_shared_docs: bool = False
    ) -> None:
        self.project = project
        self.skip_shared_docs = skip_shared_docs
        tracked = set(git_lines("ls-files", "-z"))
        untracked = set(git_lines("ls-files", "-z", "-o", "--exclude-standard"))
        if project is None:
            files = tracked | untracked
        else:
            prefixes = [f"projects/{project}/", f"assets/{project}/", *include]
            exact = {f"requirements/{project}.txt", *include}
            keep = {
                f for f in untracked
                if f in exact or any(f.startswith(p) for p in prefixes)
                or (f.startswith("scripts/") and project in f)
            }  # fmt: skip
            files = tracked | keep
        self.files = sorted(f for f in files if (ROOT / f).is_file())
        self.tracked = tracked

    def projects(self) -> list[str]:
        names = {
            f.split("/")[1] for f in self.files if f.startswith("projects/") and f.count("/") >= 2
        }
        names = {n for n in names if PROJECT_DIR_RE.match(n)}
        if self.project is not None:
            names = {self.project}
        return sorted(names)


# ------------------------------------------------------------------------------- checks
def check_required_files(scope: Scope) -> list[str]:
    return [f"missing required file: {f}" for f in REQUIRED_FILES if not (ROOT / f).is_file()]


def check_gitignore(scope: Scope) -> list[str]:
    path = ROOT / ".gitignore"
    if not path.exists():
        return [".gitignore missing"]
    text = path.read_text()
    lines = {line.strip() for line in text.splitlines()}
    errs = [f".gitignore must ignore {p}" for p in GITIGNORE_REQUIRED if p not in lines]
    if ATTRIBUTION_RE.search(text):
        errs.append(".gitignore must not reference private assistant files")
    return errs


def check_forbidden_files(scope: Scope) -> list[str]:
    errs = []
    for f in scope.files:
        parts = Path(f).parts
        if any(p in PRIVATE_NAMES for p in parts):
            errs.append(f"private file must never be committed: {f}")
        name = Path(f).name
        if name == ".env" or name.startswith(".env."):
            errs.append(f"secrets file in repo: {f}")
        if Path(f).suffix.lower() in WEIGHT_SUFFIXES:
            errs.append(f"weights/data/video file not ignored: {f}")
        if parts[0] in {"data", "weights", "runs", "outputs", "third_party"}:
            errs.append(f"file inside an ignored data directory: {f}")
    return errs


def check_sizes(scope: Scope) -> list[str]:
    errs = []
    for f in scope.files:
        size = (ROOT / f).stat().st_size
        if size > MAX_FILE_BYTES:
            errs.append(f"{f} is {size / 2**20:.1f} MiB (> 5 MiB)")
        if f.startswith("assets/") and size > MAX_ASSET_BYTES:
            errs.append(f"asset {f} is {size / 2**20:.2f} MiB (> 1 MiB)")
    return errs


def check_attribution(scope: Scope) -> list[str]:
    errs = []
    for f in scope.files:
        p = ROOT / f
        if not is_text(p):
            continue
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if ATTRIBUTION_RE.search(line) or COAUTHOR_RE.search(line):
                errs.append(f"{f}:{i}: forbidden attribution text")
                break
    grep = run(["git", "grep", "-i", "-I", "-l", "-E", "|".join(_ATTRIB_WORDS)])
    if grep.returncode == 0 and grep.stdout.strip():
        errs += [f"git grep match in tracked file: {f}" for f in grep.stdout.split()]
    log = run(["git", "log", "--format=%B"])
    if log.returncode == 0 and (
        ATTRIBUTION_RE.search(log.stdout) or COAUTHOR_RE.search(log.stdout)
    ):
        errs.append("a commit message contains forbidden attribution text or a co-author trailer")
    return errs


def check_home_paths(scope: Scope) -> list[str]:
    errs = []
    for f in scope.files:
        p = ROOT / f
        if not is_text(p):
            continue
        m = HOME_PATH_RE.search(p.read_text(errors="ignore"))
        if m:
            errs.append(
                f"{f}: absolute home path leaked ({m.group(0)}...) - use repo-relative paths"
            )
    return errs


_LINK_RE = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_HTML_SRC_RE = re.compile(r"""(?:src|href)\s*=\s*["']([^"']+)["']""")


def check_links(scope: Scope) -> list[str]:
    errs = []
    for f in scope.files:
        if not f.endswith(".md"):
            continue
        p = ROOT / f
        text = re.sub(r"```.*?```", "", p.read_text(errors="ignore"), flags=re.S)
        for target in _LINK_RE.findall(text) + _HTML_SRC_RE.findall(text):
            if re.match(r"^(https?:|mailto:|#|data:)", target):
                continue
            rel = target.split("#")[0].split("?")[0]
            if not rel:
                continue
            dest = (p.parent / rel).resolve() if not rel.startswith("/") else ROOT / rel.lstrip("/")
            if not dest.exists():
                errs.append(f"{f}: broken relative link -> {target}")
    return errs


def check_lint(scope: Scope) -> list[str]:
    if scope.project is None:
        out = run([sys.executable, "-m", "ruff", "check", "."])
    else:
        targets = [f for f in scope.files if f.endswith((".py", ".ipynb"))]
        if not targets:
            return []
        out = run([sys.executable, "-m", "ruff", "check", "--force-exclude", *targets])
    if out.returncode != 0:
        lines = (out.stdout + out.stderr).strip().splitlines()
        return ["ruff check failed:"] + lines[-40:]
    return []


def check_tests(scope: Scope) -> list[str]:
    if scope.project is None:
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    else:
        tests = [f for f in scope.files if f.startswith("tests/") and f.endswith(".py")]
        if not tests:
            return []
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests]
    out = run(cmd)
    if out.returncode not in (0, 5):  # 5 = no tests collected
        return ["pytest failed:"] + (out.stdout + out.stderr).strip().splitlines()[-40:]
    return []


def relpath(path: Path) -> Path:
    try:
        return path.resolve().relative_to(ROOT)
    except ValueError:
        return path


def check_notebook(path: Path) -> list[str]:
    """Static checks of the notebook standard for one executed tutorial notebook."""
    rel = relpath(path)
    errs: list[str] = []
    size = path.stat().st_size
    if size > MAX_NOTEBOOK_BYTES:
        errs.append(f"{rel}: {size / 2**20:.1f} MiB > 5 MiB")
    try:
        nb = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"{rel}: invalid JSON ({exc})"]
    cells = nb.get("cells", [])
    if (
        not cells
        or cells[0]["cell_type"] != "markdown"
        or not cell_source(cells[0]).lstrip().startswith("# ")
    ):
        errs.append(f"{rel}: first cell must be a markdown title ('# ...') with an overview")
    results_idx = exercises_idx = None
    for i, c in enumerate(cells):
        if c["cell_type"] == "markdown":
            src = cell_source(c)
            if results_idx is None and re.search(r"^##\s+Results\b", src, re.M):
                results_idx = i
            if exercises_idx is None and re.search(r"^##\s+Exercises\b", src, re.M):
                exercises_idx = i
    if results_idx is None:
        errs.append(f"{rel}: missing a '## Results' markdown section")
    if exercises_idx is None:
        errs.append(f"{rel}: missing a '## Exercises' markdown section")

    counts, n_images, results_images, results_tables, todo_cells = [], 0, 0, 0, 0
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = cell_source(c)
        lines = [ln for ln in src.splitlines() if ln.strip()]
        if not lines:
            errs.append(f"{rel}: cell {i} is an empty code cell")
            continue
        if not lines[0].lstrip().startswith("#"):
            errs.append(f"{rel}: cell {i} must start with a '#' comment block")
        if i == 0 or cells[i - 1]["cell_type"] != "markdown":
            errs.append(f"{rel}: cell {i} must be preceded by an explanatory markdown cell")
        if exercises_idx is not None and i > exercises_idx and "TODO" in src:
            todo_cells += 1
        ec = c.get("execution_count")
        if ec is None:
            errs.append(f"{rel}: cell {i} was not executed")
        else:
            counts.append(ec)
        for o in c.get("outputs", []):
            otype = o.get("output_type")
            if otype == "error":
                errs.append(f"{rel}: cell {i} has an error output ({o.get('ename')})")
            data = o.get("data", {})
            if any(k.startswith("application/vnd.jupyter.widget") for k in data):
                errs.append(f"{rel}: cell {i} has a widget output (does not render on GitHub)")
            if any("plotly" in k for k in data):
                errs.append(f"{rel}: cell {i} has a plotly output (use static matplotlib)")
            html = data.get("text/html", "")
            html = "".join(html) if isinstance(html, list) else html
            if any(k.startswith("video/") for k in data) or "<video" in html:
                errs.append(f"{rel}: cell {i} embeds a video")
            has_img = any(k.startswith("image/") for k in data)
            n_images += has_img
            if results_idx is not None and i > results_idx:
                results_images += has_img
                results_tables += "<table" in html
    if counts and counts != list(range(1, len(counts) + 1)):
        errs.append(
            f"{rel}: execution counts are not 1..N - re-run top-to-bottom in a fresh kernel"
        )
    if n_images == 0:
        errs.append(f"{rel}: no image output (at least one matplotlib figure is required)")
    if results_idx is not None:
        if results_images == 0:
            errs.append(f"{rel}: '## Results' section has no figure output")
        if results_tables == 0:
            errs.append(f"{rel}: '## Results' section has no summary table (pandas DataFrame)")
        tail = " ".join(cell_source(c) for c in cells[results_idx:] if c["cell_type"] == "markdown")
        if "takeaway" not in tail.lower():
            errs.append(f"{rel}: '## Results' needs a 'Takeaways' markdown cell")
    if exercises_idx is not None and not 2 <= todo_cells <= 4:
        errs.append(f"{rel}: '## Exercises' should have 2-4 TODO code cells (found {todo_cells})")
    return errs


def check_solution_notebook(path: Path) -> list[str]:
    rel = relpath(path)
    errs = []
    if path.stat().st_size > MAX_NOTEBOOK_BYTES:
        errs.append(f"{rel}: > 5 MiB")
    nb = json.loads(path.read_text())
    for i, c in enumerate(nb.get("cells", [])):
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                errs.append(f"{rel}: cell {i} has an error output")
    return errs


def check_projects(scope: Scope) -> list[str]:
    errs: list[str] = []
    files = set(scope.files)
    root_readme = (ROOT / "README.md").read_text() if (ROOT / "README.md").exists() else ""
    tpl = ROOT / "THIRD_PARTY_LICENSES.md"
    tpl_text = tpl.read_text() if tpl.exists() else ""
    stack = ROOT / "docs/stack.md"
    stack_text = stack.read_text() if stack.exists() else ""
    for proj in scope.projects():
        base = f"projects/{proj}/"
        pfiles = [f for f in files if f.startswith(base)]
        readme = ROOT / base / "README.md"
        if not readme.exists():
            errs.append(f"{proj}: README.md missing")
        else:
            text = readme.read_text()
            heads = headings(text)
            for name, pat in PROJECT_README_SECTIONS.items():
                if not any(re.search(pat, h) for h in heads):
                    errs.append(f"{proj}: README missing section '{name}'")
            if f"assets/{proj}/" not in text:
                errs.append(f"{proj}: README must embed figures from assets/{proj}/")
        nbs = sorted(
            f for f in pfiles if f.startswith(base + "notebooks/") and f.endswith(".ipynb")
        )
        if not nbs:
            errs.append(f"{proj}: no notebooks in {base}notebooks/")
        for nb in nbs:
            if "/.ipynb_checkpoints/" in nb:
                continue
            errs += check_notebook(ROOT / nb)
        sols = [f for f in pfiles if f.startswith(base + "solutions/")]
        if not sols:
            errs.append(f"{proj}: exercise solutions missing in {base}solutions/")
        for s in sols:
            if s.endswith(".ipynb"):
                errs += check_solution_notebook(ROOT / s)
        if f"requirements/{proj}.txt" not in files:
            errs.append(f"{proj}: requirements/{proj}.txt missing")
        assets = [
            f
            for f in files
            if f.startswith(f"assets/{proj}/") and Path(f).suffix.lower() in ASSET_SUFFIXES
        ]
        if not assets:
            errs.append(f"{proj}: no figures in assets/{proj}/")
        if not scope.skip_shared_docs:
            if f"projects/{proj}" not in root_readme:
                errs.append(f"{proj}: not linked from the root README project table")
            if proj not in tpl_text:
                errs.append(f"{proj}: no entry in THIRD_PARTY_LICENSES.md")
            if proj not in stack_text:
                errs.append(f"{proj}: no entry in docs/stack.md")
    return errs


def check_root_readme(scope: Scope) -> list[str]:
    path = ROOT / "README.md"
    if not path.exists():
        return ["README.md missing"]
    text = path.read_text()
    heads = headings(text)
    errs = [
        f"root README missing section '{name}'"
        for name, pat in ROOT_README_SECTIONS.items()
        if not any(re.search(pat, h) for h in heads)
    ]
    if text.count("img.shields.io") < 3:
        errs.append("root README needs badges (Python, PyTorch, license, CI)")
    if "actions/workflows/ci.yml" not in text:
        errs.append("root README needs a CI status badge")
    low = text.lower()
    if "conda" not in low or "venv" not in low:
        errs.append("root README must document both the conda and the venv install routes")
    return errs


def check_configs(scope: Scope) -> list[str]:
    errs = []
    try:
        import yaml
    except ImportError:
        return ["PyYAML not installed: cannot validate YAML configs"]
    for f in [".github/workflows/ci.yml", ".pre-commit-config.yaml", "environment.yml"]:
        p = ROOT / f
        if not p.exists():
            continue
        try:
            data = yaml.safe_load(p.read_text())
        except yaml.YAMLError as exc:
            errs.append(f"{f}: invalid YAML ({exc})")
            continue
        if f.endswith("ci.yml") and not (isinstance(data, dict) and data.get("jobs")):
            errs.append(f"{f}: no jobs defined")
    try:
        import tomllib

        tomllib.loads((ROOT / "pyproject.toml").read_text())
    except Exception as exc:  # noqa: BLE001
        errs.append(f"pyproject.toml: invalid TOML ({exc})")
    mk = ROOT / "Makefile"
    if mk.exists():
        text = mk.read_text()
        for t in MAKE_TARGETS:
            if not re.search(rf"^{re.escape(t)}:", text, re.M):
                errs.append(f"Makefile missing target '{t}'")
    return errs


# --------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--static", action="store_true", help="skip pytest (used in CI)")
    ap.add_argument("--project", help="restrict project-level checks to projects/<PROJECT>")
    ap.add_argument("--include", nargs="*", default=[], help="extra untracked paths in scope")
    ap.add_argument(
        "--skip-shared-docs",
        action="store_true",
        help="do not require root README / THIRD_PARTY_LICENSES / stack entries (pre-commit use)",
    )
    args = ap.parse_args()
    if args.project and not (ROOT / "projects" / args.project).is_dir():
        print(f"unknown project: {args.project}")
        return 2

    scope = Scope(args.project, args.include, args.skip_shared_docs)
    checks: list[tuple[str, Callable[[Scope], list[str]]]] = [
        ("required files", check_required_files),
        (".gitignore", check_gitignore),
        ("no private/secret/weight files", check_forbidden_files),
        ("file sizes", check_sizes),
        ("attribution grep", check_attribution),
        ("no absolute home paths", check_home_paths),
        ("relative links", check_links),
        ("configs (YAML/TOML/Makefile)", check_configs),
        ("root README", check_root_readme),
        ("projects (notebooks/README/assets/solutions)", check_projects),
        ("ruff", check_lint),
    ]
    if not args.static:
        checks.append(("pytest", check_tests))

    where = f"project {args.project}" if args.project else "whole repository"
    print(
        f"verify_repo: {where} | {len(scope.files)} files | "
        f"projects: {', '.join(scope.projects()) or '-'}"
    )
    failures = 0
    for name, fn in checks:
        errs = fn(scope)
        status = "PASS" if not errs else "FAIL"
        print(f"[{status}] {name}")
        for e in errs[:60]:
            print(f"    - {e}")
        if len(errs) > 60:
            print(f"    ... and {len(errs) - 60} more")
        failures += bool(errs)
    print()
    if failures:
        print(f"FAILED: {failures} check(s) failed")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
