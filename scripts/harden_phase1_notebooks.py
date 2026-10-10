"""Apply deterministic, reviewable setup cells to the six original notebooks.

Domain cells remain in their notebooks; setup contains no model-selection logic.
Run this script after editing these templates, then run notebook contract tests.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SETTINGS = '''if "RESEARCH" in globals():
    RESEARCH.close()
_ENVIRONMENT_READY = False
import sys
import platform
from pathlib import Path

if sys.version_info[:2] != (3, 13) or platform.system() != "Linux" or platform.machine() != "x86_64":
    raise RuntimeError("Use a fresh Linux x86_64 / Python 3.13 runtime")

# New research only: publish/review the code first, then pin its full Git SHA.
REPO_COMMIT = ""  # Required 40-character commit, never a moving branch.
RESEARCH_ID = ""  # Required unique name; preserve old runs and frozen evidence.
EXPERIMENT_RUN_ID = ""  # Notebook 05: copy the exact run_id printed by 00 or 04.
DATA_END_UTC = ""  # Required explicit exclusive cutoff, e.g. 2026-08-01T00:00:00+00:00.
REPO_URL = "https://github.com/umutergul74/yeniBot.git"
REPO_DIR = Path("/content/yenibot_repo")
DRIVE_BASE = Path("/content/drive/MyDrive/yeniBot")
REQUIRE_GPU = False  # Set True when running training in a GPU runtime.
ENV_DIR = Path("/content/yenibot_locked_env")
AUTO_UNASSIGN = False  # Release only after successful completion when explicitly enabled.
'''

DRIVE = '''from google.colab import drive
drive.mount('/content/drive')
'''

CHECKOUT = '''import re
import subprocess

if not re.fullmatch(r"[0-9a-f]{40}", REPO_COMMIT):
    raise ValueError("Set REPO_COMMIT to the reviewed full commit before setup")
if any(name == "yenibot" or name.startswith("yenibot.") for name in sys.modules):
    raise RuntimeError("Restart the runtime before reinstalling or changing repository code")
if not (REPO_DIR / ".git").exists():
    subprocess.run(["git", "clone", "--no-checkout", REPO_URL, str(REPO_DIR)], check=True)
else:
    current = subprocess.check_output(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True).strip()
    if current != REPO_COMMIT:
        raise RuntimeError("Existing checkout differs; start a fresh runtime instead of replacing it")
    dirty = subprocess.check_output(["git", "-C", str(REPO_DIR), "status", "--porcelain"], text=True).strip()
    if dirty:
        raise RuntimeError("Checkout has local changes; preserve them before using a fresh runtime")
subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", REPO_COMMIT], check=True)
subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "--detach", REPO_COMMIT], check=True)
print("Pinned commit:", REPO_COMMIT)
'''

INSTALL = '''_ENVIRONMENT_READY = False
if "RESEARCH" in globals():
    RESEARCH.close()
sys.path.insert(0, str(REPO_DIR))
from scripts.colab_runtime import prepare, ResearchKernel
ISOLATED_PYTHON = prepare(REPO_DIR, ENV_DIR, gpu=REQUIRE_GPU)
RESEARCH = ResearchKernel(ISOLATED_PYTHON, REPO_DIR)
RESEARCH.execute("from scripts.verify_environment import verify; verify(installed=True)")
RESEARCH.execute("from pathlib import Path; import sys; assert Path(sys.prefix) == Path(" + repr(str(ENV_DIR)) + "), 'Wrong research interpreter'")
RESEARCH.execute("%matplotlib inline")
_ENVIRONMENT_READY = True
print("Research cells now execute in:", ISOLATED_PYTHON)
'''

REMOTE_CONFIG = '''import os
from datetime import datetime, timezone
from yenibot.config import load_config
from yenibot.notebook_runtime import initialize_workspace, publish_table, verified_table, resolve_data_settings

cfg = load_config(REPO_DIR / "config.yaml")
DATA_END_UTC, RESEARCH_ID = resolve_data_settings(DATA_END_UTC, RESEARCH_ID, REPO_COMMIT, cfg)
cfg['binance']['end_date'] = DATA_END_UTC
WORKSPACE = initialize_workspace(DRIVE_BASE / "research", RESEARCH_ID, REPO_DIR, REPO_COMMIT, cfg)
DATA_DIR = str(WORKSPACE / "data")
CHECKPT_DIR = str(WORKSPACE / "checkpoints")
REPORT_DIR = str(WORKSPACE / "reports")
cfg['paths']['data_dir'] = DATA_DIR
cfg['paths']['checkpoint_dir'] = CHECKPT_DIR
print("Research workspace:", WORKSPACE)
print("Policy status:", cfg.get('experiments', {}).get('policy_review', {}).get('status'))
print("Resolved settings (copy to notebooks 02 onward):")
print("REPO_COMMIT =", repr(REPO_COMMIT))
print("RESEARCH_ID =", repr(RESEARCH_ID))
print("DATA_END_UTC =", repr(DATA_END_UTC))
'''


CONFIG = '''if not globals().get("_ENVIRONMENT_READY", False):
    raise RuntimeError("Complete environment setup successfully before research")
context = {
    "REPO_COMMIT": REPO_COMMIT, "RESEARCH_ID": RESEARCH_ID,
    "EXPERIMENT_RUN_ID": EXPERIMENT_RUN_ID, "DATA_END_UTC": DATA_END_UTC,
    "REPO_DIR": str(REPO_DIR), "DRIVE_BASE": str(DRIVE_BASE), "AUTO_UNASSIGN": False,
}
RESEARCH.execute("globals().update(" + repr(context) + ")")
RESEARCH.execute("from pathlib import Path; import re; REPO_DIR = Path(REPO_DIR); DRIVE_BASE = Path(DRIVE_BASE)")
''' + "RESEARCH.execute(" + repr(REMOTE_CONFIG) + ")\n"


def remote_source(source: str) -> str:
    """Recover reviewed domain code from the thin execution wrapper."""
    tree = ast.parse(source)
    if tree.body and isinstance(tree.body[0], ast.Expr):
        value = tree.body[0].value
        if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                and isinstance(value.func.value, ast.Name) and value.func.value.id == "RESEARCH"
                and value.func.attr == "execute"):
            return ast.literal_eval(value.args[0])
    return source


def wrap_remote(source: str) -> str:
    escaped = source.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    return 'RESEARCH.execute("""' + escaped + '""")\n'


def update_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    for index, source in {1: SETTINGS, 2: DRIVE, 3: CHECKOUT, 4: INSTALL, 5: CONFIG}.items():
        if index == 1 and path.name.startswith("01_"):
            source = source.replace('RESEARCH_ID = ""  # Required unique name; preserve old runs and frozen evidence.',
                                    'RESEARCH_ID = "auto"  # New date/config/environment gets a separate workspace.')
            source = source.replace('DATA_END_UTC = ""  # Required explicit exclusive cutoff, e.g. 2026-08-01T00:00:00+00:00.',
                                    'DATA_END_UTC = "latest_complete_day"  # Resolves once to today 00:00 UTC; prints exact downstream settings.')
        cells[index]["source"] = source.splitlines(keepends=True)
    if path.name.startswith("01_"):
        cells[6]["source"] = ['from yenibot.data.preparation import prepare_raw_data\n',
            'prepare_raw_data(cfg, DATA_DIR, archive_cache=DRIVE_BASE / "archive_cache")\n']
    for cell in cells[6:]:
        if cell["cell_type"] != "code":
            continue
        source = remote_source("".join(cell["source"]))
        source = source.replace("AUTO_UNASSIGN = True", "# AUTO_UNASSIGN is set in the settings cell.")
        source = source.replace("pd.read_parquet(", "verified_table(")
        source = source.replace("REPORT_DIR = f'{DRIVE_BASE}/reports'", "REPORT_DIR = str(WORKSPACE / 'reports')")
        # Preserve failures and their logs instead of disconnecting in finally.
        source = source.replace("finally:\n    if AUTO_UNASSIGN:", "except Exception:\n    raise\nelse:\n    if AUTO_UNASSIGN:")
        source = source.replace("finally:\n    runtime.unassign()", "except Exception:\n    raise\nelse:\n    if AUTO_UNASSIGN:\n        runtime.unassign()")
        if source.strip() == "from google.colab import runtime\nruntime.unassign()":
            source = "from google.colab import runtime\nif AUTO_UNASSIGN:\n    runtime.unassign()\n"
        source = source.replace("labeled.groupby('label')['fwd_return_10h']", "labeled.groupby('label')[f\"fwd_return_{cfg['labeling']['max_holding_bars']}h\"]")
        if path.name.startswith("01_"):
            source = source.replace("df.to_parquet(out, index=False)",
                "publish_table(df, Path(out), provenance={'kind': 'normalized_klines', 'symbol': binance['symbol'], 'interval': interval, 'start': binance['start_date'], 'end_exclusive': binance['end_date'], 'source_policy': binance.get('data_source', 'auto')})")
            source = source.replace("metrics.to_parquet(out, index=False)",
                "publish_table(metrics, Path(out), provenance={'kind': 'normalized_futures_metrics', 'end_exclusive': binance['end_date']})")
            source = source.replace("funding.to_parquet(out, index=False)",
                "publish_table(funding, Path(out), provenance={'kind': 'normalized_funding', 'end_exclusive': binance['end_date']})")
            source = source.replace("print('Funding rate download skipped; Binance REST may be restricted from this runtime:', repr(exc))",
                "raise RuntimeError('Configured funding source failed; stop rather than reuse stale data') from exc")
        if path.name.startswith("02_"):
            source = source.replace("intrabar = None", "intrabar_path = metrics_path = funding_path = None\nintrabar = None") if "intrabar_path = metrics_path = funding_path = None" not in source else source
            source = source.replace("print('Intrabar data missing; 15m profiles will run without ih15 features until 01 is rerun:', intrabar_path)", "raise FileNotFoundError(intrabar_path)")
            source = source.replace("print('Futures metrics missing; futures-context profiles will run without fut metrics until 01 is rerun:', metrics_path)", "raise FileNotFoundError(metrics_path)")
            source = source.replace("print('Funding rates missing; funding features will be absent until REST access is available:', funding_path)", "raise FileNotFoundError(funding_path)")
            source = source.replace("features.frame.to_parquet(out, index=False)",
                "parents = [Path(DATA_DIR) / 'raw/btc_1h.parquet', Path(DATA_DIR) / 'raw/btc_4h.parquet']\nfor variable in ('intrabar_path', 'metrics_path', 'funding_path'):\n    if variable in locals():\n        parents.append(Path(locals()[variable]))\npublish_table(features.frame, Path(out), parents=parents, provenance={'stage': 'features'})")
        if path.name.startswith("03_"):
            source = source.replace("quality = validate_label_quality(\n    labeled,\n", "quality = validate_label_quality(\n    labeled,\n    forward_return_column=f\"fwd_return_{label_cfg['max_holding_bars']}h\",\n") if "forward_return_column=" not in source else source
            source = source.replace("labeled.to_parquet(out, index=False)",
                "publish_table(labeled, Path(out), parents=[Path(DATA_DIR) / 'processed/features_1h.parquet'], provenance={'stage': 'labels', 'horizon': label_cfg['max_holding_bars']})")
        if path.name.startswith("05_"):
            source = source.replace("future_oos_preflight, latest_experiment_run, write_experiment_diagnostics", "future_oos_preflight, write_experiment_diagnostics")
            source = source.replace("    run_dir = latest_experiment_run(CHECKPT_DIR)\n    print('Latest experiment run:', run_dir)",
                "    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}', EXPERIMENT_RUN_ID):\n        raise ValueError('Set EXPERIMENT_RUN_ID to the exact run printed by notebook 00 or 04')\n    run_dir = Path(CHECKPT_DIR) / 'experiments' / EXPERIMENT_RUN_ID\n    if not run_dir.is_dir():\n        raise FileNotFoundError(run_dir)\n    print('Selected experiment run:', run_dir)")
            source = source.replace("        output_dir=REPORT_DIR,\n        write_full_bundles=", "        output_dir=REPORT_DIR,\n        run_id=EXPERIMENT_RUN_ID,\n        write_full_bundles=")
        source = source.replace("if variable in locals():", "if locals().get(variable) is not None:")
        source = source.replace("from google.colab import runtime", "# Runtime lifecycle belongs to the Colab host.")
        source = source.replace("runtime.unassign()", "pass  # Host releases only after every cell succeeds.")
        cell["source"] = wrap_remote(source).splitlines(keepends=True)
    last_code = next(cell for cell in reversed(cells[6:]) if cell["cell_type"] == "code")
    last_code["source"].append("RESEARCH.finish(release=AUTO_UNASSIGN)\n")
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:cell{index}", "exec")
            cell["outputs"] = []
            cell["execution_count"] = None
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    for notebook in sorted((ROOT / "notebooks").glob("0[0-5]_*.ipynb")):
        update_notebook(notebook)
