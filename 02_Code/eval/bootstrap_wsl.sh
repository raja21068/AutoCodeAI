#!/usr/bin/env bash
# eval/bootstrap_wsl.sh
# ---------------------
# Prepares a WSL2 Ubuntu distro to run the AgentForge SWE-bench pipeline.
#
# Why Linux at all: the official SWE-bench harness cannot run on native
# Windows. swebench/harness/prepare_images.py imports the POSIX-only
# `resource` module at import time, so `python -m swebench.harness.
# run_evaluation` — the whole of eval/run_official_eval.py — fails before it
# starts. Patch *generation* works fine under Windows; grading does not, and
# ungraded predictions are exactly what the reviewers objected to.
#
# Why a provisioned Python: Ubuntu 26.04 ships only Python 3.14.
# requirements.txt pins numpy<2.0 (chromadb 0.4.24 uses the removed
# np.float_). numpy 1.x publishes no 3.14 wheels and will not build from
# source. Rather than loosen a pin — which would silently change the
# environment the reported numbers come from — uv provisions CPython 3.11.
#
# Why Docker inside the distro: Docker Desktop on this machine has never
# started its Linux VM (no dockerd.log, no backend config in
# settings-store.json); it resets to onboarding and quits, which needs a
# human to accept its licence dialog. A native engine in the distro has no
# GUI dependency. Cost of that choice: its image cache is separate from any
# Docker Desktop cache, so instance images pull fresh.
#
# Usage, from Windows:
#     wsl -d Ubuntu -u root -- bash /mnt/d/AutoResearch/AutoCodeAI/eval/bootstrap_wsl.sh
#
# Idempotent. Exits 3 after enabling systemd, which needs `wsl --shutdown`
# from Windows before a re-run; every other step is safe to repeat.

set -euo pipefail

REPO=/mnt/d/AutoResearch/AutoCodeAI
VENV=/opt/agentforge-venv
PYTHON_VERSION=3.11

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die()  { printf '\n\033[91mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "run as root: wsl -d Ubuntu -u root -- bash $0"

# ---------------------------------------------------------------------------
# 1. Repository
# ---------------------------------------------------------------------------
say "Checking repository at $REPO"
[ -f "$REPO/requirements.txt" ] || die "repo not visible at $REPO"
[ -f "$REPO/.env" ] || die ".env not found at $REPO/.env"
echo "    ok"

# ---------------------------------------------------------------------------
# 2. System packages
# ---------------------------------------------------------------------------
say "Installing system packages"
apt-get update -qq
apt-get install -y -qq git curl ca-certificates docker.io iptables >/dev/null
echo "    $(git --version)"

# ---------------------------------------------------------------------------
# 3. Docker engine
# ---------------------------------------------------------------------------
# WSL2 supports systemd but it is off unless declared. Without a supervisor,
# dockerd dies with the shell that launched it — and a half-dead daemon
# mid-run looks like a wave of unexplained error_instances in the report.
if ! grep -q 'systemd=true' /etc/wsl.conf 2>/dev/null; then
    say "Enabling systemd"
    printf '[boot]\nsystemd=true\n' >> /etc/wsl.conf
    cat <<'EOF'

    systemd enabled. The distro must restart before it takes effect.
    From Windows:
        wsl --shutdown
    then re-run this script.

EOF
    exit 3
fi

say "Starting Docker"
systemctl enable --now docker >/dev/null 2>&1 || true
for _ in $(seq 1 30); do
    docker info >/dev/null 2>&1 && break
    sleep 2
done
docker info >/dev/null 2>&1 || die "docker engine did not come up; check: journalctl -u docker"
echo "    engine: $(docker info --format '{{.ServerVersion}}')"

# SWE-bench instance images are published for x86_64 only.
arch="$(uname -m)"
[ "$arch" = "x86_64" ] || echo "    WARNING: arch is $arch; instance images are x86_64"

avail_gb=$(df -BG --output=avail /var/lib/docker 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$avail_gb" ] && [ "$avail_gb" -lt 60 ]; then
    echo "    WARNING: only ${avail_gb}GB free for images; a Lite run wants 60GB+"
fi

# ---------------------------------------------------------------------------
# 4. Python
# ---------------------------------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    say "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
    export PATH="$HOME/.local/bin:$PATH"
fi

say "Provisioning CPython $PYTHON_VERSION"
uv python install "$PYTHON_VERSION"
PY="$(uv python find "$PYTHON_VERSION")"
echo "    $("$PY" --version)"

# The venv lives outside /mnt/d on purpose: the 9p filesystem is slow enough
# that import time becomes a visible cost across hundreds of instances.
say "Creating venv at $VENV"
[ -d "$VENV" ] || uv venv --python "$PY" "$VENV"

say "Installing requirements"
VIRTUAL_ENV="$VENV" uv pip install -r "$REPO/requirements.txt"

say "Verifying imports"
"$VENV/bin/python" - <<'PY'
import importlib
bad = False
for m in ("swebench", "datasets", "docker", "litellm", "numpy", "unidiff"):
    try:
        mod = importlib.import_module(m)
        print(f"    OK    {m} {getattr(mod, '__version__', '')}")
    except Exception as exc:
        print(f"    FAIL  {m}: {type(exc).__name__}: {exc}")
        bad = True
raise SystemExit(1 if bad else 0)
PY

# ---------------------------------------------------------------------------
# 5. Preflight
# ---------------------------------------------------------------------------
say "Running pipeline preflight (no API spend)"
cd "$REPO"
"$VENV/bin/python" -m eval.smoke_test

cat <<EOF

  Bootstrap complete.

  Future shells:
      wsl -d Ubuntu -u root
      source $VENV/bin/activate && cd $REPO

  Next:
      python -m eval.smoke_test --live            # one instance, end to end
      python -m eval.run_official_eval --run_id smoke

EOF
