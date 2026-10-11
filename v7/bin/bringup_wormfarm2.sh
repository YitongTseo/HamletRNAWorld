#!/usr/bin/env bash
# Consolidate EVERYTHING (8-flask poetry run + 4 sanity experiments) onto this
# host. Written 2026-10-11 for wormfarm2 after wormfarm1 went dark, but it is
# host-agnostic. Run as `web` (passwordless sudo):
#
#     bash ~/HamletRNAWorld/v7/bin/bringup_wormfarm2.sh
#
# Options (env vars):
#     POETRY_ONLY=1      poetry only: stop + disable the 4 sanity experiments
#     GIT_BRAINS=0       don't commit+push every generation's brains to GitHub
#                        (default 1: WORMLET_GIT_COMMIT=1 drop-in on the poetry units)
#     SKIP_DNS=1         don't repoint the poetry hostnames at this tunnel
#     SKIP_TESTS=1       don't run tests/run_all.py before starting
#
# Idempotent — safe to rerun. Each step prints what it did.
set -uo pipefail

REPO=/home/web/HamletRNAWorld
V7=$REPO/v7
PY=/home/web/.venv/bin/python
ENVFILE=/home/web/.wormlet.env
DOMAIN=wordswordsworms.org
POETRY=(wormlet-poetry-1 wormlet-poetry-2 wormlet-poetry-3 wormlet-poetry-4)
SANITY=(wormlet-words wormlet-nouns wormlet-semantic wormlet-pos-chain)
POETRY_HOSTS=("$DOMAIN" "www.$DOMAIN" "poetry-2.$DOMAIN" "poetry-3.$DOMAIN" "poetry-4.$DOMAIN")
SANITY_HOSTS=("words.$DOMAIN" "nouns.$DOMAIN" "semantic.$DOMAIN" "pos-chain.$DOMAIN")
# Per-process memory ceilings for THIS host. The stock units say 1700M/2G
# (14.8G total) which assumes a bigger box; wormfarm2 has 7.1G + 4G swap. The
# known ~44MB/h tick-driven leak means a process eventually hits its cap, is
# killed by the cgroup, and Restart=always brings it back from its checkpoint.
POETRY_MEM=${POETRY_MEM:-1300M}
SANITY_MEM=${SANITY_MEM:-700M}

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
warn() { printf '\033[33m!! %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- 0. access
step "0. Remote access (sshd + Tailscale SSH)"
sudo systemctl enable --now ssh 2>/dev/null || sudo systemctl enable --now sshd 2>/dev/null || warn "no ssh/sshd unit found"
echo "    sshd: $(systemctl is-active ssh 2>/dev/null || systemctl is-active sshd)"
if command -v tailscale >/dev/null; then
  # Tailscale SSH is a second door that does not depend on sshd. Enable it so a
  # dead sshd can never lock us out again. Idempotent.
  # --accept-risk: when run over Tailscale this reroutes the live session.
  # NOTE: with the tailnet's default "check" ssh policy every new session needs
  # a browser approval (~12h); set the ssh rule's action to "accept" in the
  # Tailscale admin ACL to make it silent, or `sudo tailscale set --ssh=false`.
  sudo tailscale set --ssh --accept-risk=lose-ssh && echo "    tailscale ssh: enabled (works even if sshd dies)"
fi

# ------------------------------------------------------------ 1. diagnostics
step "1. Host state (nothing is changed here)"
echo "    uptime: $(uptime -p)   cores: $(nproc)"
free -h | sed 's/^/    /'
df -h / | tail -1 | sed 's/^/    /'
echo "    failed units: $(systemctl --failed --no-legend | wc -l)"
echo "    previous boot ended: $(sudo journalctl --list-boots --no-pager 2>/dev/null | tail -2 | head -1 | awk '{print $(NF-3), $(NF-2), $(NF-1)}')"

# ------------------------------------------------------------------ 2. code
step "2. Update code"
if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
  echo "    stashing local edits:"; git -C "$REPO" status --short --untracked-files=no | sed 's/^/      /'
  # Kept in the stash, NOT popped: the per-epoch commit does `git pull --rebase`,
  # which refuses to run over unstaged edits. `git stash list` to get them back.
  git -C "$REPO" stash -q && echo "    (kept in 'git stash list')"
fi
git -C "$REPO" pull -q --ff-only origin main && echo "    pulled" || warn "git pull failed — continuing with the checked-out code"
git -C "$REPO" config pull.rebase true
git -C "$REPO" config rebase.autoStash true
git -C "$REPO" log -1 --format='    at %h %ad %s' --date=short

# ------------------------------------------------------------------ 3. venv
step "3. Python venv ($PY)"
[ -x "$PY" ] || { echo "    creating"; python3 -m venv /home/web/.venv; }
"$PY" -c 'import numpy, fastapi, uvicorn, nltk, anthropic; print("    numpy/fastapi/uvicorn/nltk/anthropic: ok")' 2>/dev/null \
  || { echo "    installing deps"; "$PY" -m pip install -q -e "$V7"; }
[ -f "$V7/cache/corpus_nomic512.json" ] && echo "    cache/corpus_nomic512.json: present" || warn "cache/corpus_nomic512.json missing"
if [ ! -d "$V7/.venv/nltk_data" ]; then
  echo "    downloading nltk tagger data to v7/.venv/nltk_data"
  (cd "$V7" && "$PY" -c "
import nltk
for p in ('averaged_perceptron_tagger','averaged_perceptron_tagger_eng','universal_tagset'):
    nltk.download(p, download_dir='.venv/nltk_data', quiet=True)") || warn "nltk download failed"
fi
if [ "${SKIP_TESTS:-0}" != 1 ] && [ -f "$V7/tests/run_all.py" ]; then
  echo "    running tests/run_all.py …"
  (cd "$V7" && timeout 600 "$PY" tests/run_all.py 2>&1 | tail -2 | sed 's/^/      /') || warn "tests did not pass cleanly — read the output above before trusting this deploy"
fi

# --------------------------------------------------------------- 4. secrets
step "4. Claude API key ($ENVFILE)"
if grep -qs '^ANTHROPIC_API_KEY=sk-' "$ENVFILE"; then
  echo "    ANTHROPIC_API_KEY: present"
else
  warn "ANTHROPIC_API_KEY not set — the poetry judge + gardener will silently skip."
  warn "fix:  echo 'ANTHROPIC_API_KEY=sk-ant-...' >> $ENVFILE"
fi
chmod 600 "$ENVFILE" 2>/dev/null || true
# .wormlet.env must be secrets-only: EnvironmentFile= shadows every Environment=.
grep -E '^WORMLET_(N_|DATA|PASSAGE|GEN|FLASK)' "$ENVFILE" 2>/dev/null && warn "operational WORMLET_* vars in $ENVFILE override the unit files — remove them"

# ----------------------------------------------------------------- 5. units
step "5. Install systemd units + drop-ins"
UNITS=("${POETRY[@]}")
[ "${POETRY_ONLY:-0}" = 1 ] || UNITS+=("${SANITY[@]}")
NCPU=$(nproc)
TMPD=$(mktemp -d)
for u in "${UNITS[@]}"; do
  src="$V7/deploy/$u.service"
  # Poetry units pin one process per core (CPUAffinity=0..3). Fewer cores:
  # wrap around; one core: drop the pin.
  if [ "$NCPU" -ge 4 ]; then cp "$src" "$TMPD/$u.service"
  elif [ "$NCPU" -gt 1 ]; then
    awk -v n="$NCPU" '/^CPUAffinity=/{sub(/[0-9]+$/, substr($0, match($0,/[0-9]+$/)) % n)} {print}' "$src" > "$TMPD/$u.service"
  else grep -v '^CPUAffinity=' "$src" > "$TMPD/$u.service"; fi
done
sudo cp "$TMPD"/*.service /etc/systemd/system/
rm -rf "$TMPD"
# Tracked drop-ins (board publish, sigma scheme, lifelike on 1-3, cost, ui).
# Flattened as deploy/dropins/wormlet-poetry-N.<name>.conf → nested path.
n_drop=0
for f in "$V7"/deploy/dropins/wormlet-poetry-*.conf; do
  [ -e "$f" ] || break
  b=$(basename "$f"); n=${b#wormlet-poetry-}; n=${n%%.*}
  d=/etc/systemd/system/wormlet-poetry-$n.service.d
  sudo mkdir -p "$d" && sudo cp "$f" "$d/${b#wormlet-poetry-$n.}" && n_drop=$((n_drop+1))
done
# Host-specific memory ceilings (see POETRY_MEM / SANITY_MEM above).
for u in "${POETRY[@]}"; do
  sudo mkdir -p "/etc/systemd/system/$u.service.d"
  printf '# %s: host memory ceiling (stock unit assumes a bigger box)\n[Service]\nMemoryMax=%s\n' "$(hostname)" "$POETRY_MEM" | sudo tee "/etc/systemd/system/$u.service.d/memory.conf" >/dev/null
done
if [ "${POETRY_ONLY:-0}" != 1 ]; then
  for u in "${SANITY[@]}"; do
    sudo mkdir -p "/etc/systemd/system/$u.service.d"
    printf '# %s: host memory ceiling\n[Service]\nMemoryMax=%s\n' "$(hostname)" "$SANITY_MEM" | sudo tee "/etc/systemd/system/$u.service.d/memory.conf" >/dev/null
  done
fi
# Archive brains in git: one commit per epoch per process, serialized by the
# repo lock, then the local purge keeps the disk tidy. `zz-` so it sorts after
# board.conf (drop-ins apply alphabetically; last wins).
for u in "${POETRY[@]}"; do
  if [ "${GIT_BRAINS:-1}" = 1 ]; then
    printf '# push every generation'"'"'s brains to GitHub, then purge locally\n[Service]\nEnvironment=WORMLET_GIT_COMMIT=1\n' | sudo tee "/etc/systemd/system/$u.service.d/zz-git.conf" >/dev/null
  else
    sudo rm -f "/etc/systemd/system/$u.service.d/zz-git.conf"
  fi
done
if [ "${POETRY_ONLY:-0}" = 1 ]; then
  sudo systemctl disable -q --now "${SANITY[@]}" 2>/dev/null
  echo "    sanity experiments: stopped + disabled (data kept under data/experiments/)"
fi
sudo mkdir -p /home/web/board_publish && sudo chown web:web /home/web/board_publish
sudo cp "$V7/deploy/wormlet-healthcheck.service" "$V7/deploy/wormlet-healthcheck.timer" /etc/systemd/system/
sudo install -m 0755 "$V7/deploy/wormlet-healthcheck" /usr/local/bin/wormlet-healthcheck
sudo systemctl daemon-reload
echo "    units: ${UNITS[*]}"
echo "    drop-ins: $n_drop tracked + memory.conf (poetry $POETRY_MEM, sanity $SANITY_MEM) + git brains=${GIT_BRAINS:-1}; cores=$NCPU"

# ------------------------------------------------------------ 6. cloudflare
step "6. Cloudflare tunnel: serve ALL hostnames from this box"
if ! command -v cloudflared >/dev/null; then
  warn "cloudflared not installed — run $V7/bin/setup_cloudflared.sh first, then rerun"
else
  UUID=$(sudo awk '/^tunnel:/{print $2}' /etc/cloudflared/config.yml 2>/dev/null)
  TNAME=$(cloudflared tunnel list -o json 2>/dev/null | "$PY" -c "import json,sys; print(next((t['name'] for t in json.load(sys.stdin) if t['id']=='$UUID'),''))" 2>/dev/null)
  if [ -z "$UUID" ] || [ -z "$TNAME" ]; then
    warn "could not identify this host's tunnel (uuid='$UUID' name='$TNAME') — run $V7/bin/setup_cloudflared.sh"
  else
    echo "    tunnel: $TNAME ($UUID)"
    CFG=$(mktemp); sed "s|<TUNNEL_UUID>|$UUID|g" "$V7/deploy/cloudflared.yml.template" > "$CFG"
    sudo install -m 0644 "$CFG" /etc/cloudflared/config.yml; rm -f "$CFG"
    echo "    wrote /etc/cloudflared/config.yml (poetry + sanity ingress)"
    if [ "${SKIP_DNS:-0}" != 1 ]; then
      # The poetry hostnames CNAME to wormfarm1's dead tunnel; --overwrite-dns
      # repoints them here. Needs ~/.cloudflared/cert.pem (account cert).
      for h in "${POETRY_HOSTS[@]}"; do
        printf '    %s: ' "$h"; cloudflared tunnel route dns --overwrite-dns "$TNAME" "$h" 2>&1 | tail -1
      done
    fi
    sudo systemctl enable -q --now cloudflared; sudo systemctl restart cloudflared
    echo "    cloudflared: $(systemctl is-active cloudflared)"
  fi
fi

# --------------------------------------------------------------- 7. start
step "7. Start the worms"
sudo systemctl enable -q --now "${UNITS[@]}" wormlet-healthcheck.timer
sudo systemctl restart "${UNITS[@]}"      # pick up new code + drop-ins
sleep 25
for u in "${UNITS[@]}"; do
  port=$(grep -oE -- '--port [0-9]+' "/etc/systemd/system/$u.service" | awk '{print $2}')
  if curl -sf --max-time 10 -o /dev/null "http://127.0.0.1:$port/healthz"; then
    echo "    $u :$port  ok   ($(systemctl show -p MemoryCurrent --value "$u" | awk '{printf "%.0fM", $1/1048576}'))"
  else
    echo "    $u :$port  NOT YET ($(systemctl is-active "$u")) — cold start can be slow; journalctl -u $u -n 30"
  fi
done

step "Done. Public check (DNS may take a minute):"
for h in "${POETRY_HOSTS[@]}" "${SANITY_HOSTS[@]}"; do echo "    https://$h/healthz"; done
echo
echo "Note: wormfarm1's live poetry lineages (data/poetry-*/) were on its disk only."
echo "Poetry flasks here cold-start, except whatever generations/ state exists in"
echo "data/poetry-2 from the old git-pushed record."
