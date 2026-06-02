#!/usr/bin/env bash
# Interactive one-time setup for the cloudflared tunnel that serves the
# 4 wormlet sanity experiments. Run this as the `web` user with sudo
# already configured for passwordless commands. The browser-auth step
# requires you to click through Cloudflare's login flow; everything else
# is automatic.
#
# After this script finishes you can:
#   sudo systemctl enable --now cloudflared
#   sudo systemctl enable --now wormlet-words wormlet-nouns wormlet-adj-noun wormlet-pos-chain
#
# Idempotent: rerunning it skips steps that already succeeded.

set -euo pipefail

TUNNEL_NAME=${TUNNEL_NAME:-wormlet}
DOMAIN=${DOMAIN:-wordswordsworms.org}
HOSTS=(words nouns adj-noun pos-chain)
CF_DIR=/etc/cloudflared
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)

echo "=== Cloudflared setup for tunnel '$TUNNEL_NAME' (domain: $DOMAIN) ==="

# Step 1: ensure cloudflared is installed.
if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared not found. Installing from cloudflare's apt-mirrored .deb…"
  sudo curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o /tmp/cloudflared.deb
  sudo dpkg -i /tmp/cloudflared.deb
fi
echo "cloudflared: $(cloudflared --version)"

# Step 2: browser auth (one-time).
if [ ! -f "$HOME/.cloudflared/cert.pem" ]; then
  echo
  echo ">>> Open the URL printed below in a browser and approve."
  cloudflared tunnel login
else
  echo "cert.pem already present — skipping login."
fi

# Step 3: create the tunnel (idempotent — skips if it exists).
EXISTING_UUID=$(cloudflared tunnel list 2>/dev/null | awk -v t="$TUNNEL_NAME" '$2==t {print $1}' || true)
if [ -z "$EXISTING_UUID" ]; then
  echo "Creating tunnel '$TUNNEL_NAME'…"
  cloudflared tunnel create "$TUNNEL_NAME"
  TUNNEL_UUID=$(cloudflared tunnel list 2>/dev/null | awk -v t="$TUNNEL_NAME" '$2==t {print $1}')
else
  TUNNEL_UUID=$EXISTING_UUID
  echo "Tunnel '$TUNNEL_NAME' already exists — UUID $TUNNEL_UUID."
fi

if [ -z "$TUNNEL_UUID" ]; then
  echo "Failed to determine tunnel UUID; aborting." >&2
  exit 1
fi
CREDS_SRC="$HOME/.cloudflared/${TUNNEL_UUID}.json"
if [ ! -f "$CREDS_SRC" ]; then
  echo "Expected creds at $CREDS_SRC but not found." >&2
  exit 1
fi

# Step 4: route DNS for each subdomain.
for h in "${HOSTS[@]}"; do
  fqdn="${h}.${DOMAIN}"
  echo "Routing DNS: $fqdn -> $TUNNEL_NAME"
  cloudflared tunnel route dns "$TUNNEL_NAME" "$fqdn" || \
    echo "  (route may already exist — continuing)"
done

# Step 5: drop the rendered config + creds into /etc/cloudflared.
sudo mkdir -p "$CF_DIR"
sudo cp "$CREDS_SRC" "$CF_DIR/${TUNNEL_UUID}.json"
sudo chmod 600 "$CF_DIR/${TUNNEL_UUID}.json"

CFG=$(mktemp)
sed "s|<TUNNEL_UUID>|$TUNNEL_UUID|g" "$REPO_DIR/deploy/cloudflared.yml.template" > "$CFG"
sudo install -m 0644 "$CFG" "$CF_DIR/config.yml"
rm -f "$CFG"
echo "Wrote $CF_DIR/config.yml (tunnel=$TUNNEL_UUID)"

# Step 6: install cloudflared as a system service.
if ! systemctl list-unit-files | grep -q '^cloudflared\.service'; then
  echo "Installing cloudflared as a systemd service…"
  sudo cloudflared service install || true
fi

echo
echo "=== Done. Next steps ==="
echo "  sudo systemctl enable --now cloudflared"
echo "  sudo cp $REPO_DIR/deploy/wormlet-*.service /etc/systemd/system/"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable --now wormlet-words wormlet-nouns wormlet-adj-noun wormlet-pos-chain"
echo
echo "Each experiment will be reachable at:"
for h in "${HOSTS[@]}"; do echo "  https://${h}.${DOMAIN}/"; done
