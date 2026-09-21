# systemd drop-ins

The base units (`../wormlet-poetry-N.service`) are the stock definition. Every
deviation this host actually runs lives in a drop-in at
`/etc/systemd/system/wormlet-poetry-N.service.d/<name>.conf`, and this
directory is the tracked copy of them.

They were previously untracked, which meant rebuilding a host from the repo
silently produced a *differently configured* fleet — the base units alone give
you no board publishing, `vs_mean` sigma, no lifelike mode and the default UI.

Filenames here are flattened as `wormlet-poetry-<N>.<dropin>.conf`; install
them back under their real nested path:

```bash
for f in dropins/wormlet-poetry-*.conf; do
  b=$(basename "$f"); n=${b#wormlet-poetry-}; n=${n%%.*}
  d=/etc/systemd/system/wormlet-poetry-$n.service.d
  sudo mkdir -p "$d"
  sudo cp "$f" "$d/${b#wormlet-poetry-$n.}"
done
sudo systemctl daemon-reload
```

## What each one does

| drop-in | on | why |
|---|---|---|
| `board.conf` | 1–4 | `WORMLET_GIT_COMMIT=0` + board publishing to `/home/web/board_publish`. **Overrides `WORMLET_GIT_COMMIT=1` in the base unit** — git is code-only now, so the base unit's value is stale. |
| `sigma-scheme.conf` | 1–4 | `vs_elite` on 1–3, `sigma_anneal` on 4 (standing comparison from the Exp-2 σ A/B). |
| `ui.conf` | 1–4 | `WORMLET_UI=classic`. Either UI is still reachable per-request with `?ui=`. |
| `lifelike.conf` | **1–3 only** | Plasticity + hunger + habituation. **poetry-4 is deliberately the stock control arm — do not add this there** without recording why, or the comparison is lost. |

Note `.bak-*` files may sit beside these in `/etc`; systemd only loads `*.conf`,
so they are inert history and are not tracked here.
