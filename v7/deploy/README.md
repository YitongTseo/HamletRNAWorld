# Deploy

Files for running the 4 sanity-check experiments on this host.

## Layout

- `wormlet-{words,nouns,adj-noun,pos-chain}.service` — one systemd unit per experiment. Each pins to its own port (8001–8004) and its own data dir (`v6/data/experiments/<mode>/`). `Restart=always` means a crash brings only that one back up; the others keep running.
- `cloudflared.yml.template` — tunnel config with hostname → port mapping. The setup script fills in the tunnel UUID at install time.

## One-time setup (interactive)

```
~/HamletRNAWorld/v6/bin/setup_cloudflared.sh
```

That walks through:
1. install `cloudflared` if not present
2. browser login (one click)
3. create the named tunnel `wormlet`
4. route DNS for `words.`, `nouns.`, `adj-noun.`, `pos-chain.` under wordswordsworms.org
5. install `/etc/cloudflared/config.yml` and the tunnel credentials
6. install cloudflared as a systemd service

You finish with:

```
sudo systemctl enable --now cloudflared
sudo cp ~/HamletRNAWorld/v6/deploy/wormlet-*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now wormlet-words wormlet-nouns wormlet-adj-noun wormlet-pos-chain
```

## Verifying

```
systemctl status wormlet-words wormlet-nouns wormlet-adj-noun wormlet-pos-chain cloudflared
journalctl -u wormlet-words -f         # tail one
curl http://127.0.0.1:8001/healthz     # the words experiment, locally
```

Then load https://words.wordswordsworms.org/ in a browser — the experiment dropdown in the header should let you hop between all five.

## Optional: ANTHROPIC_API_KEY for the gardener

The 4 sanity experiments run a gardener log every 10 generations. If `ANTHROPIC_API_KEY` is set, the gardener writes a short prose log; if not, it silently skips. Put the key in `/home/web/.wormlet.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
```

The systemd units load that file via `EnvironmentFile=`.

---

## v7.1 — per-flask embedder + wormfarm2 quickstart

**What changed (2026-07-08).** Each flask now evolves its **own** embedder
(shared by that flask's worms, fixed per generation) by a (1+1)-ES — there is
**no** cross-process coordinator / shared dir / barrier anymore. The per-tick
chemosensory smell-pass is batched over a precomputed E-table (~19× faster on
that pass; ~11% faster per body-tick overall), which is what lets each poetry
flask run **16 worms** again. Flasks are fully independent, so a fast flask/
process never waits on a slow one.

Env deltas already baked into the units:
- poetry units: `WORMLET_N_WORMS_PER_FLASK=16`, distinct
  `WORMLET_EMBEDDING_SEED` per process (`0/10/20/30`), and the old
  `WORMLET_POETRY_SHARED_DIR` / `_FLASK_OFFSET` / `_ALL_FLASKS` /
  `_BARRIER_TIMEOUT` vars **removed**.
- Each flask's embedder lineage persists at
  `data/<datadir>/generations/<flask>/embedder.json` (cold-starts fresh; brains
  keep their existing `state.json` lineage).

### Bring up a fresh host (e.g. wormfarm2)

```
git clone git@github.com:YitongTseo/HamletRNAWorld.git ~/HamletRNAWorld
cd ~/HamletRNAWorld/v7
# shared venv with numpy + fastapi + uvicorn + nltk (+ project-local nltk_data),
# and cache/corpus_nomic512.json present (tracked in git).
echo 'ANTHROPIC_API_KEY=sk-ant-...' >> ~/.wormlet.env   # only needed for the poetry judge + gardener
chmod 600 ~/.wormlet.env

sudo cp deploy/wormlet-*.service deploy/wormlet-*.timer /etc/systemd/system/
sudo systemctl daemon-reload
# 4 poetry processes (ports 8000/8010/8020/8030, one core each) …
sudo systemctl enable --now wormlet-poetry-1 wormlet-poetry-2 wormlet-poetry-3 wormlet-poetry-4
# … + the 4 sanity experiments + the healthcheck timer
sudo systemctl enable --now wormlet-words wormlet-nouns wormlet-pos-chain wormlet-semantic
sudo systemctl enable --now wormlet-healthcheck.timer
```

Verify:
```
for p in 8000 8010 8020 8030 8001 8002 8003 8004; do curl -s http://127.0.0.1:$p/healthz && echo " :$p ok"; done
journalctl -u wormlet-poetry-1 -f          # watch "flask flask_1 embedder gen=… σ=… incumbent=…" lines
```

Each `CPUAffinity` pins a poetry process to one core; on a box with a different
core count, edit the affinities (and add/remove poetry processes) to match.
If generations ever start freezing under 16 worms, drop
`WORMLET_N_WORMS_PER_FLASK` back to 14 in the poetry units — that alone is the
knob.
