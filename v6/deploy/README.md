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
