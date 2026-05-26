#!/usr/bin/env python3
"""Fetch C. elegans neuron soma positions from OpenWorm c302 NeuroML2 files.
Run from the v4/ directory: python scripts/fetch_positions.py
"""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import urllib.request

WEIGHTS = Path("sim/weights.json")
OUT = Path("sim/neuron_positions.json")
BASE = "https://raw.githubusercontent.com/openworm/c302/master/c302/NeuroML2/{}.cell.nml"
PATTERN = re.compile(r'name="Seg0_soma_0"[^>]*>.*?<proximal x="([^"]+)" y="([^"]+)" z="([^"]+)"', re.DOTALL)

def fetch(name):
    try:
        with urllib.request.urlopen(BASE.format(name), timeout=10) as r:
            txt = r.read().decode()
        m = PATTERN.search(txt)
        if m:
            return name, (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    except Exception as e:
        print(f"  miss {name}: {e}", file=sys.stderr)
    return name, None

def main():
    # All neurons: pre-synaptic keys + their post-synaptic targets.
    with open(WEIGHTS) as f:
        weights = json.load(f)
    all_names = set(weights.keys())
    for targets in weights.values():
        all_names.update(targets.keys())
    all_names = sorted(all_names)

    positions = {}
    with ThreadPoolExecutor(max_workers=20) as pool:
        futs = {pool.submit(fetch, n): n for n in all_names}
        done = 0
        for fut in as_completed(futs):
            name, pos = fut.result()
            done += 1
            if pos:
                positions[name] = pos
            if done % 50 == 0:
                print(f"  {done}/{len(all_names)} done, {len(positions)} found")

    print(f"Fetched {len(positions)}/{len(all_names)} neuron positions.")
    OUT.write_text(json.dumps(positions, indent=2))
    print(f"Written to {OUT}")

if __name__ == "__main__":
    main()
