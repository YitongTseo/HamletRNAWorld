"""Visual smoke test. Pauses the world, parks the worm head at a known spot,
sets up a sweeping facing direction, and screenshots from a couple of
orientations so we can iterate on the GFP look."""
from __future__ import annotations

import asyncio
import json
import math

from playwright.async_api import async_playwright


URL = "http://127.0.0.1:8765"
OUT = "/tmp"


async def cmd(page, *messages, settle_ms: int = 250):
    """Open a fresh WS, fire the messages, close."""
    payload = json.dumps(messages)
    await page.evaluate(
        """async (msgs) => {
            const w = new WebSocket(`ws://${location.host}/ws`);
            await new Promise((r) => w.addEventListener('open', r, { once: true }));
            for (const m of msgs) w.send(JSON.stringify(m));
            await new Promise((r) => setTimeout(r, 200));
            w.close();
        }""",
        list(messages),
    )
    await page.wait_for_timeout(settle_ms)


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await ctx.new_page()
        logs: list[str] = []
        page.on("console", lambda m: logs.append(f"[{m.type}] {m.text}"))
        page.on("pageerror", lambda e: logs.append(f"[ERROR] {e}"))

        await page.goto(URL, wait_until="networkidle")
        await page.wait_for_timeout(500)

        # First pause + park so we know where to start.
        await cmd(
            page,
            {"type": "set_paused", "paused": True},
            {"type": "set_head", "x": 800, "y": 500, "facing": 0.0},
            {"type": "clear_food"},
        )
        await page.wait_for_timeout(400)
        await page.screenshot(path=f"{OUT}/v4_a_paused.png")
        head_a = await page.evaluate("() => document.getElementById('hud').textContent")

        # Unpause and grab a series of screenshots ~1s apart so we can see
        # the body actually wiggling.
        await cmd(page, {"type": "set_paused", "paused": False}, settle_ms=200)
        snaps = []
        for i in range(5):
            await page.wait_for_timeout(900)
            await page.screenshot(path=f"{OUT}/v4_t{i+1}.png")
            hud = await page.evaluate("() => document.getElementById('hud').textContent")
            snaps.append(hud)

        print("HUD progression:")
        print(f"  paused:  {head_a}")
        for i, hud in enumerate(snaps):
            print(f"  t+{i+1}s:    {hud}")

        info = await page.evaluate("""() => {
            const s = window.__sim;
            return {
                bloomStrength: s.bloom.strength,
                bloomThreshold: s.bloom.threshold,
                bloomRadius: s.bloom.radius,
                wormExists: !!s.wormMesh,
                rendererCalls: s.renderer.info.render.calls,
                rendererTris: s.renderer.info.render.triangles,
            };
        }""")
        print("info:", json.dumps(info, indent=2))
        if logs:
            print("\nlogs (tail):")
            for ln in logs[-10:]:
                print(" ", ln)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
