"""Visual test — screenshots the dashboard and analysis page, checks for issues."""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE_URL = "http://localhost:8000"
OUT_DIR  = ROOT / "data" / "screenshots"
OUT_DIR.mkdir(exist_ok=True)

CHROMIUM = next(
    (p for p in ["/usr/bin/chromium-browser", "/usr/bin/chromium", "/snap/bin/chromium"]
     if Path(p).exists()), None
)


async def run():
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            executable_path=CHROMIUM,
            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-gpu"],
            headless=True,
        )
        page = await browser.new_page(viewport={"width": 1440, "height": 900})

        # ── 1. Dashboard ────────────────────────────────────────────────────
        await page.goto(BASE_URL, wait_until="networkidle")
        await page.screenshot(path=str(OUT_DIR / "01_dashboard.png"), full_page=True)
        print("[1] dashboard.png saved")

        # ── 2. API sanity check ─────────────────────────────────────────────
        jobs_resp = await page.goto(f"{BASE_URL}/api/jobs", wait_until="networkidle")
        jobs = json.loads(await page.evaluate("document.body.innerText"))
        done_jobs = [j for j in jobs if j["status"] == "done"]
        print(f"[2] Jobs: {len(jobs)} total, {len(done_jobs)} done")

        for job in done_jobs[:2]:
            jid = job["job_id"]
            res = await page.goto(f"{BASE_URL}/api/jobs/{jid}/result", wait_until="networkidle")
            data = json.loads(await page.evaluate("document.body.innerText"))
            metrics = data.get("metrics", {})
            ball = metrics.get("ball_metrics")
            print(f"\n    Job {jid[:8]} ({job['filename']}):")
            print(f"      ball_metrics: {ball is not None}")
            if ball:
                print(f"      possession:   {ball.get('possession_pct')}")
                print(f"      passes:       {ball.get('pass_count')}")
                print(f"      accuracy:     {ball.get('pass_accuracy')}")
            else:
                print("      ⚠ ball_metrics is None — stats will be hidden in UI!")

        # ── 3. SPA analysis view ────────────────────────────────────────────
        if done_jobs:
            target_job = done_jobs[0]["job_id"]
            await page.goto(BASE_URL, wait_until="networkidle")
            await page.wait_for_timeout(1000)

            # Click the first done job row
            rows = page.locator("[style*='cursor']")
            count = await rows.count()
            print(f"\n[3] Clickable rows found: {count}")
            if count > 0:
                await rows.first.click()
                await page.wait_for_timeout(3000)
                await page.screenshot(path=str(OUT_DIR / "02_analysis.png"), full_page=True)
                print("    analysis.png saved")

                body = await page.locator("body").inner_text()
                print(f"\n    Possession visible: {'possession' in body.lower()}")
                print(f"    Passes visible:     {'passes' in body.lower()}")
                print(f"    Pass acc visible:   {'pass acc' in body.lower()}")

                # Check for linesman/referee artefacts in overlay would require
                # frame-by-frame analysis — flag for manual review
                print(f"\n    Check overlay video manually at:")
                print(f"      {BASE_URL}/api/jobs/{target_job}/overlay")

        await browser.close()
        print(f"\nAll screenshots in: {OUT_DIR}")


if __name__ == "__main__":
    asyncio.run(run())
