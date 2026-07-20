"""
Visual QA test for invoice-frontend-ai9u.onrender.com
Uses Playwright sync_api, headless Chromium.
E2E credentials: e2e-test@invoice-test.dev / TestE2E2026! (confirmed in Supabase)
"""

import sys
import time
import urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright

# Windows consoles default to cp1252 which can't encode UI icons (▣ ◈ etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCREENSHOTS_DIR = Path(__file__).parent
BASE_URL = "https://invoice-frontend-ai9u.onrender.com"
API_BASE_URL = "https://invoice-api-mjx5.onrender.com"
EMAIL = "e2e-test@invoice-test.dev"
PASSWORD = "TestE2E2026!"

console_errors = []


def make_path(name):
    return str(SCREENSHOTS_DIR / name)


def log(msg):
    print(msg, flush=True)


def wait_for_streamlit(page, timeout=60):
    """Wait until Streamlit has fully rendered — poll until body bg is dark or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        bg = page.evaluate(
            "() => window.getComputedStyle(document.body).backgroundColor"
        )
        # Dark bg means inject_theme() has run; white means still loading
        if "255, 255, 255" not in bg:
            time.sleep(1)  # one extra tick for widgets to settle
            return
        time.sleep(2)
    # Timed out — page may still be loading, continue anyway


def warm_up_api(timeout=60):
    """Wake the Render free-tier API before the browser test so pages don't get 502s."""
    url = f"{API_BASE_URL}/api/v1/health/ready"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                if r.status == 200:
                    log(f"  API ready ({r.status})")
                    return True
        except Exception as e:
            log(f"  API not ready yet: {e}")
        time.sleep(5)
    log("  [WARN] API did not become ready in time — pages may show 502")
    return False


def wait_for_page_switch(page, label, timeout=15):
    """Wait until the main content area shows this page's header.
    Checks stMain (not the sidebar) so sidebar nav text doesn't cause false positives.
    Falls back to a fixed sleep if the page doesn't appear within timeout.
    """
    # Page-unique header texts (first word of each page title, not in sidebar)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            main_text = page.evaluate("""() => {
                const main = document.querySelector('[data-testid="stMain"]');
                return main ? main.innerText.substring(0, 300) : '';
            }""")
            if label.upper() in main_text.upper():
                time.sleep(0.5)
                return True
        except Exception:
            pass
        time.sleep(1)
    # Fallback: give Render free tier a moment even if detection timed out
    time.sleep(3)
    return False


def wait_for_login_form(page, timeout=30):
    """Wait until the SIGN IN tab button appears — proves Streamlit tabs have rendered."""
    try:
        page.wait_for_selector(
            'button[role="tab"]:has-text("SIGN IN")',
            state="visible",
            timeout=timeout * 1000,
        )
        time.sleep(0.5)
    except Exception as e:
        log(f"  [WARN] Login form wait: {e}")


def run_tests():
    log("\n=== STEP 0: Wake API ===")
    warm_up_api(timeout=90)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()

        page.on("console", lambda msg: console_errors.append(
            {"type": msg.type, "text": msg.text}
        ) if msg.type in ("error", "warning") else None)

        # ── STEP 1: Login page load ─────────────────────────────────────────
        log("\n=== STEP 1: Login page ===")
        page.goto(BASE_URL, timeout=90000)
        wait_for_streamlit(page, timeout=45)

        page.screenshot(path=make_path("01_login_page.png"), full_page=True)
        log(f"  Screenshot: 01_login_page.png")
        log(f"  Title: {page.title()}")

        body_bg = page.evaluate("() => window.getComputedStyle(document.body).backgroundColor")
        log(f"  Body bg: {body_bg}")

        # Check stApp root (the element our !important targets)
        stapp_bg = page.evaluate("""() => {
            const el = document.querySelector('[data-testid=\"stApp\"]');
            if (!el) return 'not found';
            return window.getComputedStyle(el).backgroundColor;
        }""")
        log(f"  stApp bg: {stapp_bg}")

        font = page.evaluate("() => window.getComputedStyle(document.body).fontFamily")
        log(f"  Font: {font}")

        yellow_count = page.evaluate("""() => {
            let n = 0;
            for (const el of document.querySelectorAll('*')) {
                const s = window.getComputedStyle(el);
                if (s.color.includes('245, 245, 0') || s.backgroundColor.includes('245, 245, 0')) n++;
            }
            return n;
        }""")
        log(f"  Yellow (#F5F500) elements: {yellow_count}")

        mpa_nav_visible = page.evaluate("""() => {
            const nav = document.querySelector('[data-testid=\"stSidebarNav\"]');
            if (!nav) return 'not in DOM';
            const s = window.getComputedStyle(nav);
            return s.display === 'none' ? 'hidden' : 'VISIBLE (problem!)';
        }""")
        log(f"  MPA sidebar nav: {mpa_nav_visible}")

        # ── STEP 2: Attempt login ───────────────────────────────────────────
        log("\n=== STEP 2: Login ===")

        # Wait for the primary button (SIGN IN submit) to appear — proves page is loaded
        wait_for_login_form(page, timeout=90)

        # Make sure we're on SIGN IN tab
        try:
            sign_in_tab = page.locator('button[role="tab"]:has-text("SIGN IN")').first
            if sign_in_tab.count() > 0:
                sign_in_tab.click(timeout=5000)
                time.sleep(1)
                log("  Clicked SIGN IN tab")
        except Exception as e:
            log(f"  [WARN] Tab click: {e}")

        # Fill email (Streamlit text_input with label "EMAIL" → aria-label or placeholder)
        email_filled = False
        for sel in [
            'input[placeholder="you@company.com"]',
            'input[aria-label="EMAIL"]',
            'input[type="text"]:not([aria-label*="search" i])',
        ]:
            try:
                el = page.locator(sel).first
                if el.count() > 0:
                    el.fill(EMAIL)
                    email_filled = True
                    log(f"  Email filled via: {sel}")
                    break
            except Exception:
                pass

        # Fill password
        pw_filled = False
        try:
            pw = page.locator('input[type="password"]').first
            if pw.count() > 0:
                pw.fill(PASSWORD)
                pw_filled = True
                log("  Password filled")
        except Exception as e:
            log(f"  [WARN] Password: {e}")

        # Click SIGN IN submit button (not the tab — exclude role="tab")
        login_clicked = False
        for sel in [
            'button:not([role="tab"]):has-text("SIGN IN")',
            '[data-testid="baseButton-primary"]',
            'button[kind="primary"]',
        ]:
            try:
                btn = page.locator(sel).first
                if btn.count() > 0:
                    btn.click(timeout=5000)
                    login_clicked = True
                    log(f"  SIGN IN clicked via: {sel}")
                    break
            except Exception:
                pass

        if not login_clicked:
            log("  [WARN] Could not find SIGN IN button")
            all_btns = page.locator('button').all()
            log(f"  All buttons: {[b.inner_text()[:30] for b in all_btns]}")

        # Wait for Streamlit rerun after auth — dark bg reappears once logged in
        wait_for_streamlit(page, timeout=30)
        page.screenshot(path=make_path("02_after_login.png"), full_page=True)
        log(f"  Screenshot: 02_after_login.png")

        # Detect if logged in: wait up to 15s for a nav button to appear
        logged_in = False
        for _ in range(15):
            for sel in [
                'button:has-text("INVOICES")',
                'button:has-text("▣  INVOICES")',
                '[data-testid="stSidebar"] button:has-text("INVOICES")',
            ]:
                try:
                    el = page.locator(sel).first
                    if el.count() > 0:
                        logged_in = True
                        log(f"  Login SUCCESS — found: {sel}")
                        break
                except Exception:
                    pass
            if logged_in:
                break
            time.sleep(1)

        if not logged_in:
            page_text = page.evaluate("() => document.body.innerText")[:400]
            log(f"  Login FAILED. Page: {page_text}")

        # ── STEP 3: Navigate authenticated pages ───────────────────────────
        nav_pages = [
            ("INVOICES", "▣", "03_invoices.png"),
            ("CHAT",     "◈", "04_chat.png"),
            ("Q&A",      "◎", "05_qa.png"),
            ("EXTRACT",  "◉", "06_extract.png"),
            ("COMPARE",  "⊞", "07_compare.png"),
            ("BATCH",    "◫", "08_batch.png"),
        ]

        if logged_in:
            log("\n=== STEP 3: Navigate pages ===")
            for label, icon, screenshot_name in nav_pages:
                log(f"\n  -- {label} --")
                nav_clicked = False
                for sel in [
                    f'button:has-text("{label}")',
                    f'button:has-text("{icon}")',
                    f'[data-testid="stSidebar"] button:has-text("{label[:5]}")',
                ]:
                    try:
                        el = page.locator(sel).first
                        if el.count() > 0:
                            el.click(timeout=5000)
                            nav_clicked = True
                            log(f"    Nav clicked: {sel}")
                            break
                    except Exception:
                        pass

                if not nav_clicked:
                    log(f"    [WARN] Could not click nav for {label}")

                # Wait for the main content area to show this page's header
                if not wait_for_page_switch(page, label, timeout=15):
                    log(f"    [WARN] Page header for {label} not detected in time")

                page.screenshot(path=make_path(screenshot_name), full_page=True)
                log(f"    Screenshot: {screenshot_name}")

                # Check for errors
                errors = page.evaluate("""() => {
                    const els = document.querySelectorAll('[data-testid="stException"]');
                    return Array.from(els).map(e => e.innerText.substring(0, 150));
                }""")
                if errors:
                    log(f"    ERRORS: {errors}")

                content = page.evaluate("() => document.body.innerText")[:200]
                log(f"    Content: {content}")
        else:
            log("\n=== STEP 3: SKIPPED (not logged in) ===")

        # ── STEP 4: Dark theme deep check (on loaded page) ─────────────────
        log("\n=== STEP 4: Dark theme check ===")
        # Check on current page (should be fully loaded now)
        theme = page.evaluate("""() => {
            const result = {};
            result.stApp = window.getComputedStyle(
                document.querySelector('[data-testid="stApp"]') || document.body
            ).backgroundColor;
            result.body = window.getComputedStyle(document.body).backgroundColor;
            result.font = window.getComputedStyle(document.body).fontFamily;
            result.stSidebar = (() => {
                const el = document.querySelector('[data-testid="stSidebar"]');
                return el ? window.getComputedStyle(el).backgroundColor : 'not found';
            })();
            result.mpaNav = (() => {
                const el = document.querySelector('[data-testid="stSidebarNav"]');
                if (!el) return 'not in DOM';
                return window.getComputedStyle(el).display === 'none' ? 'hidden(CSS)' : 'VISIBLE';
            })();
            result.whiteElements = Array.from(document.querySelectorAll('*')).filter(el => {
                const bg = window.getComputedStyle(el).backgroundColor;
                return bg === 'rgb(255, 255, 255)';
            }).length;
            return result;
        }""")
        log(f"  stApp bg: {theme.get('stApp')}")
        log(f"  Body bg: {theme.get('body')}")
        log(f"  Font: {theme.get('font')}")
        log(f"  Sidebar bg: {theme.get('stSidebar')}")
        log(f"  MPA nav: {theme.get('mpaNav')}")
        log(f"  White elements: {theme.get('whiteElements')}")
        log(f"  JetBrains loaded: {'jetbrains' in (theme.get('font') or '').lower()}")

        # ── STEP 5: Console errors summary ─────────────────────────────────
        app_errors = [e for e in console_errors if e["type"] == "error"]
        feature_warnings = [e for e in console_errors
                            if "Unrecognized feature" in e.get("text", "")]
        other_warnings = [e for e in console_errors
                          if e["type"] == "warning" and
                          "Unrecognized feature" not in e.get("text", "")]
        log(f"\n=== STEP 5: Console errors ===")
        log(f"  App errors: {len(app_errors)}")
        log(f"  Browser API warnings (ignorable): {len(feature_warnings)}")
        log(f"  Other warnings: {len(other_warnings)}")
        for e in app_errors:
            log(f"  [ERROR] {e['text'][:200]}")
        for e in other_warnings:
            log(f"  [WARN] {e['text'][:200]}")

        browser.close()

    log("\n=== DONE ===")
    return {
        "logged_in": logged_in,
        "theme": theme,
        "app_errors": len(app_errors),
    }


if __name__ == "__main__":
    run_tests()
