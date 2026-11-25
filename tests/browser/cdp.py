"""Tiny Chrome DevTools Protocol driver (localhost only): navigate, wait for text, evaluate JS, click, screenshot."""
import base64, json, os, subprocess, time, urllib.request
from websockets.sync.client import connect

import glob
# headless Chromium from the Playwright cache (no install needed); tests skip when it is missing
_FOUND = sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell")))
CH = _FOUND[-1] if _FOUND else None


class Browser:
    def __init__(self, port=9333, width=1400, height=900):
        self.proc = subprocess.Popen([CH, "--no-sandbox", "--disable-gpu", "--hide-scrollbars", f"--remote-debugging-port={port}", f"--window-size={width},{height}",
                                      f"--user-data-dir=/tmp/claude-1000/chrome-profile-{int(time.time()*1000)}", "--disk-cache-size=1", "--media-cache-size=1", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=2)); break
            except Exception: time.sleep(0.2)
        page = [t for t in tabs if t["type"] == "page"][0]
        self.ws = connect(page["webSocketDebuggerUrl"], max_size=None); self.i = 0; self.logs = []
        for m in ["Page.enable", "Runtime.enable", "Log.enable", "Network.enable"]: self.send(m)
        self.send("Network.setCacheDisabled", cacheDisabled=True)
        self.send("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=False)

    def send(self, method, **params):
        self.i += 1; mid = self.i
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv(timeout=60))
            if msg.get("id") == mid: return msg.get("result", msg.get("error"))
            self._note(msg)

    def _note(self, msg):
        m = msg.get("method")
        if m == "Runtime.consoleAPICalled": self.logs.append((msg["params"]["type"], " ".join(str(a.get("value", a.get("description", ""))) for a in msg["params"]["args"])[:300]))
        elif m == "Runtime.exceptionThrown": self.logs.append(("EXCEPTION", json.dumps(msg["params"]["exceptionDetails"].get("exception", {}).get("description", msg["params"]["exceptionDetails"].get("text")))[:400]))
        elif m == "Log.entryAdded" and msg["params"]["entry"]["level"] in ("error", "warning"): self.logs.append((msg["params"]["entry"]["level"], msg["params"]["entry"]["text"][:300]))

    def eval(self, js):
        r = self.send("Runtime.evaluate", expression=js, returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")

    def goto(self, url, wait_text=None, timeout=60):
        assert url.startswith("http://localhost") or url.startswith("http://127.0.0.1"), "localhost only"
        self.send("Page.navigate", url=url)
        return self.wait(wait_text, timeout) if wait_text else True

    def wait(self, text, timeout=60):
        t = time.time()
        while time.time() - t < timeout:
            try:
                if text in (self.eval("document.body ? document.body.innerText : ''") or ""): return True
            except Exception: pass
            time.sleep(0.4)
            try: self.ws.recv(timeout=0.01)
            except Exception: pass
        return False

    def click_text(self, text, selector="button, [role=tab], a, label, span, div"):
        js = f"""(() => {{ const t = {json.dumps(text)}; const els = [...document.querySelectorAll({json.dumps(selector)})].filter(e => e.children.length < 3 && (e.innerText||'').trim() === t);
                 if (!els.length) return false; els[0].scrollIntoView({{block:'center'}}); els[0].click(); return true; }})()"""
        return self.eval(js)

    def mouse(self, kind, x, y, button="left", buttons=0, **kw):
        """Real input events (Input.dispatchMouseEvent): kind is mouseMoved / mousePressed / mouseReleased / mouseWheel."""
        self.send("Input.dispatchMouseEvent", type=kind, x=x, y=y, button=button, buttons=buttons, clickCount=1 if kind != "mouseMoved" else 0, **kw)

    def wheel(self, x, y, dy):
        self.send("Input.dispatchMouseEvent", type="mouseWheel", x=x, y=y, deltaX=0, deltaY=dy)

    def click(self, x, y):
        self.mouse("mouseMoved", x, y); self.mouse("mousePressed", x, y, buttons=1); self.mouse("mouseReleased", x, y)

    def drag(self, x0, y0, x1, y1, steps=12):
        self.mouse("mouseMoved", x0, y0); self.mouse("mousePressed", x0, y0, buttons=1)
        for k in range(1, steps + 1):
            self.mouse("mouseMoved", x0 + (x1 - x0) * k / steps, y0 + (y1 - y0) * k / steps, buttons=1)
        self.mouse("mouseReleased", x1, y1)

    def screenshot(self, path, full=False, clip=None):
        kw = {"clip": {**clip, "scale": clip.get("scale", 2)}} if clip else {}
        r = self.send("Page.captureScreenshot", format="png", captureBeyondViewport=full, **kw)
        open(path, "wb").write(base64.b64decode(r["data"])); return path

    def close(self):
        try: self.send("Browser.close")
        except Exception: pass
        self.proc.terminate()
