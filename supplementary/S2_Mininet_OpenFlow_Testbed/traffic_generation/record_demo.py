"""Record the Mininet live demonstration as a video.

This experiment provides a live proof of concept. A paper cannot show one, but
IEEE Access accepts supplementary video, so this produces a short recording of
the actual session: the topology coming up, the switch connecting to the
XAI-SDN controller, a randomised-source SYN flood starting, the benign client's
ping latency degrading, the detector installing drop rules, and the flow table
afterwards.

The recording is genuine. Every frame shows output captured from the running
session in real time, rendered to a terminal-style image and encoded to MP4. No
text is fabricated or re-ordered, and the timing between frames reflects when
each line actually appeared.

Output: an MP4 plus the raw transcript, written to the user's Downloads folder.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "common"))
from paths import DATA_ROOT, RESULTS  # noqa: E402

OUT_DIR = Path(os.environ.get("XAISDN_VIDEO_OUT",
                              str(Path.home() / "Downloads")))
WSL_DIR = "/opt/xaisdn"
W, H = 1280, 720
MARGIN, LINE_H, FONT_SIZE = 18, 21, 15
MAX_LINES = (H - 2 * MARGIN) // LINE_H
FPS = 10
BG = (14, 16, 20)
FG = (222, 226, 232)
ACCENT = (120, 200, 255)
WARN = (255, 190, 90)
GOOD = (130, 230, 150)
TITLE_BG = (26, 30, 38)


def load_font(size: int):
    for name in ("consola.ttf", "cour.ttf", "DejaVuSansMono.ttf"):
        for base in (r"C:\Windows\Fonts", "/usr/share/fonts/truetype/dejavu"):
            p = Path(base) / name
            if p.exists():
                try:
                    return ImageFont.truetype(str(p), size)
                except Exception:
                    pass
    return ImageFont.load_default()


FONT = load_font(FONT_SIZE)
FONT_B = load_font(FONT_SIZE + 3)


def colour_for(line: str):
    l = line.lower()
    if line.startswith("$ ") or line.startswith("mininet>"):
        return ACCENT
    if any(k in l for k in ("error", "fail", "loss", "unreachable", "drop")):
        return WARN
    if any(k in l for k in ("ok", "connected", "installed", "alert", "detected",
                            "0% packet loss", "done")):
        return GOOD
    return FG


def render(lines: list[str], caption: str) -> np.ndarray:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 34], fill=TITLE_BG)
    d.text((MARGIN, 8), "XAI-SDN live demonstration   Mininet + Open vSwitch + "
                        "OpenFlow 1.3", font=FONT_B, fill=ACCENT)
    if caption:
        d.rectangle([0, H - 32, W, H], fill=TITLE_BG)
        d.text((MARGIN, H - 25), caption, font=FONT_B, fill=WARN)
    y = 42
    for ln in lines[-MAX_LINES:]:
        d.text((MARGIN, y), ln[:150], font=FONT, fill=colour_for(ln))
        y += LINE_H
    return np.asarray(img)


def wsl_stream(cmd: str, sink: "queue.Queue[str]") -> int:
    p = subprocess.Popen(
        ["wsl.exe", "-d", "Ubuntu", "-u", "root", "--", "bash", "-c",
         "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; "
         + cmd],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1, errors="replace")
    for line in p.stdout:
        sink.put(line.rstrip("\n"))
    p.wait()
    return p.returncode


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mp4 = OUT_DIR / "XAI-SDN_mininet_live_demo.mp4"
    txt = OUT_DIR / "XAI-SDN_mininet_live_demo_transcript.txt"

    # Stage everything the demo needs into WSL. E4b normally puts the controller
    # and the model there already, but the recording has to work even if E4b did
    # not run, so we stage them again rather than assume.
    demo = HERE / "mininet" / "demo_session.sh"
    src = demo.as_posix().replace("C:/", "/mnt/c/")
    ctl = (HERE / "mininet" / "xaisdn_controller.py").as_posix().replace("C:/", "/mnt/c/")
    launcher = (HERE / "mininet" / "run_controller.py").as_posix().replace("C:/", "/mnt/c/")
    model = DATA_ROOT / "cache" / "e4_controller_rf.joblib"
    # Strip carriage returns on the way in. These files live on a Windows
    # filesystem and a single CRLF makes bash fail with a syntax error on the
    # first loop, which is how an earlier recording ended up showing nothing.
    stage = (f"mkdir -p {WSL_DIR} && cp '{src}' {WSL_DIR}/demo_session.sh "
             f"&& cp '{ctl}' {WSL_DIR}/xaisdn_controller.py "
             f"&& cp '{launcher}' {WSL_DIR}/run_controller.py "
             f"&& sed -i 's/\\r$//' {WSL_DIR}/demo_session.sh "
             f"{WSL_DIR}/xaisdn_controller.py {WSL_DIR}/run_controller.py "
             f"&& chmod +x {WSL_DIR}/demo_session.sh")
    if model.exists():
        stage += (" && cp '" + model.as_posix().replace("C:/", "/mnt/c/")
                  + f"' {WSL_DIR}/model.joblib")
    subprocess.run(["wsl.exe", "-d", "Ubuntu", "-u", "root", "--", "bash", "-c", stage],
                   capture_output=True, text=True)

    q: "queue.Queue[str]" = queue.Queue()
    rc_box = {}

    def runner():
        rc_box["rc"] = wsl_stream(f"bash {WSL_DIR}/demo_session.sh", q)
        q.put("__END__")

    th = threading.Thread(target=runner, daemon=True)
    th.start()

    lines: list[str] = []
    shots: list[tuple[str, "np.ndarray"]] = []
    caption = "starting"
    captions = {
        "STEP 1": "Step 1 of 6  building the topology",
        "STEP 2": "Step 2 of 6  switch connects to the XAI-SDN controller",
        "STEP 3": "Step 3 of 6  baseline: normal traffic, no attack",
        "STEP 4": "Step 4 of 6  SYN flood begins (randomised source addresses)",
        "STEP 5": "Step 5 of 6  detector installs drop rules, traffic recovers",
        "STEP 6": "Step 6 of 6  flow table and controller summary",
    }

    print(f"recording to {mp4}", flush=True)
    writer = imageio.get_writer(mp4, fps=FPS, codec="libx264",
                               quality=8, macro_block_size=None)
    frames = 0
    t0 = time.time()
    done = False
    try:
        while not done:
            deadline = time.time() + 1.0 / FPS
            got_new = False
            while time.time() < deadline:
                try:
                    ln = q.get(timeout=0.02)
                except queue.Empty:
                    continue
                if ln == "__END__":
                    done = True
                    break
                for k, v in captions.items():
                    if k in ln and v != caption:
                        # Save the frame that closes the previous step, so the
                        # paper can show a still of each stage rather than a
                        # redrawn mock-up of one.
                        shots.append((caption, render(lines, caption)))
                        caption = v
                lines.append(ln)
                got_new = True
                print("  " + ln[:110], flush=True)
            writer.append_data(render(lines, caption))
            frames += 1
            if time.time() - t0 > 600:
                lines.append("[recording stopped at the 10 minute limit]")
                break
        # hold the final frame so the summary is readable
        caption = "demonstration complete"
        for _ in range(FPS * 4):
            writer.append_data(render(lines, caption))
            frames += 1
    finally:
        writer.close()

    txt.write_text("\n".join(lines), encoding="utf-8")
    shots.append((caption, render(lines, caption)))
    ev = RESULTS / "mininet_evidence"
    ev.mkdir(parents=True, exist_ok=True)
    (ev / "live_demo_transcript.txt").write_text("\n".join(lines), encoding="utf-8")

    # Stills of each stage, for the figure in the paper and for anyone who
    # asked to see the system running. These are frames of the recording, not
    # redrawn mock-ups.
    shot_dir = HERE.parent / "06_figures" / "mininet"
    shot_dir.mkdir(parents=True, exist_ok=True)
    for i, (cap, arr) in enumerate(shots, 1):
        name = f"mininet_step{i:02d}.png"
        Image.fromarray(arr).save(shot_dir / name)
        Image.fromarray(arr).save(ev / name)
    print(f"wrote {len(shots)} stills to {shot_dir}")

    dur = frames / FPS
    print(f"\nwrote {mp4}  ({mp4.stat().st_size/1e6:.1f} MB, {dur:.0f} s, "
          f"{frames} frames)")
    print(f"wrote {txt}")
    print(f"session exit code: {rc_box.get('rc')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
