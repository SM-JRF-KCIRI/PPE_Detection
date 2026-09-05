"""
feed_monitor.py
=============================================================================
Drop-in monitor for your REAL running system (patrol_track_main.py +
ptz_shared.py). Unlike benchmark_feed_methods.py (which tests the two
encoding approaches in isolation with synthetic frames), this module
attaches to your actual FeedServer instance while everything else -
YOLO, ByteTrack, PTZ patrol/tracking, the real camera - is running live.
It measures the true cost of the MediaMTX/FFmpeg pipeline on your Jetson,
under real load.

WHAT IT MEASURES (sampled every --interval seconds, default 2s):
  - CPU% and RAM (RSS) of every subprocess FeedServer owns:
        MediaMTX itself, the "raw" FFmpeg publisher, the "annotated"
        FFmpeg publisher  (each is a separate OS process/PID)
  - How often update_raw()/update_annotated() are actually called
    (your real achieved fps pushed INTO the feed pipeline) and how long
    each call takes (should be near-zero since it's just a queue.put)
  - A live one-line status printed every --interval seconds, plus a
    JSON summary written on shutdown (Ctrl+C) so you can review it later

HOW TO USE - two lines added to patrol_track_main.py:

    from feed_monitor import FeedServerMonitor
    ...
    feed_server = FeedServer(port=FEED_SERVER_PORT)
    feed_server.start()
    monitor = FeedServerMonitor(feed_server)      # <-- ADD THIS
    monitor.start()                                # <-- ADD THIS

And in the `finally:` block where you already call feed_server-related
cleanup, add:

    monitor.stop()                                 # <-- ADD THIS

That's it - no other changes needed. It does NOT modify FeedServer's
behavior; it only wraps update_raw/update_annotated to time them (the
wrapped calls still do exactly what they did before) and reads process
stats from the outside via psutil.

Works with the MediaMTX-based FeedServer (the one with self._mtx_proc,
self._raw_pub._proc, self._ann_pub._proc). If you're running the older
Flask/MJPEG-based FeedServer instead, this will still track call-timing
stats but will report "no subprocess" for the CPU/memory rows, since
that version doesn't spawn FFmpeg/MediaMTX subprocesses.
"""

import json
import threading
import time
import statistics

try:
    import psutil
except ImportError:
    psutil = None


class _CallTimer:
    """Tracks how often a function is called and how long each call takes,
    without changing what the function does."""

    def __init__(self):
        self.lock = threading.Lock()
        self.total_count = 0     # lifetime total - never reset, used in the final summary
        self.window_count = 0    # calls since the last reset_window() - used for live fps
        self.durations_ms = []
        self.last_call_time = None
        self.window_start = time.time()

    def record(self, duration_ms):
        with self.lock:
            self.total_count += 1
            self.window_count += 1
            self.durations_ms.append(duration_ms)
            self.last_call_time = time.time()
            # Cap memory: keep only the most recent 2000 samples
            if len(self.durations_ms) > 2000:
                self.durations_ms = self.durations_ms[-2000:]

    def achieved_fps_since_window_start(self):
        with self.lock:
            elapsed = time.time() - self.window_start
            return self.window_count / elapsed if elapsed > 0 else 0.0

    def reset_window(self):
        with self.lock:
            self.window_start = time.time()
            self.window_count = 0

    def summary(self):
        with self.lock:
            if not self.durations_ms:
                return {"calls": self.total_count, "avg_ms": None, "max_ms": None}
            return {
                "calls": self.total_count,
                "avg_ms": round(statistics.mean(self.durations_ms), 4),
                "max_ms": round(max(self.durations_ms), 4),
            }


class FeedServerMonitor:
    """Attaches to a live FeedServer instance and reports its real
    resource cost while your patrol/tracking system runs."""

    def __init__(self, feed_server, interval=2.0, log_path=None):
        """
        feed_server : the FeedServer instance you already created and
                      called .start() on in patrol_track_main.py.
        interval    : seconds between printed status lines.
        log_path    : optional path to also append each sample as a
                      JSON line (for later plotting/analysis).
        """
        self.feed_server = feed_server
        self.interval = interval
        self.log_path = log_path

        self._raw_timer = _CallTimer()
        self._ann_timer = _CallTimer()
        self._running = False
        self._thread = None
        self._history = []  # list of per-interval sample dicts, for the final summary

        self._wrap_feed_server()

    # ------------------------------------------------------------------
    # Wrap update_raw / update_annotated so we can time real call rate
    # without touching FeedServer's own code.
    # ------------------------------------------------------------------
    def _wrap_feed_server(self):
        original_update_raw = self.feed_server.update_raw
        original_update_annotated = self.feed_server.update_annotated

        def timed_update_raw(frame):
            t0 = time.perf_counter()
            result = original_update_raw(frame)
            self._raw_timer.record((time.perf_counter() - t0) * 1000.0)
            return result

        def timed_update_annotated(frame):
            t0 = time.perf_counter()
            result = original_update_annotated(frame)
            self._ann_timer.record((time.perf_counter() - t0) * 1000.0)
            return result

        self.feed_server.update_raw = timed_update_raw
        self.feed_server.update_annotated = timed_update_annotated

    # ------------------------------------------------------------------
    # Find the real OS subprocesses FeedServer owns (MediaMTX-based
    # version only - _mtx_proc / _raw_pub._proc / _ann_pub._proc).
    # ------------------------------------------------------------------
    def _get_tracked_processes(self):
        procs = {}
        fs = self.feed_server

        mtx = getattr(fs, "_mtx_proc", None)
        if mtx is not None and mtx.poll() is None:
            procs["mediamtx"] = mtx.pid

        raw_pub = getattr(fs, "_raw_pub", None)
        if raw_pub is not None:
            p = getattr(raw_pub, "_proc", None)
            if p is not None and p.poll() is None:
                procs["ffmpeg_raw"] = p.pid

        ann_pub = getattr(fs, "_ann_pub", None)
        if ann_pub is not None:
            p = getattr(ann_pub, "_proc", None)
            if p is not None and p.poll() is None:
                procs["ffmpeg_annotated"] = p.pid

        return procs

    def _sample_processes(self):
        """Return {label: {cpu_pct, rss_mb}} for every tracked subprocess
        that's currently alive. Returns {} if psutil isn't installed or
        FeedServer isn't the MediaMTX-based version."""
        if psutil is None:
            return {}

        stats = {}
        for label, pid in self._get_tracked_processes().items():
            try:
                p = psutil.Process(pid)
                stats[label] = {
                    "cpu_pct": p.cpu_percent(interval=None),
                    "rss_mb": round(p.memory_info().rss / (1024 * 1024), 1),
                }
            except psutil.NoSuchProcess:
                continue
        return stats

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------
    def _loop(self):
        # Prime cpu_percent() for each process (first call always
        # returns 0.0 - needs a baseline before it's meaningful).
        self._sample_processes()
        time.sleep(min(1.0, self.interval))

        while self._running:
            time.sleep(self.interval)
            if not self._running:
                break

            proc_stats = self._sample_processes()
            raw_fps = self._raw_timer.achieved_fps_since_window_start()
            ann_fps = self._ann_timer.achieved_fps_since_window_start()
            self._raw_timer.reset_window()
            self._ann_timer.reset_window()

            sample = {
                "t": round(time.time(), 1),
                "raw_fps": round(raw_fps, 1),
                "annotated_fps": round(ann_fps, 1),
                "processes": proc_stats,
            }
            self._history.append(sample)
            self._print_line(sample)

            if self.log_path:
                try:
                    with open(self.log_path, "a") as f:
                        f.write(json.dumps(sample) + "\n")
                except Exception as exc:
                    print(f"[FeedMonitor] Could not write log line: {exc}")

    def _print_line(self, sample):
        parts = [f"[FeedMonitor] raw={sample['raw_fps']}fps ann={sample['annotated_fps']}fps"]
        if not sample["processes"]:
            parts.append("| no FFmpeg/MediaMTX subprocess found (MJPEG-based FeedServer, or psutil missing)")
        else:
            for label, s in sample["processes"].items():
                parts.append(f"| {label}: {s['cpu_pct']}% CPU, {s['rss_mb']}MB")
        print(" ".join(parts))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self):
        if psutil is None:
            print("[FeedMonitor] WARNING: psutil not installed - CPU/memory tracking disabled. "
                  "Install with: pip3 install psutil --break-system-packages")
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="FeedServerMonitor")
        self._thread.start()
        print(f"[FeedMonitor] Started (sampling every {self.interval}s).")

    def stop(self, summary_path=None):
        """Stop monitoring and print/save a final summary. Call this from
        your shutdown/finally block."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.interval + 2.0)

        summary = self._build_summary()
        self._print_summary(summary)

        if summary_path:
            try:
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2)
                print(f"[FeedMonitor] Summary written to {summary_path}")
            except Exception as exc:
                print(f"[FeedMonitor] Could not write summary: {exc}")

        return summary

    def _build_summary(self):
        raw_fps_samples = [h["raw_fps"] for h in self._history if h["raw_fps"] > 0]
        ann_fps_samples = [h["annotated_fps"] for h in self._history if h["annotated_fps"] > 0]

        cpu_by_process = {}
        mem_by_process = {}
        for h in self._history:
            for label, s in h["processes"].items():
                cpu_by_process.setdefault(label, []).append(s["cpu_pct"])
                mem_by_process.setdefault(label, []).append(s["rss_mb"])

        return {
            "duration_samples": len(self._history),
            "raw_push": {
                "call_stats_ms": self._raw_timer.summary(),
                "avg_fps": round(statistics.mean(raw_fps_samples), 1) if raw_fps_samples else None,
            },
            "annotated_push": {
                "call_stats_ms": self._ann_timer.summary(),
                "avg_fps": round(statistics.mean(ann_fps_samples), 1) if ann_fps_samples else None,
            },
            "processes": {
                label: {
                    "avg_cpu_pct": round(statistics.mean(vals), 1),
                    "max_cpu_pct": round(max(vals), 1),
                    "avg_rss_mb": round(statistics.mean(mem_by_process[label]), 1),
                    "max_rss_mb": round(max(mem_by_process[label]), 1),
                }
                for label, vals in cpu_by_process.items()
            },
        }

    def _print_summary(self, summary):
        print("\n" + "=" * 60)
        print("  FEED SERVER - SESSION SUMMARY")
        print("=" * 60)
        print(f"  Samples collected     : {summary['duration_samples']} (~{summary['duration_samples'] * self.interval:.0f}s)")
        print(f"  Raw push avg fps      : {summary['raw_push']['avg_fps']}")
        print(f"  Annotated push avg fps: {summary['annotated_push']['avg_fps']}")
        print(f"  Push call cost (raw)  : {summary['raw_push']['call_stats_ms']}")
        print(f"  Push call cost (ann.) : {summary['annotated_push']['call_stats_ms']}")
        if summary["processes"]:
            print("  Subprocess load:")
            for label, s in summary["processes"].items():
                print(f"    {label:18s}: avg {s['avg_cpu_pct']}% CPU (max {s['max_cpu_pct']}%), "
                      f"avg {s['avg_rss_mb']}MB RAM (max {s['max_rss_mb']}MB)")
        else:
            print("  Subprocess load       : none found (MJPEG-based FeedServer, or psutil missing)")
        print("=" * 60 + "\n")
