"""HyperFlare backend: flare_* Prometheus counters and gauges.

The engine publishes no JSON telemetry — /health is just {"status": "ok"} — so a
sample is one /metrics scrape plus a one-off /v1/models read for the model card.
Every token and byte counter is cumulative, so live throughput comes from
delta-sampling between polls. When the engine restarts, counters go backwards:
the rate window is dropped rather than turned into a negative rate.

Metric meanings below quote the HELP text the engine publishes; anything the
engine does not publish renders as an explicit gap, never as a fabricated zero.
"""

import curses
import time
from datetime import datetime

from .common import (
    COLOR_BORDER, COLOR_DIM, COLOR_GOOD, COLOR_GREEN, COLOR_HEADER, COLOR_RED,
    COLOR_TITLE, COLOR_VALUE, COLOR_WARN, append_rate_sample, avg_wall_rate,
    draw_bar_row, draw_box_end, draw_box_start, draw_row, draw_two_col_row,
    fetch_json, fetch_metrics_labeled, format_bytes, format_latency, format_num,
    format_speed, format_uptime, quantile_strs, sparkline_fit, wall_rate_safe,
)

PREFIX = "flare_"

# Histograms published by the engine. Each carries only an `le` label (verified
# against the live engine: cumulative buckets, the "+Inf" bucket equals _count).
HOST_HISTOGRAMS = {
    "missing_tokens": "host_cache_missing_tokens",
    "prefill_ns": "host_cache_estimated_prefill_ns",
    "transfer_ns": "host_cache_estimated_transfer_ns",
}


def _finite(v):
    """A finite float, or None. NaN/Infinity pass json/text parsing but are not
    measurements, so they must not reach a rate or a bar."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        return None
    if f != f or abs(f) == float("inf"):
        return None
    return f


def _scalars(series):
    """{short_name: value} for every label-free series, 'flare_' stripped.

    Bucket lines are skipped: their last value wins in a flat map and would
    silently replace a real measurement.
    """
    out = {}
    for name, rows in series.items():
        if name.endswith("_bucket"):
            continue
        short = name[len(PREFIX):] if name.startswith(PREFIX) else name
        for labels, value in rows:
            if labels:
                continue
            f = _finite(value)
            if f is not None:
                out[short] = f
            break
    return out


def _sum(series, base, **match):
    """Sum of one series family, optionally restricted to label values."""
    total, seen = 0.0, False
    for labels, value in series.get(base, []):
        if any(labels.get(k) != v for k, v in match.items()):
            continue
        f = _finite(value)
        if f is None:
            continue
        total += f
        seen = True
    return total if seen else 0.0


def _by(series, base, label):
    """{label_value: summed_value} for one labelled counter family."""
    out = {}
    for labels, value in series.get(base, []):
        key = labels.get(label)
        if key is None:
            continue
        f = _finite(value)
        if f is None:
            continue
        out[key] = out.get(key, 0.0) + f
    return out


def _buckets(series, base):
    """{upper_bound: cumulative_count} for one histogram.

    Several label sets for the same histogram yield {} (percentiles then render
    as em dashes) instead of silently mixing unrelated series.
    """
    out = {}
    for labels, value in series.get(base + "_bucket", []):
        try:
            bound, count = float(labels.get("le")), float(value)
        except (TypeError, ValueError):
            continue
        if count != count or abs(count) == float("inf"):
            continue
        if bound in out:
            return {}
        out[bound] = count
    return out


def _pct(num, den):
    return None if not den else num / den * 100.0


def _pct_txt(num, den, digits=1):
    p = _pct(num, den)
    return "n/a" if p is None else f"{p:.{digits}f}%"


def _share_txt(num, den, digits=1):
    """Share of a label group that the engine saw at all: an em dash when the
    denominator is zero, because 0/0 is an absent measurement, not a 0% rate."""
    return "—" if not den else f"{num / den * 100.0:.{digits}f}%"


def _ms_from_ns(ns):
    return None if ns is None else ns / 1e6


def sample(host, port):
    """One HyperFlare sample: wall time + /metrics counters and gauges.

    The text exposition carries the labelled breakdowns (drafts by depth, graph
    launches by kind, scheduler reasons, HTTP APIs). The same route answers with
    one JSON object at ?format=json, which is the only place the 60-second
    first-response summary and the drafter booleans are published.
    """
    metrics, series, _ = fetch_metrics_labeled(host, port, keep_buckets=True)
    s = {"t": time.time(), "_error": metrics.get("_error")}
    s.update(_scalars(series))

    # Speculative decode: `speculative_*{batch,width}` covers every completed
    # record, `speculative_drafts_total{depth,source,outcome}` breaks the same
    # work down by one-based depth and proposal source.
    s["spec_rounds"] = _sum(series, PREFIX + "speculative_rounds_total")
    s["spec_offered"] = _sum(series, PREFIX + "speculative_offered_drafts_total")
    s["spec_accepted"] = _sum(series, PREFIX + "speculative_accepted_drafts_total")
    s["spec_emitted"] = _sum(series, PREFIX + "speculative_emitted_tokens_total")
    s["spec_gpu_ns"] = _sum(series, PREFIX + "speculative_gpu_nanoseconds_total")
    depth_offered, depth_accepted = {}, {}
    source_offered, source_accepted = {}, {}
    for labels, value in series.get(PREFIX + "speculative_drafts_total", []):
        f = _finite(value)
        depth, source, outcome = labels.get("depth"), labels.get("source"), labels.get("outcome")
        if f is None or depth is None or outcome not in ("offered", "accepted"):
            continue
        try:
            d = int(depth)
        except (TypeError, ValueError):
            continue
        if outcome == "offered":
            depth_offered[d] = depth_offered.get(d, 0.0) + f
            if source:
                source_offered[source] = source_offered.get(source, 0.0) + f
        else:
            depth_accepted[d] = depth_accepted.get(d, 0.0) + f
            if source:
                source_accepted[source] = source_accepted.get(source, 0.0) + f
    s["depth_offered"] = depth_offered
    s["depth_accepted"] = depth_accepted
    s["source_offered"] = source_offered
    s["source_accepted"] = source_accepted

    # Label breakdowns
    s["graph_by_kind"] = _by(series, PREFIX + "graph_launches_total", "kind")
    s["sampling_by_kind"] = _by(series, PREFIX + "sampling_graph_launches_total", "kind")
    s["exec_by_reason"] = _by(series, PREFIX + "scheduler_executions_total", "reason")
    s["exec_ns_by_reason"] = _by(series, PREFIX + "scheduler_execution_nanoseconds_total", "reason")
    s["http_by_api"] = _by(series, PREFIX + "http_requests_total", "api")
    s["captures_by_role"] = _by(series, PREFIX + "checkpoint_captures_total", "role")
    s["skips_by_reason"] = _by(series, PREFIX + "checkpoint_skips_total", "reason")
    s["tier_records"] = {
        "dram": s.get("host_cache_memory_records", 0.0),
        "nvme": s.get("host_cache_disk_records", 0.0),
    }
    s["tier_bytes"] = {
        "dram": s.get("host_cache_memory_bytes", 0.0),
        "nvme": s.get("host_cache_disk_bytes", 0.0),
    }
    s["tier_events"] = {
        "held": _by(series, PREFIX + "host_cache_held_total", "tier"),
        "dropped": _by(series, PREFIX + "host_cache_dropped_total", "tier"),
        "expired": _by(series, PREFIX + "host_cache_expired_total", "tier"),
        "thinned": _by(series, PREFIX + "host_cache_rows_thinned_total", "tier"),
    }

    # Histograms + their observation counts (the `_count` scalars)
    for short, base in HOST_HISTOGRAMS.items():
        s[short + "_buckets"] = _buckets(series, PREFIX + base)
        s[short + "_count"] = s.get(base + "_count", 0.0)

    s["upslots"] = s.get("active_requests", 0.0) + s.get("free_request_slots", 0.0) + s.get("retiring_request_slots", 0.0)
    s["busy"] = bool(s.get("active_requests", 0.0) > 0 or s.get("pending_requests", 0.0) > 0
                     or s.get("http_inflight_requests", 0.0) > 0)

    # JSON mirror of the same route. A build without the JSON form answers with
    # the text exposition instead (not a dict), so every field here stays None
    # and the panels print an explicit gap rather than a zero.
    snap = fetch_json(host, port, "/metrics?format=json")
    snap = snap if isinstance(snap, dict) else {}
    first_response = snap.get("first_response")
    s["first_response"] = first_response if isinstance(first_response, dict) else None
    # Boolean: the model has a learned drafter.
    s["learned_drafter"] = bool(snap["learned_drafter"]) if "learned_drafter" in snap else None
    # Draft positions run by the learned drafter; published only in JSON.
    s["draft_steps"] = _finite(snap.get("draft_steps"))
    return s


def tick(stdscr, args, state):
    cur = sample(args.host, args.port)
    if not state["model_name"]:
        models = fetch_json(args.host, args.port, "/v1/models")
        if isinstance(models, dict) and models.get("data"):
            first = models["data"][0]
            state["model_name"] = first.get("id")
            flare = first.get("flare") if isinstance(first.get("flare"), dict) else {}
            state["model_meta"] = dict(flare)
            state["context_length"] = first.get("max_model_len")
            state["model_root"] = first.get("root")

    prev = state["prev"]
    if prev and not cur.get("_error"):
        if cur.get("uptime_seconds", 0.0) < prev.get("uptime_seconds", 0.0):
            # engine restart: counters restarted from zero, so the window's
            # deltas straddle two different counter lifetimes
            state["history"].clear()
            state["rate_hist"].clear()
            prev = None
    if not cur.get("_error"):
        state["history"].append(cur)
    draw(stdscr, cur, prev, state["history"], state["start_time"], args.host, args.port,
         args.interval, state["model_name"], rate_hist=state["rate_hist"],
         ctx_len=state.get("context_length"), model_root=state.get("model_root"),
         model_meta=state.get("model_meta"))
    state["prev"] = cur if not cur.get("_error") else prev


def _fmt_tier_events(events):
    """'held 0 · dropped 0 · expired 0 · thinned 0' with dram/nvme split only
    when a tier actually counted something."""
    parts = []
    for name in ("held", "dropped", "expired", "thinned"):
        by_tier = events.get(name) or {}
        total = sum(by_tier.values())
        if total and len([v for v in by_tier.values() if v]):
            split = " ".join(f"{tier} {format_num(v)}" for tier, v in sorted(by_tier.items()))
            parts.append(f"{name} {format_num(total)} ({split})")
        else:
            parts.append(f"{name} {format_num(total)}")
    return " · ".join(parts)


def _fmt_rate_map(mapping, order=None, fmt=None):
    """'a 1 · b 2' sorted by value (descending), zero entries kept so a missing
    kind is visible instead of vanishing. A fixed `order` puts the known kinds
    first and still appends any kind the engine added later."""
    fmt = fmt or (lambda v: format_num(v))
    ranked = [k for k, _ in sorted(mapping.items(), key=lambda kv: (-kv[1], kv[0]))]
    keys = list(order or [])
    keys += [k for k in ranked if k not in keys]
    return " · ".join(f"{k} {fmt(mapping.get(k, 0.0))}" for k in keys)


def _window_delta(history, key):
    """Change of one counter across the sampling window (newest sample is
    history[-1]); None while the window holds fewer than two samples."""
    if not history or len(history) < 2:
        return None
    return history[-1].get(key, 0.0) - history[0].get(key, 0.0)


def draw(stdscr, cur, prev, history, start_time, host, port, interval, model_name,
         rate_hist=None, ctx_len=None, model_root=None, model_meta=None):
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    x, y = 0, 0
    title = "HyperFlare Performance Dashboard"
    try:
        stdscr.addstr(y, x, "╔" + "═" * (width - 2) + "╗", curses.color_pair(COLOR_BORDER) | curses.A_BOLD)
        y += 1
        tpad = max(0, (width - 2 - len(title)) // 2)
        line = "║" + " " * tpad + title + " " * max(0, width - 2 - tpad - len(title)) + "║"
        stdscr.addstr(y, x, line[:width], curses.color_pair(COLOR_TITLE) | curses.A_BOLD)
        y += 1
        info = (f"  Backend: hyperflare  Model: {model_name or 'unknown'}  Host: {host}:{port}"
                f"  Uptime: {format_uptime(time.time() - start_time)}")
        stdscr.addstr(y, x, "║" + info[: width - 2].ljust(width - 2) + "║", curses.color_pair(COLOR_DIM))
        y += 1
        stdscr.addstr(y, x, "╚" + "═" * (width - 2) + "╝", curses.color_pair(COLOR_BORDER) | curses.A_BOLD)
        y += 1
    except curses.error:
        pass

    if cur.get("_error"):
        try:
            stdscr.addstr(y, x, f"  Status: DOWN  {cur['_error']}"[:width],
                          curses.color_pair(COLOR_RED) | curses.A_BOLD)
        except curses.error:
            pass
        stdscr.refresh()
        return
    y += 1
    status = "BUSY" if cur["busy"] else "IDLE"
    scolor = COLOR_WARN if cur["busy"] else COLOR_GREEN
    engine_up = cur.get("uptime_seconds")
    up_txt = f"   engine up {format_uptime(engine_up)}" if engine_up else ""
    try:
        stdscr.addstr(y, x, f"  Status: OK ({status}){up_txt}", curses.color_pair(scolor) | curses.A_BOLD)
    except curses.error:
        pass
    y += 1

    # ---- Live throughput ----
    pp_live = wall_rate_safe(cur, prev, "prefill_tokens_total")
    tg_live = wall_rate_safe(cur, prev, "generated_tokens_total")
    pp_avg = avg_wall_rate(history, "prefill_tokens_total")
    tg_avg = avg_wall_rate(history, "generated_tokens_total")
    if rate_hist is not None:
        append_rate_sample(rate_hist, cur["t"], pp_live, tg_live, min_gap=interval * 0.9)

    draw_box_start(stdscr, y, x, width, "Live Throughput (delta-sampled)")
    y += 1
    draw_row(stdscr, y, x, "Prefill PP (wall):", f"{format_speed(pp_live)} tok/s   avg {format_speed(pp_avg)}", width)
    y += 1
    tg_color = COLOR_GOOD if (tg_live or 0) >= 30 else COLOR_WARN if (tg_live or 0) > 0 else COLOR_DIM
    draw_row(stdscr, y, x, "Decode TG (wall):", f"{format_speed(tg_live)} tok/s   avg {format_speed(tg_avg)}", width,
             value_color=tg_color)
    y += 1
    # HELP: committed generated tokens during the most recent minute, sampled
    # every five seconds — the engine's own windowed rate, not a delta of ours.
    last_min = cur.get("generated_tokens_last_minute")
    if last_min is None:
        draw_row(stdscr, y, x, "Decode TG (engine 60s):", "n/a (not reported)", width, value_color=COLOR_DIM)
    else:
        draw_row(stdscr, y, x, "Decode TG (engine 60s):",
                 f"{format_speed(last_min / 60.0)} tok/s   ({format_num(last_min)} tok in the last minute)", width)
    y += 1
    draw_row(stdscr, y, x, "Slots:",
             f"active {int(cur.get('active_requests', 0))}  free {int(cur.get('free_request_slots', 0))}  "
             f"pending {int(cur.get('pending_requests', 0))}  retiring {int(cur.get('retiring_request_slots', 0))}"
             f"  / {int(cur['upslots'])}   HTTP inflight {int(cur.get('http_inflight_requests', 0))}"
             f"/{int(cur.get('http_request_capacity', 0))}", width)
    y += 1
    # Server-side time from HTTP admission to the first offered answer or
    # reasoning text, over a 60-second window (JSON mirror of /metrics only).
    fr = cur.get("first_response")
    if not isinstance(fr, dict):
        draw_row(stdscr, y, x, "First response:", "n/a (JSON mirror of /metrics unavailable)",
                 width, value_color=COLOR_DIM)
    else:
        median = _finite(fr.get("median_seconds"))
        samples = _finite(fr.get("samples")) or 0.0
        capacity = _finite(fr.get("capacity")) or 0.0
        window_s = _finite(fr.get("window_seconds")) or 0.0
        truncated = "truncated" if fr.get("truncated") else "complete"
        median_txt = "n/a" if median is None else format_latency(median)
        draw_row(stdscr, y, x, "First response:",
                 f"median {median_txt}   samples {format_num(samples)}/{format_num(capacity)}"
                 f" over {format_num(window_s)}s   {truncated}", width,
                 value_color=COLOR_VALUE if median is not None else COLOR_DIM)
    y += 1
    if rate_hist is not None and len(rate_hist) >= 2:
        pp_pts = [r[1] for r in rate_hist]
        tg_pts = [r[2] for r in rate_hist]
        pp_max = max((v for v in pp_pts if v is not None), default=0.0)
        tg_max = max((v for v in tg_pts if v is not None), default=0.0)
        pp_label = f"PP 120s [{format_speed(pp_max)}]:"
        tg_label = f"TG 120s [{format_speed(tg_max)}]:"
        draw_row(stdscr, y, x, pp_label, f"[{sparkline_fit(pp_pts, width - len(pp_label) - 5)}]", width)
        y += 1
        draw_row(stdscr, y, x, tg_label, f"[{sparkline_fit(tg_pts, width - len(tg_label) - 5)}]", width)
        y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    # ---- KV pool ----
    # HELP: total_kv_pages counts the per-rank pool BEFORE checkpoint retention,
    # free_kv_pages is reusable after completed fences, cached_kv_pages is owned
    # by immutable checkpoints. In-use pages are therefore total - free - cached.
    total_pages = cur.get("total_kv_pages", 0.0)
    free_pages = cur.get("free_kv_pages", 0.0)
    cached_pages = cur.get("cached_kv_pages", 0.0)
    used_pages = max(total_pages - free_pages - cached_pages, 0.0)
    total_tokens = cur.get("total_kv_tokens", 0.0)
    free_tokens = cur.get("free_kv_tokens", 0.0)
    cached_tokens = cur.get("cached_kv_tokens", 0.0)
    used_tokens = max(total_tokens - free_tokens - cached_tokens, 0.0)
    used_pct = _pct(used_pages, total_pages) or 0.0
    cached_pct = _pct(cached_pages, total_pages) or 0.0
    kv_color = COLOR_GREEN if used_pct < 70 else COLOR_WARN if used_pct < 90 else COLOR_RED

    draw_box_start(stdscr, y, x, width, "KV pool")
    y += 1
    if total_pages > 0:
        draw_bar_row(stdscr, y, x, "KV pages in use:", used_pct, width, kv_color)
        y += 1
        draw_bar_row(stdscr, y, x, "Checkpoint pages:", cached_pct, width, COLOR_HEADER)
        y += 1
    else:
        draw_row(stdscr, y, x, "KV pages:", "n/a (engine reports no KV page pool)", width,
                 value_color=COLOR_DIM)
        y += 1
    draw_row(stdscr, y, x, "Pages:",
             f"{format_num(used_pages)} / {format_num(total_pages)} used   free {format_num(free_pages)}"
             f"   {format_num(cur.get('kv_page_tokens', 0))} tok/page", width)
    y += 1
    draw_row(stdscr, y, x, "Tokens:",
             f"{format_num(used_tokens)} / {format_num(total_tokens)}   free {format_num(free_tokens)}"
             f"   cached {format_num(cached_tokens)}", width)
    y += 1
    snap_cap = cur.get("snapshot_capacity", 0.0)
    snap_free = cur.get("free_snapshots", 0.0)
    draw_row(stdscr, y, x, "Snapshots:",
             f"{format_num(snap_free)} / {format_num(snap_cap)} free   backing {format_num(cur.get('snapshot_backing_pages', 0))} pages"
             f"   published {format_num(cur.get('snapshot_publications_total', 0))}"
             f"   evictions {format_num(cur.get('snapshot_evictions_total', 0))}"
             f"   checkpoints {format_num(cur.get('cached_checkpoints', 0))}", width)
    y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    # ---- Prefix cache ----
    draw_box_start(stdscr, y, x, width, "Prefix cache")
    y += 1
    gpu_lookups = cur.get("cache_lookups_total", 0.0)
    gpu_hits = cur.get("cache_hits_total", 0.0)
    draw_row(stdscr, y, x, "GPU:",
             f"lookups {format_num(gpu_lookups)}   hits {_pct_txt(gpu_hits, gpu_lookups)}"
             f"   reused {format_num(cur.get('cache_reused_tokens_total', 0))} tok"
             f"   misses {format_num(cur.get('cache_misses_total', 0))}", width)
    y += 1
    host_lookups = cur.get("host_cache_lookups_total", 0.0)
    host_hits = cur.get("host_cache_hits_total", 0.0)
    draw_row(stdscr, y, x, "Host store:",
             f"lookups {format_num(host_lookups)}   hits {_pct_txt(host_hits, host_lookups)}"
             f"   reused {format_num(cur.get('host_cache_reused_tokens_total', 0))} tok", width)
    y += 1
    draw_row(stdscr, y, x, "Store:",
             f"stored {format_num(cur.get('host_cache_stored_total', 0))}"
             f"  skipped {format_num(cur.get('host_cache_store_skipped_total', 0))} (duplicates/retries)"
             f"  failed {format_num(cur.get('host_cache_store_failed_total', 0))}"
             f"   ·   fetched {format_num(cur.get('host_cache_fetched_total', 0))}"
             f"  failed {format_num(cur.get('host_cache_fetch_failed_total', 0))}"
             f"  refills {format_num(cur.get('host_cache_row_refills_total', 0))}", width)
    y += 1
    out_b = cur.get("host_cache_bytes_out_total", 0.0)
    in_b = cur.get("host_cache_bytes_in_total", 0.0)
    draw_two_col_row(stdscr, y, x, width, "Copied out:",
                     f"{format_bytes(out_b)} · {format_bytes(cur.get('host_cache_outbound_bytes_per_second', 0))}/s",
                     "Copied in:",
                     f"{format_bytes(in_b)} · {format_bytes(cur.get('host_cache_inbound_bytes_per_second', 0))}/s")
    y += 1
    records = cur.get("tier_records", {})
    tier_bytes = cur.get("tier_bytes", {})
    draw_row(stdscr, y, x, "Resident:",
             f"DRAM {format_num(records.get('dram', 0))} rec / {format_bytes(tier_bytes.get('dram', 0))}"
             f"   NVMe {format_num(records.get('nvme', 0))} rec / {format_bytes(tier_bytes.get('nvme', 0))}"
             f"   parked {format_num(cur.get('host_cache_parked_records', 0))}"
             f"   slots {format_num(cur.get('host_cache_slots_in_use', 0))}/{format_num(cur.get('host_cache_slots_total', 0))}"
             f" (intake {format_num(cur.get('host_cache_intake_slots', 0))})"
             f"   inflight out {format_num(cur.get('host_cache_outbound_in_flight', 0))}"
             f" in {format_num(cur.get('host_cache_inbound_in_flight', 0))}"
             f" waiting {format_num(cur.get('host_cache_inbound_waiting', 0))}", width)
    y += 1
    draw_row(stdscr, y, x, "Tier events:", _fmt_tier_events(cur.get("tier_events", {})), width)
    y += 1
    miss_q = quantile_strs(cur.get("missing_tokens_buckets"), cur.get("missing_tokens_count", 0.0),
                           fmt=format_num)
    draw_row(stdscr, y, x, "Missing tokens:",
             f"p50 {miss_q[0]}  p90 {miss_q[1]}  p99 {miss_q[2]}"
             f"   (n={format_num(cur.get('missing_tokens_count', 0))})", width)
    y += 1
    # Bounds are nanoseconds; the formatter converts them to ms/s for display.
    pre_q = quantile_strs(cur.get("prefill_ns_buckets"), cur.get("prefill_ns_count", 0.0),
                          fmt=lambda ns: format_latency(ns / 1e9))
    xfer_q = quantile_strs(cur.get("transfer_ns_buckets"), cur.get("transfer_ns_count", 0.0),
                           fmt=lambda ns: format_latency(ns / 1e9))
    draw_row(stdscr, y, x, "Est. prefill / transfer:",
             f"p50 {pre_q[0]} / {xfer_q[0]}   p90 {pre_q[1]} / {xfer_q[1]}"
             f"   p99 {pre_q[2]} / {xfer_q[2]}", width)
    y += 1
    captures = cur.get("captures_by_role", {})
    skips = cur.get("skips_by_reason", {})
    # Skip counters are selection/capture rejections, not failed requests.
    draw_two_col_row(stdscr, y, x, width, "Checkpoints captured:",
                     _fmt_rate_map(captures, order=("anchor", "endpoint", "message")),
                     "rejected:",
                     _fmt_rate_map(skips, order=("storage", "unknown_cost", "work_budget")))
    y += 1
    extra_work = cur.get("checkpoint_extra_work_nanoseconds_total", 0.0) / 1e9
    draw_row(stdscr, y, x, "Checkpoint overhead:",
             f"{format_latency(extra_work)} charged extra work (predicted, not a GPU measurement)"
             f"   ·   {format_num(cur.get('checkpoint_anchor_reservations', 0))} unused anchor guards"
             f"   ·   abandoned stores {format_num(cur.get('host_cache_abandoned_stores_total', 0))}", width)
    y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    # ---- Speculative decode ----
    # HELP: draft_request_rounds_total is the denominator for mean acceptance
    # length and counts rounds that offered at least one draft, so the mean
    # acceptance length is 1 + accepted drafts per such round.
    draw_box_start(stdscr, y, x, width, "Speculative decode")
    y += 1
    acc_tok = cur.get("accepted_draft_tokens_total", 0.0)
    req_rounds = cur.get("draft_request_rounds_total", 0.0)
    ver_tok = cur.get("verification_draft_tokens_total", 0.0)
    ver_rounds = cur.get("verification_request_rounds_total", 0.0)
    # Both headline numbers are window deltas, as the engine's own dashboard
    # defines them: DEPTH is the change in verification_drafts over the change
    # in verification_rounds, and the mean acceptance length is 1 + accepted
    # drafts per round that offered at least one draft (the bonus token is not
    # part of accepted_drafts). An idle window has no denominator and shows n/a.
    d_acc = _window_delta(history, "accepted_draft_tokens_total")
    d_rounds = _window_delta(history, "draft_request_rounds_total")
    d_ver_tok = _window_delta(history, "verification_draft_tokens_total")
    d_ver_rounds = _window_delta(history, "verification_request_rounds_total")
    mean_len = 1.0 + d_acc / d_rounds if d_rounds and d_rounds > 0 and d_acc is not None else None
    depth = d_ver_tok / d_ver_rounds if d_ver_rounds and d_ver_rounds > 0 and d_ver_tok is not None else None
    mean_txt = "n/a" if mean_len is None else f"{mean_len:.2f} tok/round"
    depth_txt = "n/a" if depth is None else f"{depth:.2f} positions/round"
    acc_color = COLOR_GREEN if (mean_len or 0) >= 2.5 else COLOR_WARN if (mean_len or 0) > 1 else COLOR_DIM
    draw_row(stdscr, y, x, "Acceptance (window):",
             f"mean {mean_txt}   planned verification {depth_txt}", width, value_color=acc_color)
    y += 1
    draw_row(stdscr, y, x, "Draft rounds:",
             f"{format_num(req_rounds)} offering drafts   {format_num(ver_rounds)} decode/verification"
             f"   {format_num(acc_tok)} accepted   {format_num(ver_tok)} allocated", width)
    y += 1
    spec_rounds = cur.get("spec_rounds", 0.0)
    gpu_ms = _ms_from_ns(cur.get("spec_gpu_ns"))
    per_round = gpu_ms / spec_rounds if spec_rounds and gpu_ms is not None else None
    per_round_txt = "n/a" if per_round is None else f"{per_round:.1f} ms/round"
    draw_row(stdscr, y, x, "Spec records:",
             f"rounds {format_num(spec_rounds)}   offered {format_num(cur.get('spec_offered', 0))}"
             f"   accepted {format_num(cur.get('spec_accepted', 0))}"
             f" ({_pct_txt(cur.get('spec_accepted', 0), cur.get('spec_offered', 0))})"
             f"   emitted {format_num(cur.get('spec_emitted', 0))} tok"
             f"   GPU {per_round_txt}", width)
    y += 1
    src_off, src_acc = cur.get("source_offered", {}), cur.get("source_accepted", {})
    src_parts = []
    for source in sorted(set(src_off) | set(src_acc)):
        src_parts.append(f"{source} {_share_txt(src_acc.get(source, 0.0), src_off.get(source, 0.0))}"
                         f" ({format_num(src_acc.get(source, 0.0))}/{format_num(src_off.get(source, 0.0))})")
    draw_row(stdscr, y, x, "By source:", "   ".join(src_parts) or "n/a (no draft outcomes yet)", width,
             value_color=COLOR_VALUE if src_parts else COLOR_DIM)
    y += 1
    d_off, d_acc = cur.get("depth_offered", {}), cur.get("depth_accepted", {})
    depth_parts = [f"d{d} {_share_txt(d_acc.get(d, 0.0), d_off.get(d, 0.0))}" for d in sorted(set(d_off) | set(d_acc))]
    draw_row(stdscr, y, x, "By depth:", "   ".join(depth_parts) or "n/a (no drafts yet)", width,
             value_color=COLOR_VALUE if depth_parts else COLOR_DIM)
    y += 1
    hist = cur.get("history_drafts_enabled")
    hist_txt = "n/a (not reported)" if hist is None else ("on" if hist else "off")
    learned = cur.get("learned_drafter")
    learned_txt = "n/a" if learned is None else ("yes" if learned else "no")
    draft_steps = cur.get("draft_steps")
    steps_txt = "n/a" if draft_steps is None else format_num(draft_steps)
    draw_row(stdscr, y, x, "Drafters:",
             f"learned {learned_txt}   draft steps {steps_txt}   history {hist_txt}"
             f"   ·   mixed grammar steps {format_num(cur.get('mixed_grammar_decode_steps_total', 0))}"
             f"   reasoning cuts {format_num(cur.get('reasoning_budget_cuts_total', 0))}", width)
    y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    # ---- Scheduler & graphs ----
    draw_box_start(stdscr, y, x, width, "Scheduler & graphs")
    y += 1
    exec_by_reason = cur.get("exec_by_reason", {})
    exec_total = sum(exec_by_reason.values())
    replay_s = cur.get("replay_gpu_nanoseconds_total", 0.0) / 1e9
    replay_per = replay_s / exec_total * 1000 if exec_total else None
    draw_row(stdscr, y, x, "Executions:",
             f"{format_num(exec_total)}   GPU replay {replay_s:.1f}s"
             f"   {('n/a' if replay_per is None else f'{replay_per:.2f} ms')} / execution"
             f"   readiness waits {cur.get('replay_readiness_wait_nanoseconds_total', 0.0) / 1e9:.2f}s", width)
    y += 1
    ns_by_reason = cur.get("exec_ns_by_reason", {})
    reason_parts = []
    for reason, count in sorted(exec_by_reason.items(), key=lambda kv: (-kv[1], kv[0])):
        ns = ns_by_reason.get(reason, 0.0)
        reason_parts.append(f"{reason} {ns / count / 1e6:.1f} ms" if count else f"{reason} n/a")
    draw_row(stdscr, y, x, "Per execution:",
             "   ".join(reason_parts) or "n/a (no executions yet)", width,
             value_color=COLOR_VALUE if reason_parts else COLOR_DIM)
    y += 1
    graph_by_kind = cur.get("graph_by_kind", {})
    draw_row(stdscr, y, x, "Graph launches:", _fmt_rate_map(graph_by_kind) or "n/a", width)
    y += 1
    sampling = cur.get("sampling_by_kind", {})
    draw_row(stdscr, y, x, "Sampling graphs:", _fmt_rate_map(sampling) or "n/a", width)
    y += 1
    http_by_api = cur.get("http_by_api", {})
    draw_row(stdscr, y, x, "HTTP:",
             f"{_fmt_rate_map(http_by_api, order=('chat', 'completions', 'messages', 'responses', 'count_tokens'))}"
             f"   inflight {int(cur.get('http_inflight_requests', 0))}/{int(cur.get('http_request_capacity', 0))}", width)
    y += 1
    draw_row(stdscr, y, x, "Requests:",
             f"admitted {format_num(cur.get('requests_admitted_total', 0))}"
             f"   completed {format_num(cur.get('requests_completed_total', 0))}"
             f"   failed {format_num(cur.get('requests_failed_total', 0))}"
             f"   cancelled {format_num(cur.get('requests_cancelled_total', 0))}"
             f"   preemptions {format_num(cur.get('preemptions_total', 0))}", width)
    y += 1
    draw_row(stdscr, y, x, "Host side:",
             f"frontend workers {int(cur.get('frontend_active_workers', 0))}/{int(cur.get('frontend_worker_capacity', 0))}"
             f"   grammar frames {format_num(cur.get('constrained_max_inflight', 0))}"
             f"   reasoning cuts {format_num(cur.get('reasoning_budget_cuts_total', 0))}", width)
    y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    # ---- Model card (from /v1/models, fetched once) ----
    meta = model_meta if isinstance(model_meta, dict) else {}
    draw_box_start(stdscr, y, x, width, "Model card")
    y += 1
    draw_row(stdscr, y, x, "Model:",
             f"{model_name or 'unknown'}   root {model_root or 'n/a'}"
             f"   ctx {format_num(ctx_len or 0)}", width)
    y += 1
    modalities = "/".join(meta.get("input_modalities") or []) or "n/a"
    outputs = "/".join(meta.get("output_modalities") or []) or "n/a"
    draw_row(stdscr, y, x, "Config:",
             f"max concurrency {int(meta.get('max_concurrency', 0))}"
             f"   prefill chunk {format_num(meta.get('prefill_chunk_size', 0))}"
             f"   vocab {format_num(meta.get('vocab_size', 0))}"
             f"   top logprobs {int(meta.get('max_top_logprobs', 0))}"
             f"   in {modalities} → out {outputs}", width)
    y += 1
    # HELP: bytes of weights one decode step reads for one request, as the core
    # declares; zero means undeclared, not "no weights are read".
    weights = cur.get("decode_weight_bytes", 0.0)
    weights_txt = f"{format_bytes(weights)} per step per request" if weights else "n/a (core declares none)"
    draw_row(stdscr, y, x, "Decode weights:",
             f"{weights_txt}   ·   image window {format_num(meta.get('image_feature_window_tokens', 0))} tok",
             width, value_color=COLOR_VALUE if weights else COLOR_DIM)
    y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    try:
        stdscr.addstr(y, x, "─" * width, curses.color_pair(COLOR_BORDER))
        y += 1
        footer = (f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  q=quit  r=reset  c=clear rates  "
                  f"|  {interval}s refresh  |  window {len(history)} samples")
        stdscr.addstr(y, x, footer[:width], curses.color_pair(COLOR_DIM))
    except curses.error:
        pass
    stdscr.refresh()
