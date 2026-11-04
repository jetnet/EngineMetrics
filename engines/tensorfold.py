"""TensorFold backend: tensorfold:* metrics + /health JSON; cumulative counters
are delta-sampled between polls for live PP/TG."""

import curses
import time
from collections import deque
from datetime import datetime

from .common import (
    COLOR_BORDER, COLOR_DIM, COLOR_GOOD, COLOR_GREEN, COLOR_HEADER, COLOR_RED,
    COLOR_TITLE, COLOR_WARN, avg_rate, avg_wall_rate, draw_bar_row, draw_box_end,
    draw_box_start, draw_row, draw_two_col_row, fetch_json, fetch_metrics_labeled,
    format_bytes, format_latency, format_num, format_speed, format_uptime, get_value,
    percentile_from_buckets, prune_rate_hist, rate, sparkline, wall_rate, wall_rate_safe,
)

TF_COUNTERS = [
    "tensorfold:prompt_tokens_total",
    "tensorfold:generation_tokens_total",
    "tensorfold:mtp_drafted_total",
    "tensorfold:mtp_accepted_total",
    "tensorfold:request_latency_seconds_sum",
    "tensorfold:request_latency_seconds_count",
    "tensorfold:time_to_first_token_seconds_sum",
    "tensorfold:time_to_first_token_seconds_count",
]

# The original code read these from /health. A server observed on localhost:8080
# returns only a slim /health (status, live{...}, max_batch_size, ...) without them.
# /health still wins when it carries the key; otherwise fall back to /metrics below.
HEALTH_COUNTERS = [
    "requests_total", "prompt_tokens_total", "completion_tokens_total",
    "cached_tokens_total", "rounds_total", "drafted_total", "accepted_total",
    "prefill_seconds_total", "decode_seconds_total",
    "pool_tokens", "pool_free_tokens", "kept_prompts", "context_length",
]

METRIC_FALLBACK = {
    "requests_total": "tensorfold:request_latency_seconds_count",   # finished requests
    "prompt_tokens_total": "tensorfold:prompt_tokens_total",
    "cached_tokens_total": "tensorfold:prompt_tokens_cached_total",
    "rounds_total": "tensorfold:decode_rounds_total",
    "drafted_total": "tensorfold:mtp_drafted_total",
    "accepted_total": "tensorfold:mtp_accepted_total",
    "prefill_seconds_total": "tensorfold:request_prefill_seconds_sum",
    "decode_seconds_total": "tensorfold:request_decode_seconds_sum",
}


def _num(v):
    try:
        v = float(v or 0)
    except (TypeError, ValueError):
        return 0.0
    return v if v == v and abs(v) != float("inf") else 0.0


def _buckets(series, base):
    """{upper_bound: cumulative_count} for one histogram. The '+Inf' bound is
    kept as float('inf') and dropped later by the percentile helper. A server
    that exposes no buckets (slim /metrics, older build) yields {}.

    ASSUMPTION: one label set per histogram. Verified against the live engine
    (jetspark01:8045): the only label is `le`; buckets are cumulative and the
    '+Inf' bucket equals `_count`. Several label sets on one histogram yield {}
    (the percentiles then render as em dashes) rather than a silently mixed
    series.
    """
    out = {}
    for labels, v in series.get(base + "_bucket", []):
        try:
            bound, count = float(labels.get("le")), float(v)
        except (TypeError, ValueError):
            continue
        if count != count or abs(count) == float("inf"):
            continue    # NaN/Inf count: unusable bucket, better skipped than 0
        if bound in out:
            return {}   # several label sets: refuse to silently mix series
        out[bound] = count
    return out


def _quantile_strs(buckets, total, qs=(0.5, 0.9, 0.99)):
    """p50/p90/p99 as display strings, or em dashes when the server exposes no
    histogram or nothing has finished yet.

    '>x' when the quantile falls past the last finite bucket: the tail is not
    observable there, so claiming the bound as exact would be a lie. '~x'
    otherwise, because the value is interpolated inside a coarse bucket rather
    than measured.
    """
    finite = sorted((u, c) for u, c in (buckets or {}).items() if u != float("inf"))
    out = []
    for q in qs:
        if total <= 0 or not finite:
            out.append("—")
            continue
        if q * total > finite[-1][1]:
            out.append(f">{format_latency(finite[-1][0])}")
            continue
        v = percentile_from_buckets(buckets, total, q)
        out.append("—" if v is None else f"~{format_latency(v)}")
    return out


# /health "spill" dict (patch 0088 disk tier) — flattened as spill_<field> so
# delta-rate math sees flat counters. Older engines without the tier simply
# don't set spill_present.
SPILL_FIELDS = [
    "gib", "entries", "stored_bytes", "written", "written_bytes", "loaded",
    "loaded_bytes", "loaded_tokens", "restore_ms_p50", "restore_ms_p95",
    "write_failed", "crc_failed", "load_failed", "load_unfit", "disk_free_bytes",
]


def sample(host, port):
    """One TensorFold sample: wall time + counters from /metrics + /health."""
    # keep_buckets: the two histograms' buckets are the only source for tail
    # latency — the dashboard used to show sum/count averages only.
    metrics, series, _ = fetch_metrics_labeled(host, port, keep_buckets=True)
    health = fetch_json(host, port, "/health")
    if not isinstance(health, dict):
        health = {}
    s = {"t": time.time(), "_error": metrics.get("_error")}
    for k in TF_COUNTERS:
        s[k] = get_value(metrics, k)
    for k in HEALTH_COUNTERS:
        if k in health:
            s[k] = _num(health[k])
        elif k in METRIC_FALLBACK:
            s[k] = _num(get_value(metrics, METRIC_FALLBACK[k]))
        else:
            s[k] = 0.0
    gen_total = _num(get_value(metrics, "tensorfold:generation_tokens_total"))
    gen_running = _num(get_value(metrics, "tensorfold:generation_tokens_running"))
    s["generation_tokens_total"] = gen_total
    if "completion_tokens_total" not in health:
        # generation_tokens_total appears to move only when a request finishes, while
        # generation_tokens_running looks like in-flight tokens. Their sum is used as a
        # live count for the wall-clock TG rate. ASSUMPTION: the gauge drops by exactly
        # what the counter gains at finish; only checked empirically on a short window.
        s["completion_tokens_total"] = gen_total + gen_running
    # Is any cached-token counter exposed? One server build exposed
    # prompt_tokens_cached_total, a later restart did not; absent != zero.
    s["cache_known"] = ("cached_tokens_total" in health
                        or "tensorfold:prompt_tokens_cached_total" in metrics)
    # tokens actually prefilled: in one measured build the /metrics prompt counter
    # included cache hits, so subtract them. Without a cached counter the split is
    # unknown (prompt may or may not include cached tokens); draw() then shows n/a.
    if "prompt_tokens_total" in health:
        s["prefill_tokens_total"] = s["prompt_tokens_total"]
    else:
        s["prefill_tokens_total"] = max(s["prompt_tokens_total"] - s["cached_tokens_total"], 0.0)
    s["requests_running"] = _num(get_value(metrics, "tensorfold:requests_running",
                                           get_value(metrics, "tensorfold:num_requests_running")))
    s["requests_waiting"] = _num(get_value(metrics, "tensorfold:requests_waiting",
                                           get_value(metrics, "tensorfold:num_requests_waiting")))
    s["kv_ratios"] = [(lb.get("pool", lb.get("stream", "?")), _num(v))
                      for lb, v in series.get("tensorfold:kv_cache_usage_ratio", [])]
    s["lat_buckets"] = _buckets(series, "tensorfold:request_latency_seconds")
    s["ttft_buckets"] = _buckets(series, "tensorfold:time_to_first_token_seconds")
    s["streams"] = health.get("streams", {}) or {}
    s["multi_prefill"] = health.get("multi_prefill", {}) or {}
    # Cache-reuse picture from /health. Only prefix_waits*, chats_continued* and
    # kept_prompts have authoritative /metrics HELP text; kept_mix and
    # kept_bytes_cap are not documented there, so they are shown under their own
    # key names and never interpreted (a cap of 0 is NOT claimed to mean
    # "unlimited"). Absent keys stay None so the panel can print an honest gap.
    reuse = {k: (_num(health[k]) if k in health else None)
             for k in ("kept_bytes", "kept_bytes_cap", "prefix_waits", "prefix_wait_tokens",
                       "chats_continued", "chats_continued_prefill_tokens")}
    mix = health.get("kept_mix")
    reuse["mix"] = ({k: _num(mix[k]) for k in ("own", "shared", "chats", "rows") if k in mix}
                    if isinstance(mix, dict) else None)
    s["reuse"] = reuse
    live = health.get("live")
    s["live"] = live if isinstance(live, dict) else {}
    s["max_batch_size"] = _num(health.get("max_batch_size"))
    s["kept_bytes"] = float(health.get("kept_bytes", 0) or 0)
    spill = health.get("spill") or {}
    s["spill_present"] = bool(spill)
    for k in SPILL_FIELDS:
        s[f"spill_{k}"] = float(spill.get(k, 0) or 0)
    if "busy" in health:
        s["busy"] = bool(health["busy"])
    else:
        s["busy"] = s["requests_running"] > 0 or s["requests_waiting"] > 0
    s["ok"] = bool(health.get("ok", health.get("status") == "ok"))
    return s


def tick(stdscr, args, state):
    cur = sample(args.host, args.port)
    if not state["model_name"]:
        mm = fetch_json(args.host, args.port, "/v1/models")
        if mm and mm.get("data"):
            state["model_name"] = mm["data"][0].get("id")
            state["context_length"] = mm["data"][0].get("context_length")
    if not cur.get("_error"):
        state["history"].append(cur)
    if "spill_hist" not in state:
        state["spill_hist"] = deque()
    draw(stdscr, cur, state["prev"], state["history"], state["start_time"],
         args.host, args.port, args.interval, state["model_name"], state["rate_hist"],
         ctx_len=state.get("context_length"),
         spill_hist=state["spill_hist"], spill_cap=getattr(args, "spill_cap", None))
    state["prev"] = cur if not cur.get("_error") else state["prev"]


def draw(stdscr, cur, prev, history, start_time, host, port, interval, model_name, rate_hist=None,
         ctx_len=None, spill_hist=None, spill_cap=None):
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    x = 0
    y = 0
    title = "TensorFold Performance Dashboard"
    try:
        stdscr.addstr(y, x, "╔" + "═" * (width - 2) + "╗", curses.color_pair(COLOR_BORDER) | curses.A_BOLD)
        y += 1
        tpad = max(0, (width - 2 - len(title)) // 2)
        line = "║" + " " * tpad + title + " " * max(0, width - 2 - tpad - len(title)) + "║"
        stdscr.addstr(y, x, line[:width], curses.color_pair(COLOR_TITLE) | curses.A_BOLD)
        y += 1
        info = f"  Backend: tensorfold  Model: {model_name or 'unknown'}  Host: {host}:{port}  Uptime: {format_uptime(time.time() - start_time)}"
        stdscr.addstr(y, x, "║" + info[: width - 2].ljust(width - 2) + "║", curses.color_pair(COLOR_DIM))
        y += 1
        stdscr.addstr(y, x, "╚" + "═" * (width - 2) + "╝", curses.color_pair(COLOR_BORDER) | curses.A_BOLD)
        y += 1
    except curses.error:
        pass

    if cur.get("_error"):
        try:
            stdscr.addstr(y, x, f"  Status: DOWN  {cur['_error']}"[:width], curses.color_pair(COLOR_RED) | curses.A_BOLD)
        except curses.error:
            pass
        stdscr.refresh()
        return
    y += 1
    status = "BUSY" if cur["busy"] else "IDLE"
    scolor = COLOR_WARN if cur["busy"] else COLOR_GREEN
    try:
        stdscr.addstr(y, x, f"  Status: OK ({status})", curses.color_pair(scolor) | curses.A_BOLD)
    except curses.error:
        pass
    y += 1

    # ---- Live throughput (delta-sampled) ----
    # engine-time rates (finished-request based, health counters mirror /metrics)
    cache_known = cur.get("cache_known", True)
    pp_engine_rate = rate(cur, prev, "prefill_tokens_total", "prefill_seconds_total") if cache_known else None
    tg_engine_rate = rate(cur, prev, "generation_tokens_total", "decode_seconds_total")
    # live wall rates (completion_tokens_total includes in-flight replies)
    tg_live = wall_rate(cur, prev, "completion_tokens_total")
    pp_live = wall_rate(cur, prev, "prompt_tokens_total")
    # window averages
    pp_eng_avg = avg_rate(history, "prefill_tokens_total", "prefill_seconds_total") if cache_known else None
    tg_eng_avg = avg_rate(history, "generation_tokens_total", "decode_seconds_total")
    tg_live_avg = avg_wall_rate(history, "completion_tokens_total")

    # 120s rate history for sparklines
    if rate_hist is not None:
        rate_hist.append((cur["t"], pp_engine_rate, tg_live if tg_live is not None else tg_engine_rate))
        prune_rate_hist(rate_hist, 120.0)

    draw_box_start(stdscr, y, x, width, "Live Throughput (delta-sampled)")
    y += 1
    if cache_known:
        draw_row(stdscr, y, x, "Prefill PP (engine):", f"{format_speed(pp_engine_rate)} tok/s   avg {format_speed(pp_eng_avg)}", width)
    else:
        draw_row(stdscr, y, x, "Prefill PP (engine):", "n/a (no cached-token metric; see Engine-reported)", width, value_color=COLOR_DIM)
    y += 1
    draw_row(stdscr, y, x, "Decode TG (engine):", f"{format_speed(tg_engine_rate)} tok/s   avg {format_speed(tg_eng_avg)}", width)
    y += 1
    tg_color = COLOR_GOOD if (tg_live or 0) >= 30 else COLOR_WARN if (tg_live or 0) > 0 else COLOR_DIM
    draw_row(stdscr, y, x, "TG live (wall):", f"{format_speed(tg_live)} tok/s   avg {format_speed(tg_live_avg)}", width, value_color=tg_color)
    y += 1
    st = cur["streams"]
    if st:
        draw_row(stdscr, y, x, "Streams:",
                 f"decoding {int(st.get('decoding', 0))}  prefilling {int(st.get('prefilling', 0))}  "
                 f"filling {int(st.get('filling', 0))}  paused {int(st.get('paused', 0))}  / max {int(st.get('max', 0))}",
                 width)
    else:
        draw_row(stdscr, y, x, "Requests:",
                 f"running {int(cur['requests_running'])}  waiting {int(cur['requests_waiting'])}  "
                 f"/ max batch {int(cur['max_batch_size'])}", width)
    y += 1
    lv = cur["live"]
    if lv:
        draw_row(stdscr, y, x, "Engine-reported:",
                 f"TG {format_speed(_num(lv.get('decode_tokens_per_second')))} tok/s   "
                 f"PP {format_speed(_num(lv.get('prefill_tokens_per_second')))} tok/s", width)
        y += 1
    # 120s sparklines (PP engine rate, TG live rate)
    if rate_hist is not None and len(rate_hist) >= 2:
        pp_pts = [r[1] for r in rate_hist]
        tg_pts = [r[2] for r in rate_hist]
        pp_max = max((v for v in pp_pts if v is not None), default=0.0)
        tg_max = max((v for v in tg_pts if v is not None), default=0.0)
        draw_row(stdscr, y, x, f"PP 120s [{format_speed(pp_max)}]:",
                 f"[{sparkline(pp_pts)}]", width)
        y += 1
        draw_row(stdscr, y, x, f"TG 120s [{format_speed(tg_max)}]:",
                 f"[{sparkline(tg_pts)}]", width)
        y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    # ---- Live cache (two columns: RAM pool | SSD spill tier, when present) ----
    pool = cur["pool_tokens"]
    free = cur["pool_free_tokens"]
    used_pct = (pool - free) / pool * 100 if pool > 0 else 0.0
    ratios = cur.get("kv_ratios", []) or []
    kv_ratio = max((v for _, v in ratios), default=0.0) * 100
    if not kv_ratio:
        kv_ratio = get_value(cur, "tensorfold:kv_cache_usage_ratio") * 100
    kept = int(cur["kept_prompts"])
    ctx = int(cur["context_length"] or ctx_len or 0)
    cache_color = COLOR_GREEN if used_pct < 70 else COLOR_WARN if used_pct < 90 else COLOR_RED
    kv_color = COLOR_GREEN if kv_ratio < 70 else COLOR_WARN if kv_ratio < 90 else COLOR_RED
    spill = bool(cur.get("spill_present"))

    draw_box_start(stdscr, y, x, width, "Live Cache")
    y += 1
    if spill:
        if pool > 0:
            draw_bar_row(stdscr, y, x, "Pool occupancy:", used_pct, width, cache_color)
            y += 1
    elif pool > 0:
        draw_bar_row(stdscr, y, x, "Pool occupancy:", used_pct, width, cache_color)
        y += 1
        draw_row(stdscr, y, x, "Pool:", f"{format_num(pool - free)} / {format_num(pool)} rows free {format_num(free)}", width)
        y += 1
        draw_row(stdscr, y, x, "Stream KV ratio:", f"{kv_ratio:.1f}% (per-stream / ctx window)", width)
        y += 1
        draw_row(stdscr, y, x, "Kept prompts:", f"{kept}   context window {format_num(ctx)}", width)
        y += 1
    else:
        # slim /health: no shared-pool gauge, only per-pool KV ratios
        draw_bar_row(stdscr, y, x, "KV usage (max):", kv_ratio, width, kv_color)
        y += 1
        per = "  ".join(f"{p}:{v * 100:.1f}%" for p, v in ratios) or "n/a"
        draw_row(stdscr, y, x, "KV per pool:", per, width)
        y += 1
        draw_row(stdscr, y, x, "Context window:", format_num(ctx) if ctx else "n/a", width)
        y += 1
    if spill:
        stored = cur["spill_stored_bytes"]
        entries = int(cur["spill_entries"])
        cap = cur.get("spill_gib") or spill_cap
        if cap:
            fill_pct = stored / (cap * 2 ** 30) * 100
            fill_color = COLOR_GREEN if fill_pct < 70 else COLOR_WARN if fill_pct < 90 else COLOR_RED
            draw_bar_row(stdscr, y, x, "Spill fill:", fill_pct, width, fill_color,
                         suffix=f"{stored / 2 ** 30:.1f}/{cap:g} GiB")
            y += 1
        # delta rates: None on engine restart (negative delta) or an idle window — honest gap
        w_rate = wall_rate_safe(cur, prev, "spill_written_bytes")
        l_rate = wall_rate_safe(cur, prev, "spill_loaded_tokens")
        if spill_hist is not None:
            spill_hist.append((cur["t"], l_rate))
            prune_rate_hist(spill_hist, 120.0)
        written = f"{cur['spill_written_bytes'] / 2 ** 30:.2f} GiB · {int(cur['spill_written'])} writes"
        if w_rate:
            written += f" (+{format_bytes(w_rate)}/s)"
        loaded = f"{cur['spill_loaded_bytes'] / 2 ** 30:.2f} GiB · {format_num(cur['spill_loaded_tokens'])} tok"
        if l_rate:
            loaded += f" (+{format_num(l_rate)}/s)"
        fails = (int(cur["spill_write_failed"]), int(cur["spill_crc_failed"]),
                 int(cur["spill_load_failed"]), int(cur["spill_load_unfit"]))
        fail_color = COLOR_GREEN if not any(fails) else COLOR_RED
        draw_two_col_row(stdscr, y, x, width, "RAM pool", "", "SSD spill tier (streaming)", "",
                         left_label_color=COLOR_HEADER, right_label_color=COLOR_HEADER)
        y += 1
        draw_two_col_row(stdscr, y, x, width, "Pool:",
                         f"{format_num(pool - free)} / {format_num(pool)} rows free {format_num(free)}",
                         "Stored:", f"{stored / 2 ** 30:.1f} GiB ({entries} entries)")
        y += 1
        draw_two_col_row(stdscr, y, x, width, "Kept prompts:", f"{kept} ({format_bytes(cur.get('kept_bytes', 0.0))})",
                         "Written:", written)
        y += 1
        draw_two_col_row(stdscr, y, x, width, "Stream KV ratio:", f"{kv_ratio:.1f}% (per-stream / ctx window)",
                         "Loaded:", loaded)
        y += 1
        draw_two_col_row(stdscr, y, x, width, "Context window:", format_num(ctx),
                         "Restore p50/p95:", f"{cur['spill_restore_ms_p50']:.1f} / {cur['spill_restore_ms_p95']:.1f} ms")
        y += 1
        draw_two_col_row(stdscr, y, x, width, "", "",
                         "Failures:", f"write {fails[0]} · crc {fails[1]} · load {fails[2]} · unfit {fails[3]}",
                         right_value_color=fail_color)
        y += 1
        draw_two_col_row(stdscr, y, x, width, "", "",
                         "Disk free:", format_bytes(cur["spill_disk_free_bytes"]))
        y += 1
        if spill_hist is not None and len(spill_hist) >= 2:
            pts = [r[1] for r in spill_hist]
            rmax = max((v for v in pts if v is not None), default=0.0)
            draw_row(stdscr, y, x, f"Restore 120s [{format_speed(rmax)}]:", f"[{sparkline(pts)}]", width)
            y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    # ---- Cache reuse (only when the engine reports it) ----
    reuse = cur.get("reuse") or {}
    if any(v is not None for k, v in reuse.items() if k != "mix") or reuse.get("mix"):
        draw_box_start(stdscr, y, x, width, "Cache reuse (cumulative)")
        y += 1
        kb = reuse.get("kept_bytes")
        cap = reuse.get("kept_bytes_cap")
        if kb is None:
            draw_row(stdscr, y, x, "Kept memory:", "n/a (not reported)", width, value_color=COLOR_DIM)
        else:
            val = format_bytes(kb) + (f"  cap {format_bytes(cap)}" if cap else "")
            draw_row(stdscr, y, x, "Kept memory:", val, width)
        y += 1
        mix = reuse.get("mix")
        if mix:
            draw_row(stdscr, y, x, "Kept mix:",
                     f"own {int(mix.get('own', 0))}  shared {int(mix.get('shared', 0))}  "
                     f"chats {int(mix.get('chats', 0))}  rows {format_num(mix.get('rows', 0))}", width)
        else:
            draw_row(stdscr, y, x, "Kept mix:", "n/a (no kept_mix in /health)", width, value_color=COLOR_DIM)
        y += 1
        pw, pwt = reuse.get("prefix_waits"), reuse.get("prefix_wait_tokens")
        if pw is None and pwt is None:
            draw_row(stdscr, y, x, "Prefix waits:", "n/a (not reported)", width, value_color=COLOR_DIM)
        else:
            # HELP: requests that waited for a prefix another request was prefilling;
            # the tokens are the prompt tokens they did not prefill again.
            draw_row(stdscr, y, x, "Prefix waits:",
                     f"{int(pw or 0)} waits  {format_num(pwt or 0)} tok not re-prefilled", width)
        y += 1
        cc, cct = reuse.get("chats_continued"), reuse.get("chats_continued_prefill_tokens")
        if cc is None and cct is None:
            draw_row(stdscr, y, x, "Chats continued:", "n/a (not reported)", width, value_color=COLOR_DIM)
        else:
            # HELP: streams continued without resuming from their own turn state
            draw_row(stdscr, y, x, "Chats continued:",
                     f"{int(cc or 0)} streams  {format_num(cct or 0)} tok prefilled", width)
        y += 1
        draw_box_end(stdscr, y, x, width)
        y += 1

    # ---- Averages (cumulative histograms) ----
    lat_sum = cur["tensorfold:request_latency_seconds_sum"]
    lat_n = cur["tensorfold:request_latency_seconds_count"]
    ttft_sum = cur["tensorfold:time_to_first_token_seconds_sum"]
    ttft_n = cur["tensorfold:time_to_first_token_seconds_count"]
    avg_lat = lat_sum / lat_n if lat_n else None
    avg_ttft = ttft_sum / ttft_n if ttft_n else None
    drafted = cur["drafted_total"]
    accepted = cur["accepted_total"]
    acc_rate = accepted / drafted * 100 if drafted else 0.0
    ptok = cur["prompt_tokens_total"]
    ctok = cur["cached_tokens_total"]
    hit = ctok / ptok * 100 if ptok else 0.0
    cache_known = cur.get("cache_known", True)

    draw_box_start(stdscr, y, x, width, "Latency (cumulative)")
    y += 1
    lat_q = _quantile_strs(cur.get("lat_buckets"), lat_n)
    ttft_q = _quantile_strs(cur.get("ttft_buckets"), ttft_n)
    draw_row(stdscr, y, x, "Latency:",
             f"avg {format_latency(avg_lat)}   p50 {lat_q[0]}  "
             f"p90 {lat_q[1]}  p99 {lat_q[2]}  (n={int(lat_n)})", width)
    y += 1
    draw_row(stdscr, y, x, "TTFT:",
             f"avg {format_latency(avg_ttft)}   p50 {ttft_q[0]}  "
             f"p90 {ttft_q[1]}  p99 {ttft_q[2]}  (n={int(ttft_n)})", width)
    y += 1
    acc_color = COLOR_GREEN if acc_rate >= 70 else COLOR_WARN if acc_rate >= 50 else COLOR_RED
    if not drafted:
        acc_color = COLOR_DIM
    draw_row(stdscr, y, x, "MTP Acceptance:", f"{acc_rate:.1f}%  ({format_num(accepted)}/{format_num(drafted)})", width, value_color=acc_color)
    y += 1
    if cache_known:
        hit_color = COLOR_GREEN if hit >= 70 else COLOR_WARN
        draw_row(stdscr, y, x, "Cache Hit (tokens):", f"{hit:.1f}%  ({format_num(ctok)}/{format_num(ptok)})", width, value_color=hit_color)
    else:
        draw_row(stdscr, y, x, "Cache Hit (tokens):", "n/a (server exposes no cached-token metric)", width, value_color=COLOR_DIM)
    y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    # ---- Totals ----
    draw_box_start(stdscr, y, x, width, "Totals")
    y += 1
    draw_row(stdscr, y, x, "Requests:", f"{int(cur['requests_total'])}  (running {int(cur['requests_running'])}, waiting {int(cur['requests_waiting'])})", width)
    y += 1
    draw_row(stdscr, y, x, "Prompt / Completion:", f"{format_num(cur['prompt_tokens_total'])} / {format_num(cur['completion_tokens_total'])}", width)
    y += 1
    draw_row(stdscr, y, x, "Engine time:", f"prefill {cur['prefill_seconds_total']:.0f}s  decode {cur['decode_seconds_total']:.0f}s  rounds {int(cur['rounds_total'])}", width)
    y += 1
    mp = cur["multi_prefill"]
    if mp:
        draw_row(stdscr, y, x, "Multi-prefill:", f"chunks {int(mp.get('chunks', 0))}  pieces {int(mp.get('pieces', 0))}  rows {format_num(mp.get('rows', 0))}", width)
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
