"""apply_qc_v2.py — turn describe_v2's qc_v2 flags into a soft cleanup signal.

For every window with qc_v2.person_or_face / on_screen_text / off_topic = true, write
window.usable_v2 = false (with a `usable_v2_reason`). This is ADDITIVE and REVERSIBLE
(no field deletion). The video-maker's shared_library will then skip these as if they
were `talking_head` flagged. Idempotent: re-running is a no-op when verdicts unchanged.

Usage: python apply_qc_v2.py            # apply
       python apply_qc_v2.py --status   # show usable vs flagged tallies
"""
import os, sys, json, glob

HERE = os.path.dirname(os.path.abspath(__file__))
RECS = os.path.abspath(os.path.join(HERE, "..", "outputs", "shared_db_v2", "records"))


def _flag_reasons(qc):
    r = []
    if qc.get("person_or_face"): r.append("person_or_face")
    if qc.get("on_screen_text"): r.append("on_screen_text")
    if qc.get("off_topic"):      r.append("off_topic")
    return r


def apply():
    rec_paths = sorted(glob.glob(os.path.join(RECS, "*.json")))
    rec_changed = win_flagged = win_unflagged = win_total = win_usable = 0
    for p in rec_paths:
        try:
            r = json.load(open(p))
        except Exception:
            continue
        wins = r.get("windows") or []
        changed = False
        for w in wins:
            win_total += 1
            qc = w.get("qc_v2") or {}
            reasons = _flag_reasons(qc) if qc else []
            had = "usable_v2" in w
            if reasons:
                if w.get("usable_v2") is not False or w.get("usable_v2_reason") != ",".join(reasons):
                    w["usable_v2"] = False
                    w["usable_v2_reason"] = ",".join(reasons)
                    changed = True; win_flagged += 1
            else:
                # explicitly mark on-topic clean windows so the matcher knows v2 has spoken
                if "qc_v2" in w and not had:
                    w["usable_v2"] = True
                    changed = True; win_unflagged += 1
            if w.get("usable_v2", True):
                win_usable += 1
        if changed:
            json.dump(r, open(p, "w"))
            rec_changed += 1
    print(f"[apply] records changed: {rec_changed}/{len(rec_paths)}")
    print(f"[apply] windows total: {win_total} | now flagged: {win_flagged} | now marked clean: {win_unflagged}")
    print(f"[apply] USABLE windows after cleanup: {win_usable}")


def status():
    rec_paths = sorted(glob.glob(os.path.join(RECS, "*.json")))
    total = usable = flagged = no_v2 = 0
    by_reason = {}
    for p in rec_paths:
        try: r = json.load(open(p))
        except Exception: continue
        for w in r.get("windows", []):
            total += 1
            if "qc_v2" not in w:
                no_v2 += 1; usable += 1; continue
            if w.get("usable_v2", True):
                usable += 1
            else:
                flagged += 1
                for tag in (w.get("usable_v2_reason") or "").split(","):
                    by_reason[tag] = by_reason.get(tag, 0) + 1
    print(f"total windows: {total} | USABLE: {usable} | flagged unusable: {flagged} | no qc_v2 yet: {no_v2}")
    for k, v in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    {"status": status, "apply": apply}.get(
        sys.argv[1].lstrip("-") if len(sys.argv) > 1 else "apply", apply)()
