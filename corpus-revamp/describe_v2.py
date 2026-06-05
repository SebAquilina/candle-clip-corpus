"""describe_v2.py — second-by-second, context-aware clip understanding via Claude vision-agents.

Pipeline (resumable, runs on the pre-downloaded clip cache so nothing re-downloads):
  extract  : 1 frame/sec from each cached window -> outputs/describe_v2/frames/<key>/sec_NN.jpg
  worklist : emit the windows still needing description (+ frame paths, transcript, title)
  merge    : fold the agents' per-window JSON (outputs/describe_v2/desc/<key>.json) into the
             corpus records as ADDITIVE v2 fields (keeps BLIP + transcript intact).

A "team of agents" is given slices of the worklist; each agent READS a window's ordered frames
(so it has the whole action arc as context), then writes, PER SECOND, a context-aware
description, plus a structured window rollup. Schema is in worklist()'s instructions.

key = "<video_id>_<start_s:.2f>"  (matches outputs/clip_cache/<key>.mp4)
"""
import os, sys, json, glob, subprocess, math
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CACHE = os.path.join(ROOT, "outputs", "clip_cache")
DV2   = os.path.join(ROOT, "outputs", "describe_v2")
FRAMES= os.path.join(DV2, "frames")
DESC  = os.path.join(DV2, "desc")
RECS  = os.path.join(ROOT, "outputs", "shared_db_v2", "records")
for d in (FRAMES, DESC): os.makedirs(d, exist_ok=True)


def _windows():
    out = []
    for p in sorted(glob.glob(os.path.join(RECS, "*.json"))):
        r = json.load(open(p))
        for w in r.get("windows", []):
            s = float(w["start_s"])
            key = f"{r['video_id']}_{s:.2f}"
            out.append({"key": key, "video_id": r["video_id"], "video_url": r.get("video_url",""),
                        "video_title": r.get("video_title",""), "channel": r.get("channel",""),
                        "niche": r.get("niche",""), "start_s": s, "end_s": float(w["end_s"]),
                        "action_label": w.get("action_label",""),
                        "transcript": (w.get("transcript") or "").strip(),
                        "blip": (w.get("embed_text") or "").strip(),
                        "clip": os.path.join(CACHE, f"{key}.mp4"),
                        "rec": p, "window_index": w.get("window_index")})
    return out


def _probe(p):
    try:
        return float(subprocess.check_output(
            ["ffprobe","-v","error","-show_entries","format=duration","-of",
             "default=noprint_wrappers=1:nokey=1", p], stderr=subprocess.DEVNULL, timeout=20).decode().strip())
    except Exception:
        return 0.0


def extract():
    ws = _windows(); done = n = 0
    for w in ws:
        if not os.path.exists(w["clip"]):
            continue
        fdir = os.path.join(FRAMES, w["key"])
        if os.path.isdir(fdir) and glob.glob(os.path.join(fdir,"sec_*.jpg")):
            done += 1; continue
        os.makedirs(fdir, exist_ok=True)
        dur = min(_probe(w["clip"]), w["end_s"]-w["start_s"]+1.0)
        nsec = max(1, int(math.floor(dur)))
        for t in range(nsec):
            out = os.path.join(fdir, f"sec_{t:02d}.jpg")
            subprocess.run(["ffmpeg","-y","-ss",f"{t+0.5:.2f}","-i",w["clip"],"-frames:v","1",
                            "-vf","scale=640:-2","-q:v","4", out],
                           stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, timeout=30)
        n += 1
        if n % 20 == 0: print(f"  extracted {n} windows ...", flush=True)
    print(f"[extract] frames present for {done+n} windows ({n} new); clips cached: "
          f"{sum(1 for w in ws if os.path.exists(w['clip']))}/{len(ws)}", flush=True)


def worklist():
    ws = _windows(); todo = []
    for w in ws:
        fdir = os.path.join(FRAMES, w["key"])
        frames = sorted(glob.glob(os.path.join(fdir, "sec_*.jpg")))
        if not frames: continue
        if os.path.exists(os.path.join(DESC, w["key"]+".json")): continue
        todo.append({**{k:w[k] for k in ("key","video_id","video_title","niche","action_label",
                                          "transcript","blip","start_s","end_s")},
                     "frames": frames})
    json.dump(todo, open(os.path.join(DV2,"worklist.json"),"w"), indent=2)
    print(f"[worklist] {len(todo)} windows need description -> {DV2}/worklist.json "
          f"(of {len(ws)} total; {len(ws)-len(todo)} done/no-frames)", flush=True)
    return todo


def status():
    ws=_windows()
    clips=sum(1 for w in ws if os.path.exists(w['clip']))
    framed=sum(1 for w in ws if glob.glob(os.path.join(FRAMES,w['key'],"sec_*.jpg")))
    described=len(glob.glob(os.path.join(DESC,"*.json")))
    print(f"windows {len(ws)} | clips cached {clips} | frames extracted {framed} | described {described}")


def merge():
    by_rec = {}
    for d in glob.glob(os.path.join(DESC,"*.json")):
        try: j=json.load(open(d))
        except Exception: continue
        by_rec.setdefault(j.get("rec") or "", []).append(j)
    # group by record path via the windows() map (desc files only carry key)
    keymap={w["key"]:w for w in _windows()}
    updates={}
    for d in glob.glob(os.path.join(DESC,"*.json")):
        try: j=json.load(open(d))
        except Exception: continue
        k=j.get("key");  w=keymap.get(k)
        if not w: continue
        updates.setdefault(w["rec"], {})[w["window_index"]]=j
    n=0
    for recpath, wins in updates.items():
        r=json.load(open(recpath)); changed=False
        for w in r.get("windows",[]):
            j=wins.get(w.get("window_index"))
            if not j: continue
            w["embed_text_v2"]=j.get("embed_text") or j.get("summary") or w.get("embed_text")
            w["summary_v2"]=j.get("summary","")
            w["tags_v2"]=j.get("tags",{})
            w["seconds_v2"]=j.get("seconds",[])
            w["qc_v2"]=j.get("qc_flags",{})
            changed=True
        if changed:
            json.dump(r, open(recpath,"w")); n+=1
    print(f"[merge] updated {n} records with v2 descriptions", flush=True)


if __name__=="__main__":
    cmd = sys.argv[1] if len(sys.argv)>1 else "status"
    {"extract":extract,"worklist":worklist,"merge":merge,"status":status}.get(cmd, status)()
