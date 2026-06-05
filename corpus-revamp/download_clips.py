"""download_clips.py — pre-download every clean corpus window into a persistent local cache
(outputs/clip_cache/<vid>_<start>.mp4), so the second-by-second re-describe (and the
video-maker) never re-download. Resumable: skips windows already cached. Paced + retry-aware
to survive YouTube rate-limits. Cookies via REVAMP_COOKIES / VM_COOKIES.

Usage: python download_clips.py            (grind all windows)
       python download_clips.py --status   (how many cached vs total)
"""
import os, sys, json, glob, time
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "video-maker-3", "scripts"))
os.environ.setdefault("WS", os.path.join(HERE, "..", "video-maker-3"))
from pathlib import Path
CACHE=Path(HERE)/".."/"outputs"/"clip_cache"; CACHE.mkdir(parents=True, exist_ok=True)
PAD=float(os.environ.get("CLIP_PAD","1.0"))
GAP=float(os.environ.get("CLIP_GAP_SEC","2.0"))   # pause between downloads (be gentle on rate limits)

def windows():
    out=[]
    for p in sorted(glob.glob(os.path.join(HERE,"..","outputs","shared_db_v2","records","*.json"))):
        r=json.load(open(p))
        for w in r.get("windows",[]):
            out.append((r["video_id"], r["video_url"], float(w["start_s"]), float(w["end_s"])))
    return out

def cache_path(vid,start): return CACHE/f"{vid}_{start:.2f}.mp4"

def main():
    ws=windows()
    if "--status" in sys.argv:
        have=sum(1 for v,u,s,e in ws if cache_path(v,s).exists())
        print(f"cached {have}/{len(ws)} windows in {CACHE}"); return
    from app.services import youtube
    todo=[(v,u,s,e) for v,u,s,e in ws if not cache_path(v,s).exists()]
    print(f"[clips] {len(ws)} windows, {len(ws)-len(todo)} cached, {len(todo)} to download", flush=True)
    ok=fail=0
    for i,(v,u,s,e) in enumerate(todo,1):
        out=cache_path(v,s)
        got=youtube.download_segment(u, s, e+PAD, out)
        if got and out.exists(): ok+=1
        else: fail+=1
        if i%5==0 or i==len(todo):
            print(f"[clips] {i}/{len(todo)} | ok {ok} fail {fail} | last {v}@{s:.0f} {'OK' if got else 'FAIL'}", flush=True)
        time.sleep(GAP)
    print(f"[clips] DONE: {ok} downloaded, {fail} failed, cache now {sum(1 for v,u,s,e in ws if cache_path(v,s).exists())}/{len(ws)}", flush=True)

if __name__=="__main__": main()
