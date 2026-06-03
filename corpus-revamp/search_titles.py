import yt_dlp, json, re, glob, os, sys
TITLES=json.load(open("/tmp/workbook_titles.json"))
# existing corpus ids (dedupe target): old corpus + v2 records + v2 rejected
existing=set()
for f in glob.glob("/home/user/candle-clip-corpus/outputs/shared_db/records/*.json")+\
         glob.glob("/home/user/candle-clip-corpus/outputs/shared_db_v2/records/*.json")+\
         glob.glob("/home/user/candle-clip-corpus/outputs/shared_db_v2/rejected/*.json"):
    try: existing.add(json.load(open(f)).get("video_id"))
    except: pass
_STRIP=re.compile(r"\b(i|tested|tried|melted|burned|burning|almost|all|of|them|are|is|lies|here'?s|what|went|wrong|stop|buying|can|you|make|made|for|hours|straight|every|my|house|together|the|a|an|in|with|—|vs)\b", re.I)
def keywords(title):
    t=re.split(r"[—:?\-]", title)[0]            # topic is usually before the dash/colon/?
    t=_STRIP.sub(" ", t)
    t=re.sub(r"[^A-Za-z0-9 ]"," ",t); t=re.sub(r"\s+"," ",t).strip()
    if "candle" not in t.lower() and "wax" not in t.lower(): t+=" candle"
    return (t+" candle making").strip()[:80]
def search(q, n=3):
    opts={"quiet":True,"no_warnings":True,"extract_flat":True,"nocheckcertificate":True,
          "cookiefile":"/tmp/revamp_cookies.txt","extractor_args":{"youtube":{"player_client":["mweb","web"]}}}
    with yt_dlp.YoutubeDL(opts) as y:
        info=y.extract_info(f"ytsearch{n}:{q}", download=False)
    return [e.get("id") for e in (info.get("entries") or []) if e.get("id")]
LIMIT=int(sys.argv[1]) if len(sys.argv)>1 else len(TITLES)
seen=set(existing); out=[]
for i,title in enumerate(TITLES[:LIMIT]):
    q=keywords(title)
    try: ids=search(q,3)
    except Exception as e: print(f"  [searchfail] {title[:40]}: {str(e)[:50]}"); continue
    new=[v for v in ids if v not in seen]
    for v in new: seen.add(v); out.append(v)
    print(f"  '{q}' -> {ids} (new: {len(new)})")
json.dump(out, open("/tmp/search_ids.json","w"))
print(f"\n{len(out)} new deduped ids from {LIMIT} titles ({len(existing)} already in corpus)")
