"""Forced alignment between a (punctuated) script and Whisper word timings.

Implements P1 + P2 from PATCH-LIST.md. The script has authoritative
punctuation. Whisper has authoritative timings. We align tokens with a
phonetic-relaxed match so EdgeTTS oddities, ASR drift, and unit-vs-numeral
forms ("185°F" vs "one hundred eighty five") don't break sentence boundaries.
"""
from __future__ import annotations
import re, json, hashlib
from dataclasses import dataclass, field
from pathlib import Path
import pysbd
from rapidfuzz import fuzz

_seg = pysbd.Segmenter(language="en", clean=False)

_NUM_WORD = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen",
    "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen",
    "19": "nineteen", "20": "twenty", "30": "thirty", "40": "forty",
    "50": "fifty", "60": "sixty", "70": "seventy", "80": "eighty", "90": "ninety",
    "100": "hundred", "1000": "thousand",
}

# Abbreviations that should NOT end a sentence even when followed by space + cap.
# pysbd handles most; this is belt-and-braces for our domains.
_ABBR_NOEND = {"oz", "tsp", "tbsp", "ml", "fl", "g", "kg", "mr", "mrs", "ms",
               "dr", "vs", "st", "rd", "ave", "fig", "no", "vol", "ch"}


def split_into_sentences(text: str) -> list[str]:
    """Real sentence splitter that handles decimals + abbreviations.

    Replaces the naive regex in `tts.py::split_into_sentences`. F1 from
    RED-TEAM-algorithm.md.
    """
    sents = [s.strip() for s in _seg.segment(text) if s.strip()]
    # pysbd already handles abbreviations + decimals; this is fine.
    return sents


def _normalize(tok: str) -> str:
    """Loose normalization for fuzzy alignment.

    - lowercase
    - strip punctuation
    - expand single-digit numerals (185 -> "one hundred eighty five" approx)
      via partial-ratio rather than exact rewrite.
    """
    t = tok.lower()
    t = re.sub(r"[^a-z0-9°]", "", t)
    return t


def _expand_number(tok: str) -> str:
    """Cheap numeral-to-words approximation. Good enough for partial-ratio match."""
    if not re.fullmatch(r"\d+", tok):
        return tok
    n = int(tok)
    if str(n) in _NUM_WORD:
        return _NUM_WORD[str(n)]
    if n < 100:
        tens = (n // 10) * 10
        ones = n % 10
        if tens and str(tens) in _NUM_WORD:
            base = _NUM_WORD[str(tens)]
            return f"{base} {_NUM_WORD[str(ones)]}" if ones else base
    if n < 1000:
        h = n // 100
        rest = n % 100
        out = f"{_NUM_WORD.get(str(h), str(h))} hundred"
        if rest:
            out += " " + _expand_number(str(rest))
        return out
    # Large numbers — return as-is, partial-ratio handles "one hundred eighty-five"
    return tok


@dataclass
class AlignedWord:
    text: str             # the script token (with original punctuation kept on word_punct)
    word_punct: str       # token as it appears in the script (e.g. "185°F.")
    sent_idx: int         # which sentence this word belongs to
    para_idx: int         # which paragraph
    start: float | None = None   # filled from Whisper
    end: float | None = None
    confidence: float = 0.0      # 0..1 alignment confidence


@dataclass
class WhisperWord:
    text: str
    start: float
    end: float


def transcribe_words(audio_path: Path, model_size: str = "tiny.en") -> list[WhisperWord]:
    """Run faster-whisper to get word-level timestamps. Cached on disk.

    If faster-whisper is unavailable (e.g. ctranslate2 can't install on a read-only
    mount), return [] so the caller falls back to known-text forced alignment
    (`synthesize_word_timings`) instead of crashing the plan phase (Rule v17.4)."""
    cache = audio_path.with_suffix(audio_path.suffix + f".whisper.{model_size}.json")
    if cache.exists() and cache.stat().st_size > 100:
        data = json.loads(cache.read_text())
        return [WhisperWord(**w) for w in data]
    try:
        from faster_whisper import WhisperModel
    except Exception:
        return []
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(audio_path), word_timestamps=True,
                                    vad_filter=True, beam_size=1)
    out: list[WhisperWord] = []
    for seg in segments:
        for w in seg.words or []:
            out.append(WhisperWord(text=w.word.strip(), start=float(w.start), end=float(w.end)))
    cache.write_text(json.dumps([w.__dict__ for w in out]))
    return out


def synthesize_word_timings(text: str, duration: float) -> list[WhisperWord]:
    """Known-text forced-alignment fallback for when ASR (faster-whisper) is unavailable.

    We already know the exact script text and the measured audio duration (ffprobe), so we
    spread the script's words across [0, duration] weighted by word length (longer words
    take longer to say). Feeding these back through `align()` matches each script token to
    its synthetic word, so sentence boundaries land at sensible, monotonic timestamps —
    good enough for per-sentence shot sizing (Rule v17.4). It is not ASR-accurate, just a
    graceful degradation so the pipeline never hard-stops on a missing ASR dependency."""
    toks: list[str] = []
    for sent in split_into_sentences(text):
        toks.extend(sent.split())
    if not toks:
        toks = (text or "").split()
    if not toks or duration <= 0:
        return []
    weights = [max(1, len(re.sub(r"[^A-Za-z0-9]", "", t))) for t in toks]
    total = float(sum(weights)) or float(len(toks))
    out: list[WhisperWord] = []
    t = 0.0
    for tok, wt in zip(toks, weights):
        share = duration * (wt / total)
        start = t
        end = min(duration, t + share)
        out.append(WhisperWord(text=tok, start=round(start, 3), end=round(end, 3)))
        t = end
    return out


def align(script_paragraphs: list[str], whisper_words: list[WhisperWord]) -> list[AlignedWord]:
    """Align script tokens (authoritative punctuation) to Whisper words (authoritative timings).

    Greedy with fuzzy match — much simpler than Needleman-Wunsch and adequate
    because EdgeTTS output is close to the input text. Returns one AlignedWord
    per script token with timings filled in from whisper. Unmatched tokens
    interpolate timings from neighbors and get confidence=0.
    """
    # Tokenize each paragraph into sentences then words
    script_tokens: list[AlignedWord] = []
    for pi, para in enumerate(script_paragraphs):
        sents = split_into_sentences(para)
        for si, sent in enumerate(sents):
            for tok in sent.split():
                script_tokens.append(AlignedWord(
                    text=_normalize(tok), word_punct=tok, sent_idx=si, para_idx=pi,
                ))

    wi = 0
    n_w = len(whisper_words)
    for st in script_tokens:
        if wi >= n_w:
            break
        # Try direct fuzzy match with a small lookahead window
        best_j, best_score = -1, 0
        for j in range(wi, min(wi + 6, n_w)):
            wt = _normalize(whisper_words[j].text)
            # Try direct + expanded-number
            score = max(fuzz.ratio(st.text, wt),
                        fuzz.partial_ratio(_expand_number(st.text), wt) // 1.2)
            if score > best_score:
                best_j, best_score = j, score
        if best_score >= 70 and best_j >= 0:
            st.start = whisper_words[best_j].start
            st.end = whisper_words[best_j].end
            st.confidence = best_score / 100.0
            wi = best_j + 1
        # Unmatched: leave start/end None; we'll interpolate at the end.

    # Interpolate missing timings from nearest neighbors
    for i, st in enumerate(script_tokens):
        if st.start is not None:
            continue
        # Find nearest left + right matched tokens
        l, r = None, None
        for k in range(i - 1, -1, -1):
            if script_tokens[k].start is not None:
                l = script_tokens[k]; break
        for k in range(i + 1, len(script_tokens)):
            if script_tokens[k].start is not None:
                r = script_tokens[k]; break
        if l and r:
            # Linear interpolation
            n_gap = 1
            for k in range(i - 1, -1, -1):
                if script_tokens[k] is l: break
                n_gap += 1
            for k in range(i + 1, len(script_tokens)):
                if script_tokens[k] is r: break
                n_gap += 1
            ratio = 1.0  # rough; we just put it midway
            st.start = (l.end + r.start) / 2
            st.end = st.start + 0.2
            st.confidence = 0.2
        elif l:
            st.start = (l.end or 0) + 0.15
            st.end = st.start + 0.2
            st.confidence = 0.1
        elif r:
            st.start = max(0, (r.start or 0) - 0.15)
            st.end = st.start + 0.2
            st.confidence = 0.1
    return script_tokens


def sentence_boundaries(aligned: list[AlignedWord]) -> list[tuple[int, int, float, float, str]]:
    """Return list of (para_idx, sent_idx, start_sec, end_sec, sentence_text)."""
    out = []
    cur = None
    cur_words: list[AlignedWord] = []
    for w in aligned:
        key = (w.para_idx, w.sent_idx)
        if cur is None or key != cur:
            if cur is not None and cur_words:
                pi, si = cur
                txt = " ".join(x.word_punct for x in cur_words)
                start = next((x.start for x in cur_words if x.start is not None), 0.0) or 0.0
                end = next((x.end for x in reversed(cur_words) if x.end is not None), start) or start
                out.append((pi, si, start, end, txt))
            cur = key
            cur_words = [w]
        else:
            cur_words.append(w)
    if cur is not None and cur_words:
        pi, si = cur
        txt = " ".join(x.word_punct for x in cur_words)
        start = next((x.start for x in cur_words if x.start is not None), 0.0) or 0.0
        end = next((x.end for x in reversed(cur_words) if x.end is not None), start) or start
        out.append((pi, si, start, end, txt))
    return out
