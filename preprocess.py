#!/usr/bin/env python3
"""
preprocess.py -- Ilocano dictionary preprocessing + culinary-term extraction.

Run:  python preprocess.py            (Python 3, standard library only)
Opts: --input PATH   path to ilocano_dictionary.json (default: auto-locate)
      --outdir DIR   output folder (default: ./output next to this script)

Pipeline (execution order; steps 3/4 are preceded by the step-5 safety check):
  1 culinary filter -> 2 invert -> 5 tricky check -> 3 c->k -> 4 j->h ->
  6 split phrases -> 7 normalise word types -> 8 de-duplicate -> 9 save.
Deterministic: no randomness except a fixed-seed sample for the console preview.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
import unicodedata
import urllib.request
from collections import Counter, OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
FILENAME = "ilocano_dictionary.json"
RAW_URL = ("https://github.com/luisligunas/pinoy-dictionary-scraper/raw/refs/heads/"
           "main/Scraped%20Data/Dictionaries/ilocano_dictionary.json")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
CORE_KEYWORDS = """food eat cook boil fry grill roast bake steam stew soup broth sauce spice salt
sugar vinegar oil rice grain corn meat pork beef chicken goat fish shrimp shellfish seafood egg
vegetable fruit bean legume squash eggplant|bitter melon|tomato onion garlic ginger chili pepper
banana coconut tamarind ferment dried smoked pickled raw bitter sour sweet salty flavor taste dish
meal breakfast lunch dinner snack dessert cake bread drink beverage wine coffee tea kitchen pot pan
bowl plate spoon fork knife ladle stove dinengdeng pinakbet bagnet pinapaitan"""
# the multi-word keyword "bitter melon" is protected with '|' above
CORE_KEYWORDS = [k.replace("_", " ") for k in
                 re.sub(r"\|bitter melon\|", " bitter_melon ", CORE_KEYWORDS).split()]

# Assumption: inflected forms of core keywords are needed because matching is whole-word.
INFLECTED_KEYWORDS = """eating eaten ate eats cooks cooked cooking cooker boils boiled boiling fries fried
frying grilled grilling roasted roasting baked baking steamed steaming stewed spices spiced spicy
salted sugars oils eggs beans fruits vegetables onions peppers bananas cakes drinks drinking
dishes meals snacks breads pots pans bowls plates spoons forks knives tomatoes chilies
coconuts dried smoked pickled fermented grains meats""".split()

# Assumption: a few obvious food words missing from the user's list (tagged "extra" in the log).
EXTRA_KEYWORDS = """pumpkin potato cabbage mango pineapple lemon grape melon cucumber lettuce salad
ham bacon sausage crab clam oyster duck milk cheese butter flour honey juice beer restaurant oven
kettle tasty hungry thirsty feast bakery baker cuisine appetite""".split()

# Ilocano dish names can only occur on the Ilocano side, so they are matched there as well.
ILOCANO_DISH_KEYWORDS = ["dinengdeng", "pinakbet", "bagnet", "pinapaitan"]

# (english-headword regex, optional definition regex, reason)
EXCLUSIONS = [
    (r"\bsmall[- ]fry\b", None, "'small fry' = insignificant people / young fish"),
    (r"\bpepper[- ]spray\b", None, "'pepper spray' = self-defence weapon"),
    (r"\bsalt (?:away|of the earth)\b|\bold salt\b|\bworth one'?s salt\b", None,
     "'salt' used metaphorically (idiom)"),
    (r"\bchicken flea\b", None, "'chicken flea' = a parasite, not food"),
    (r"\bchicken[- ]hearted\b|\bchicken[- ]out\b", None, "'chicken' = cowardly (idiom)"),
    (r"\bmelting pot\b", None, "'melting pot' = cultural mix, not a cooking pot"),
    (r"\bpot[- ]shot\b|\bpot[- ]belly\b|\bpot ?luck\b", None, "'pot' used non-literally"),
    (r"\bflash in the pan\b|\bpan out\b", None, "'pan' used in an idiom"),
    (r"\bsour grapes\b|\bbitter end\b|\braw deal\b|\bsweet[- ]talk\b", None,
     "idiom, not a culinary sense"),
    (r"^boil$", r"^n\b", "'boil' (noun) = skin abscess (Ilocano 'letteg'), not cooking"),
    (r"^clam$", r"carpentero", "source definition describes a carpenter's clamp (headword truncated), not a shellfish"),
]

# Part-of-speech variant -> clean label.  Keys are lower-case with dots stripped.
POS_LABEL = OrderedDict([
    ("n", "noun"), ("noun", "noun"),
    ("v", "verb"), ("vb", "verb"), ("verb", "verb"),
    ("adj", "adjective"), ("adjective", "adjective"), ("ajd", "adjective"), ("dj", "adjective"),
    ("adv", "adverb"), ("adverb", "adverb"), ("ad", "adverb"),
    ("pron", "pronoun"), ("pronoun", "pronoun"),
    ("prep", "preposition"), ("preposition", "preposition"),
    ("conj", "conjunction"), ("conjunction", "conjunction"),
    ("inter", "interjection"), ("interj", "interjection"), ("interjection", "interjection"),
    ("int", "interjection"),
    ("art", "article"), ("article", "article"),
    ("prefix", "prefix"),
])

# Step 5: Spanish-looking patterns (checked BEFORE c->k / j->h).
TRICKY_PATTERNS = [
    ("ch", re.compile(r"ch", re.I), "contains 'ch' (Spanish digraph)"),
    ("qu", re.compile(r"qu", re.I), "contains 'qu' (Spanish/old-orthography k-sound)"),
    ("ll", re.compile(r"ll", re.I), "contains 'll' (Spanish digraph)"),
    ("n-tilde", re.compile(r"ñ", re.I), "contains 'ñ' (Spanish letter)"),
    ("z", re.compile(r"z", re.I), "contains 'z' (Spanish loan letter)"),
    ("x", re.compile(r"x", re.I), "contains 'x' (Spanish/English loan letter)"),
    ("soft-c", re.compile(r"c(?=[eiy])", re.I), "soft 'c' before e/i/y (s-sound; c->k would be wrong)"),
    ("gu+e/i", re.compile(r"gu(?=[ei])", re.I), "contains 'gu' before e/i (Spanish silent-u spelling)"),
    ("ck", re.compile(r"ck", re.I), "contains 'ck' (English loan; c->k would give 'kk')"),
    ("accent", re.compile(r"[áéíóúüÁÉÍÓÚÜ]"), "contains accented vowel (Spanish loan)"),
]

# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def clean(s):
    """NFC-normalise, collapse whitespace, strip. Never raises on None/non-str."""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def kw_regex(kw):
    parts = kw.split()
    body = r"[\s-]+".join(re.escape(p) for p in parts)
    return re.compile(r"(?<![A-Za-z])" + body + r"(?![A-Za-z])", re.I)


class Log:
    def __init__(self):
        self.lines, self.steps = [], []

    def add(self, msg=""):
        self.lines.append(msg)

    def h(self, title):
        self.lines += ["", "=" * 78, title, "=" * 78]

    def step(self, step, unit, before, after, note=""):
        self.steps.append((step, unit, before, after, note))
        self.add("[STEP %s] %s: before=%s  after=%s  %s" % (step, unit, before, after, note))


# ----------------------------------------------------------------------------
# Loading + schema detection
# ----------------------------------------------------------------------------
def locate_input(arg, log):
    cands = ([arg] if arg else []) + [
        os.path.join(os.getcwd(), FILENAME), os.path.join(HERE, FILENAME),
        os.path.join(os.path.dirname(HERE), FILENAME),
        os.path.join("/mnt/user-data/uploads", FILENAME)]
    for c in cands:
        if c and os.path.isfile(c):
            log.add("Input located at: %s" % os.path.abspath(c))
            return os.path.abspath(c)
    log.add("No local copy found; trying to download %s" % RAW_URL)
    dest = os.path.join(HERE, FILENAME)
    try:
        with urllib.request.urlopen(RAW_URL, timeout=60) as r, open(dest, "wb") as out:
            shutil.copyfileobj(r, out)
        log.add("Downloaded to %s" % dest)
        return dest
    except Exception as exc:  # noqa
        sys.exit("ERROR: could not find or download %s (%s). Pass --input PATH." % (FILENAME, exc))


POS_START = re.compile(r"^\s*(?:n|v|adj|adv|pron|prep|conj|inter|interj|interjection|art|prefix|"
                       r"noun|verb|adjective|adverb|ad|dj|ajd)\b\.?", re.I)
ILO_MARKERS = {"ti", "iti", "nga", "ken", "dagiti", "ket", "ni", "ngem", "kas", "saan", "ag",
               "nag", "na", "dagita", "daytoy", "dayta", "a", "no", "ta"}
EN_STOP = {"the", "of", "and", "to", "in", "is", "for", "with", "that", "it", "as", "on", "be",
           "or", "an", "by", "at", "from", "this", "who", "which"}


def to_records(data, log):
    """Return a list of dict records from whatever top-level shape was loaded."""
    if isinstance(data, list):
        recs = [r for r in data if isinstance(r, dict)]
        skipped = len(data) - len(recs)
        if skipped:
            log.add("ASSUMPTION: %d non-object list items skipped." % skipped)
        return recs
    if isinstance(data, dict):
        lists = [(k, v) for k, v in data.items() if isinstance(v, list)]
        if len(lists) == 1:
            log.add("ASSUMPTION: records taken from top-level key %r." % lists[0][0])
            return to_records(lists[0][1], log)
        log.add("ASSUMPTION: top-level object treated as {word: definition} mapping.")
        return [{"word": k, "definition": v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}
                for k, v in data.items()]
    sys.exit("ERROR: unsupported top-level JSON type %s" % type(data).__name__)


def detect_schema(recs, log):
    cols = OrderedDict()
    for r in recs:
        for k, v in r.items():
            cols.setdefault(k, []).append(v)
    stats = {}
    for k, vals in cols.items():
        sv = [clean(v) for v in vals if isinstance(v, str) and clean(v)]
        n = max(len(sv), 1)
        toks = [t.lower() for v in sv for t in re.findall(r"[A-Za-zñÑ'-]+", v)]
        stats[k] = dict(
            filled=len(sv), pos_rate=sum(1 for v in sv if POS_START.match(v)) / n,
            url_rate=sum(1 for v in sv if re.match(r"https?://", v)) / n,
            distinct=len(set(sv)) / n, ndistinct=len(set(sv)),
            mean_len=sum(map(len, sv)) / n, mean_tok=sum(len(v.split()) for v in sv) / n,
            ilo=sum(t in ILO_MARKERS for t in toks) / max(len(toks), 1),
            en=sum(t in EN_STOP for t in toks) / max(len(toks), 1),
            sample=sv[:1])
    log.add("Per-field profile (used for detection):")
    for k, s in stats.items():
        log.add("  %-12s filled=%d distinct_ratio=%.2f mean_len=%.1f mean_tokens=%.2f "
                "POS_prefix_rate=%.2f url_rate=%.2f ilocano_marker_rate=%.3f english_stopword_rate=%.3f"
                % (k, s["filled"], s["distinct"], s["mean_len"], s["mean_tok"], s["pos_rate"],
                   s["url_rate"], s["ilo"], s["en"]))
    text_fields = [k for k, s in stats.items() if s["filled"] and s["url_rate"] < 0.5]
    const_fields = [k for k in text_fields if stats[k]["ndistinct"] <= 3]
    free = [k for k in text_fields if k not in const_fields]
    pos_field = max(free, key=lambda k: stats[k]["pos_rate"], default=None)
    sep_pos_field = None
    if pos_field is None or stats[pos_field]["pos_rate"] < 0.5:
        pos_field = None
        for k in free:  # a dedicated, low-cardinality POS column?
            if stats[k]["ndistinct"] <= 40 and stats[k]["mean_len"] <= 14 and stats[k]["pos_rate"] >= 0.5:
                sep_pos_field = k
    rest = [k for k in free if k not in (pos_field, sep_pos_field)]
    if pos_field:
        def_field = pos_field
        word_cands = [k for k in rest if k != def_field]
    else:
        word_cands = rest
        def_field = None
    word_field = sorted(word_cands, key=lambda k: (-stats[k]["distinct"], stats[k]["mean_tok"]))[0] \
        if word_cands else None
    if def_field is None:
        others = [k for k in rest if k != word_field]
        def_field = max(others, key=lambda k: stats[k]["mean_len"], default=None)
    if word_field is None or def_field is None:
        sys.exit("ERROR: could not detect word/definition fields; keys=%s" % list(cols))
    # direction
    evidence, score = [], 0
    if stats[def_field]["ilo"] > stats[word_field]["ilo"]:
        score += 1
        evidence.append("Ilocano function-word rate higher in %r (%.3f) than in %r (%.3f)"
                        % (def_field, stats[def_field]["ilo"], word_field, stats[word_field]["ilo"]))
    elif stats[def_field]["ilo"] < stats[word_field]["ilo"]:
        score -= 1
        evidence.append("Ilocano function-word rate higher in %r" % word_field)
    if stats[word_field]["en"] > stats[def_field]["en"]:
        score += 1
        evidence.append("English stop-word rate higher in %r" % word_field)
    elif stats[word_field]["en"] < stats[def_field]["en"]:
        score -= 1
        evidence.append("English stop-word rate higher in %r" % def_field)
    for k in const_fields:
        vals = {clean(v) for v in cols[k] if isinstance(v, str)}
        if any(v.lower() == "ilocano" for v in vals):
            score += 1
            evidence.append("constant field %r = 'Ilocano' (target language of a headword lookup)" % k)
    link_field = next((k for k, s in stats.items() if s["url_rate"] >= 0.5), None)
    if link_field:
        hits = tot = 0
        for r in recs:
            u, w = r.get(link_field), clean(r.get(word_field))
            if isinstance(u, str) and w:
                tot += 1
                slug = u.rstrip("/").rsplit("/", 1)[-1].replace("-", " ").lower()
                hits += slug == w.lower()
        if tot:
            evidence.append("URL slug equals %r value in %.0f%% of rows (site headword = %r)"
                            % (word_field, 100.0 * hits / tot, word_field))
    direction = "eng2ilo" if score >= 0 else "ilo2eng"
    if score == 0:
        log.add("ASSUMPTION: direction evidence was a tie; defaulting to English->Ilocano.")
    schema = dict(word_field=word_field, def_field=def_field, pos_field=sep_pos_field,
                  link_field=link_field, direction=direction, evidence=evidence)
    return schema


# ----------------------------------------------------------------------------
# Definition parsing
# ----------------------------------------------------------------------------
TOKEN_RE = re.compile(r"\s*([A-Za-z]+)\s*(\.)?")
SEP_RE = re.compile(r"\s*(?:,|&|/|\band\b)\s*")


def split_pos_prefix(text):
    """Return (raw_pos_text, rest). Handles 'n.', 'adj., v.', 'n. & adj.', 'n.n', 'n .', '. x'."""
    s, pos, toks = text, 0, []
    m = re.match(r"\s*\.\s*", s)
    if m:  # stray leading dot with the POS letters lost
        return "", s[m.end():].strip()
    while True:
        m = TOKEN_RE.match(s, pos)
        if not m or m.group(1).lower() not in POS_LABEL:
            break
        toks.append(m.group(1))
        pos = m.end()
        m2 = SEP_RE.match(s, pos)
        if m2:
            m3 = TOKEN_RE.match(s, m2.end())
            if m3 and m3.group(1).lower() in POS_LABEL:
                pos = m2.end()
                continue
        m3 = TOKEN_RE.match(s, pos)
        if m3 and m3.group(1).lower() in POS_LABEL:
            continue
        break
    return s[:pos].strip(), s[pos:].strip()


def normalize_pos(raw):
    labels = []
    for t in re.findall(r"[A-Za-z]+", raw or ""):
        lab = POS_LABEL.get(t.lower())
        if lab and lab not in labels:
            labels.append(lab)
    return "/".join(labels) if labels else "unknown"


def split_top_level(text):
    parts, buf, depth = [], [], 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch in ",;" and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return parts


def clean_segment(seg):
    had_paren = "(" in seg or ")" in seg
    seg = re.sub(r"\([^)]*\)", " ", seg)
    seg = re.sub(r"\([^)]*$", " ", seg)
    seg = seg.replace(")", " ")
    seg = clean(seg).strip(" .,;:!?\"'“”‘’")
    if not re.search(r"[A-Za-zÀ-ÿ]", seg):
        seg = ""
    return seg, had_paren


# ----------------------------------------------------------------------------
# Spelling rules
# ----------------------------------------------------------------------------
def tricky_tokens(word):
    """Return ({token_index: [reasons]}, ) for whitespace tokens that look Spanish."""
    flagged = {}
    for i, tok in enumerate(word.split(" ")):
        rs = []
        for name, rx, why in TRICKY_PATTERNS:
            m = rx.search(tok)
            if m:
                rs.append("%s ['%s' in '%s']" % (why, m.group(0), tok))
        if rs:
            flagged[i] = rs
    return flagged


def convert_tokens(word, flagged, src, dst):
    out = []
    for i, tok in enumerate(word.split(" ")):
        out.append(tok if i in flagged else tok.replace(src, dst).replace(src.upper(), dst.upper()))
    return " ".join(out)


# ----------------------------------------------------------------------------
# De-duplication
# ----------------------------------------------------------------------------
def uniq_ci(items):
    seen, out = set(), []
    for it in items:
        k = it.lower()
        if it and k not in seen:
            seen.add(k)
            out.append(it)
    return out


def dedupe(rows):
    seen, uniq = set(), []
    for r in rows:
        key = (r["old"].lower(), r["new"].lower(), r["english"].lower(), r["pos"].lower(), r["kw"].lower())
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    exact_removed = len(rows) - len(uniq)
    groups = OrderedDict()
    for r in uniq:
        groups.setdefault((r["new"].lower(), r["pos"].lower()), []).append(r)
    merged, groups_merged, absorbed, old_conflicts = [], 0, 0, 0
    for g in groups.values():
        if len(g) == 1:
            merged.append(g[0])
            continue
        groups_merged += 1
        absorbed += len(g) - 1
        olds = uniq_ci([x["old"] for x in g])
        if len(olds) > 1:
            old_conflicts += 1
        merged.append(dict(
            old="; ".join(olds), new=g[0]["new"],
            english="; ".join(uniq_ci([x["english"] for x in g])), pos=g[0]["pos"],
            kw="; ".join(uniq_ci([k for x in g for k in x["kw"].split("; ")])),
            tricky=next((x["tricky"] for x in g if x["tricky"]), "")))
    merged.sort(key=lambda r: (r["new"].lower(), r["new"], r["pos"], r["english"].lower()))
    return merged, exact_removed, groups_merged, absorbed, old_conflicts


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="path to ilocano_dictionary.json")
    default_out = HERE if os.path.basename(HERE) == "output" else os.path.join(HERE, "output")
    ap.add_argument("--outdir", default=default_out, help="output folder (default: %(default)s)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    log = Log()
    log.add("cleaning_log.txt -- Ilocano dictionary preprocessing")

    # ---- Step 0: input, hash, schema -----------------------------------------------------------
    log.h("0. INPUT, INTEGRITY HASH, SCHEMA DETECTION")
    src = locate_input(args.input, log)
    h_before = sha256_file(src)
    log.add("SHA-256 of source BEFORE processing: %s" % h_before)
    log.add("Source size: %d bytes" % os.path.getsize(src))
    with open(src, "r", encoding="utf-8-sig") as fh:
        data = json.load(fh)
    log.add("Top-level JSON type: %s" % type(data).__name__)
    print("Top-level type:", type(data).__name__)
    if isinstance(data, dict):
        log.add("Top-level keys: %s" % list(data.keys())[:50])
        print("Top-level keys:", list(data.keys())[:50])
    recs = to_records(data, log)
    keys = list(OrderedDict((k, 1) for r in recs for k in r))
    log.add("Number of records: %d" % len(recs))
    log.add("Record keys: %s" % keys)
    print("Records:", len(recs), "| record keys:", keys)
    log.add("10 sample entries:")
    print("10 sample entries:")
    for r in recs[:10]:
        line = json.dumps(r, ensure_ascii=False)
        log.add("  " + line)
        print("  " + line)
    sch = detect_schema(recs, log)
    wf, df = sch["word_field"], sch["def_field"]
    eng_is_word = sch["direction"] == "eng2ilo"
    log.add("")
    log.add("DETECTED SCHEMA:")
    log.add("  field holding the headword          : %r  -> %s" % (wf, "ENGLISH" if eng_is_word else "ILOCANO"))
    log.add("  field holding the definition        : %r  -> %s" % (df, "ILOCANO (comma/semicolon separated list)" if eng_is_word else "ENGLISH"))
    log.add("  word type (part of speech)          : %s"
            % ("separate field %r" % sch["pos_field"] if sch["pos_field"] else
               "embedded as a prefix of %r, e.g. 'n. canen' / 'adj., v. awan'" % df))
    log.add("  link / language fields (ignored)    : %r / constant 'Ilocano'" % sch["link_field"])
    log.add("  direction                           : %s" % ("English -> Ilocano" if eng_is_word else "Ilocano -> English"))
    for e in sch["evidence"]:
        log.add("  evidence: " + e)
    print("Detected: headword=%r (%s), definition=%r (%s), POS embedded in definition prefix; direction=%s"
          % (wf, "English" if eng_is_word else "Ilocano", df, "Ilocano" if eng_is_word else "English",
             sch["direction"]))

    log.h("ASSUMPTIONS")
    A = []
    A.append("Input is used from the local copy (no network needed). A download from the GitHub raw URL "
             "is attempted only if no local file is found.")
    A.append("'English side' for keyword matching = the English headword (cleaned). Ilocano definition text "
             "is NOT searched, except that the four Ilocano dish names (dinengdeng, pinakbet, bagnet, "
             "pinapaitan) are searched on the Ilocano side because they cannot occur in English headwords.")
    A.append("Whole-word matching is literal, so inflected forms (eating, cooked, plates ...) were added to "
             "the keyword list; a few obvious food words not in your list were also added "
             "(pumpkin, ham, cheese ...). Both groups are listed below and visible in match_keyword.")
    A.append("All matched keywords are recorded in match_keyword, joined with '; ' (list order).")
    A.append("Text is NFC-normalised, whitespace collapsed; capitalisation kept; comparisons for duplicates "
             "are case-insensitive.")
    A.append("Parenthetical notes in Ilocano text, e.g. '(prefix)', '(reservoir)', are removed from the Ilocano "
             "word; segments left empty are dropped. Commas/semicolons inside parentheses do not split.")
    A.append("Headwords carrying a stray POS suffix (e.g. 'mason n.') are cleaned and the POS taken from it "
             "if the definition has none.")
    A.append("Typos in POS labels are mapped by intent: 'ad.'->adverb, 'dj.'->adjective, 'ajd'->adjective; "
             "a lone leading '.' (POS letters lost) -> unknown.")
    A.append("Entries with several POS (e.g. 'n., adj.') get one combined label 'noun/adjective'.")
    A.append("Tricky Spanish-looking words stay in the lexicon (new_spelling == old_spelling) AND are listed "
             "in unchanged_spellings.csv. The check runs per whitespace-separated token, so in a phrase only "
             "the tricky token is protected; the other tokens are converted normally.")
    A.append("Tricky patterns beyond your list: gu+e/i, 'ck', 'c' before y, accented vowels. Words flagged for "
             "'ll', 'z', 'ch' etc. are listed even if they contain no c/j (the reason says so).")
    A.append("Step order of execution: 1, 2, 5, 3, 4, 6, 7, 8, 9 (step 5 must precede 3 and 4).")
    A.append("Merging (step 8): key = (new_spelling, word_type), case-insensitive. English meanings and "
             "match_keywords are joined with '; '; differing old_spellings are joined with '; ' too.")
    A.append("Phrases (step 6) = Ilocano entries that still contain a space after cleaning; they are "
             "de-duplicated separately from single words.")
    for i, a in enumerate(A, 1):
        log.add("%2d. %s" % (i, a))

    # ---- Parse entries ---------------------------------------------------------------------------
    entries, pos_from_headword, empty_def = [], 0, 0
    for i, r in enumerate(recs):
        w, d = clean(r.get(wf)), clean(r.get(df))
        if sch["pos_field"]:
            raw_pos, rest = clean(r.get(sch["pos_field"])), d
        else:
            raw_pos, rest = split_pos_prefix(d)
        m = re.match(r"^(.*?)\s+(n|v|adj|adv)\.?$", w, re.I)
        if m:
            w = m.group(1).strip()
            if not raw_pos:
                raw_pos = m.group(2)
            pos_from_headword += 1
        if not rest or not w:
            empty_def += 1
        english, ilo_text = (w, rest) if eng_is_word else (rest, w)
        entries.append(dict(idx=i, english=english, ilo_text=ilo_text, raw_pos=raw_pos))
    log.add("")
    log.add("Parsed %d entries; %d with a POS suffix on the headword; %d with an empty word or definition "
            "(kept in parsing, produce no rows)." % (len(entries), pos_from_headword, empty_def))

    # ---- Step 1: culinary filter -----------------------------------------------------------------
    log.h("STEP 1 -- CULINARY FILTER (whole-word, case-insensitive, on the English side)")
    core = [(k, kw_regex(k)) for k in CORE_KEYWORDS]
    infl = [(k, kw_regex(k)) for k in INFLECTED_KEYWORDS if k not in CORE_KEYWORDS]
    extra = [(k, kw_regex(k)) for k in EXTRA_KEYWORDS]
    dish = [(k, kw_regex(k)) for k in ILOCANO_DISH_KEYWORDS]
    log.add("Core keywords (%d, from your list): %s" % (len(core), ", ".join(CORE_KEYWORDS)))
    log.add("Added inflected forms (%d): %s" % (len(infl), ", ".join(k for k, _ in infl)))
    log.add("Added extra food words (%d): %s" % (len(extra), ", ".join(EXTRA_KEYWORDS)))
    log.add("Ilocano-side dish names also searched on the Ilocano text: %s" % ", ".join(ILOCANO_DISH_KEYWORDS))
    log.add("")
    log.add("Exclusion list (applied to entries that matched a keyword):")
    excl = [(re.compile(a, re.I), re.compile(b, re.I) if b else None, why) for a, b, why in EXCLUSIONS]
    for a, b, why in EXCLUSIONS:
        log.add("  headword~/%s/%s -> %s" % (a, (" + definition~/%s/" % b) if b else "", why))
    n0 = len(entries)
    kept, excluded = [], []
    dish_hits = 0
    for e in entries:
        hits = [k for k, rx in core + infl + extra if rx.search(e["english"])]
        for k, rx in dish:
            if rx.search(e["english"]) or rx.search(e["ilo_text"]):
                hits.append(k)
                dish_hits += 1
        if not hits:
            continue
        why = None
        for hx, dx, reason in excl:
            if hx.search(e["english"]) and (dx is None or dx.search((e["raw_pos"] + " " + e["ilo_text"]) if eng_is_word else (e["raw_pos"] + " " + e["english"]))):
                why = reason
                break
        if why:
            excluded.append((e, hits, why))
            continue
        e["kw"] = "; ".join(uniq_ci(hits))
        kept.append(e)
    log.step("1", "entries", n0, len(kept), "(%d keyword hits, %d removed by exclusions)"
             % (len(kept) + len(excluded), len(excluded)))
    log.add("Ilocano-side dish-name hits: %d" % dish_hits)
    log.add("Exclusions actually applied (%d):" % len(excluded))
    for e, hits, why in excluded:
        log.add("  EXCLUDED %r | %s | matched=%s | %s" % (e["english"], e["ilo_text"], "; ".join(hits), why))

    # ---- Step 2: invert (one row per Ilocano word) -------------------------------------------------
    log.h("STEP 2 -- ONE ROW PER ILOCANO WORD")
    rows, paren_cnt, dropped_segments = [], 0, 0
    for e in kept:
        for seg in split_top_level(e["ilo_text"]):
            word, hp = clean_segment(seg)
            paren_cnt += hp
            if not word:
                dropped_segments += 1
                continue
            rows.append(dict(old=word, new=word, english=e["english"], raw_pos=e["raw_pos"],
                             pos="", kw=e["kw"], tricky="", flagged={}))
    if eng_is_word:
        log.add("Source is English -> Ilocano, so it was INVERTED: each English entry's Ilocano list was split on "
                "commas/semicolons (outside parentheses) into one row per Ilocano word.")
    else:
        log.add("Source is already Ilocano -> English: inversion SKIPPED; rows are only expanded by splitting "
                "multi-word Ilocano headword lists on commas/semicolons.")
    log.add("Segments with parenthetical notes cleaned: %d; segments dropped as empty after cleaning: %d"
            % (paren_cnt, dropped_segments))
    log.step("2", "entries->rows", len(kept), len(rows))

    # ---- Step 5 (runs before 3 and 4): tricky check -------------------------------------------------
    log.h("STEP 5 (executed BEFORE steps 3 and 4) -- SPANISH-LOOKING / TRICKY WORDS")
    log.add("Patterns checked per token: " + "; ".join("%s: %s" % (n, w) for n, _, w in TRICKY_PATTERNS))
    pat_counter, affected = Counter(), 0
    for r in rows:
        fl = tricky_tokens(r["old"])
        r["flagged"] = fl
        if fl:
            r["tricky"] = "; ".join(x for rs in fl.values() for x in rs)
            for rs in fl.values():
                for x in rs:
                    pat_counter[x.split(" [")[0]] += 1
            if re.search(r"[cCjJ]", r["old"]):
                affected += 1
            else:
                r["tricky"] += " (no c/j present, so unchanged either way)"
    n_tricky = sum(1 for r in rows if r["flagged"])
    log.step("5", "rows", len(rows), len(rows), "(%d rows flagged tricky; %d of them contain c/j and are "
             "therefore actually protected from replacement)" % (n_tricky, affected))
    for k, v in pat_counter.most_common():
        log.add("   %4d  %s" % (v, k))

    # ---- Steps 3 / 4: c->k, j->h -------------------------------------------------------------------
    log.h("STEP 3 -- c -> k  (tricky tokens skipped)")
    ch3 = 0
    for r in rows:
        new = convert_tokens(r["new"], r["flagged"], "c", "k")
        ch3 += new != r["new"]
        r["new"] = new
    log.step("3", "rows", len(rows), len(rows), "(%d spellings changed)" % ch3)
    log.h("STEP 4 -- j -> h  (tricky tokens skipped)")
    ch4 = 0
    for r in rows:
        new = convert_tokens(r["new"], r["flagged"], "j", "h")
        ch4 += new != r["new"]
        r["new"] = new
    log.step("4", "rows", len(rows), len(rows), "(%d spellings changed)" % ch4)

    # ---- Step 6: phrases --------------------------------------------------------------------------
    log.h("STEP 6 -- SEPARATE PHRASES FROM SINGLE WORDS")
    lex = [r for r in rows if " " not in r["old"]]
    phr = [r for r in rows if " " in r["old"]]
    log.step("6", "rows (single words)", len(rows), len(lex), "(%d phrase rows moved to culinary_phrases.csv)" % len(phr))

    # ---- Step 7: word types ------------------------------------------------------------------------
    log.h("STEP 7 -- NORMALISE WORD TYPES")
    raw_counter = Counter()
    for r in lex + phr:
        r["pos"] = normalize_pos(r["raw_pos"])
        raw_counter[(r["raw_pos"] or "(none)", r["pos"])] += 1
    log.add("Mapping table used (variant, case-insensitive, dots ignored -> label):")
    for k, v in POS_LABEL.items():
        log.add("   %-12s -> %s" % (k, v))
    log.add("   multiple POS (e.g. 'n., adj.', 'n. & adj.') -> labels joined by '/' in order, e.g. noun/adjective")
    log.add("   anything else / missing -> unknown")
    log.add("Raw labels actually seen in the culinary rows (rows):")
    for (raw, lab), c in sorted(raw_counter.items(), key=lambda x: (-x[1], x[0])):
        log.add("   %-14r -> %-22s %d" % (raw, lab, c))
    log.step("7", "rows (lexicon)", len(lex), len(lex))
    log.step("7", "rows (phrases)", len(phr), len(phr))

    # ---- Step 8: de-duplicate -----------------------------------------------------------------------
    log.h("STEP 8 -- REMOVE DUPLICATES / MERGE MEANINGS")
    lex_b, phr_b = len(lex), len(phr)
    lex, ex1, g1, a1, oc1 = dedupe(lex)
    phr, ex2, g2, a2, oc2 = dedupe(phr)
    log.step("8", "rows (lexicon)", lex_b, len(lex),
             "(exact duplicates removed=%d; groups merged=%d absorbing %d rows; groups with differing old_spelling=%d)"
             % (ex1, g1, a1, oc1))
    log.step("8", "rows (phrases)", phr_b, len(phr),
             "(exact duplicates removed=%d; groups merged=%d absorbing %d rows; groups with differing old_spelling=%d)"
             % (ex2, g2, a2, oc2))

    # ---- Step 9: save -------------------------------------------------------------------------------
    log.h("STEP 9 -- SAVE")
    header = ["old_spelling", "new_spelling", "english_meaning", "word_type", "match_keyword"]
    to_row = lambda r: [r["old"], r["new"], r["english"], r["pos"], r["kw"]]
    write_csv(os.path.join(args.outdir, "culinary_lexicon.csv"), header, [to_row(r) for r in lex])
    write_csv(os.path.join(args.outdir, "culinary_phrases.csv"), header, [to_row(r) for r in phr])
    unch = [(r["new"], r["english"], r["pos"], r["tricky"]) for r in lex if r["tricky"]]
    unch += [(r["new"], r["english"], r["pos"], "[phrase] " + r["tricky"]) for r in phr if r["tricky"]]
    unch.sort(key=lambda x: (x[0].lower(), x[0], x[2]))
    write_csv(os.path.join(args.outdir, "unchanged_spellings.csv"),
              ["word", "english_meaning", "word_type", "reason"], unch)
    log.step("9", "files", "-", "5 data/log files",
             "(lexicon=%d rows, phrases=%d rows, unchanged_spellings=%d rows)" % (len(lex), len(phr), len(unch)))

    # original copy + hash proof
    dst = os.path.join(args.outdir, FILENAME)
    if os.path.abspath(dst) != os.path.abspath(src):
        shutil.copyfile(src, dst)
    h_src_after, h_copy = sha256_file(src), sha256_file(dst)
    me = os.path.abspath(__file__)
    my_dst = os.path.join(args.outdir, "preprocess.py")
    if os.path.abspath(my_dst) != me:
        shutil.copyfile(me, my_dst)
    log.h("ORIGINAL FILE INTEGRITY")
    log.add("SHA-256 source BEFORE : %s" % h_before)
    log.add("SHA-256 source AFTER  : %s" % h_src_after)
    log.add("SHA-256 output copy   : %s" % h_copy)
    ok = h_before == h_src_after == h_copy
    log.add("Byte-for-byte identical and unmodified: %s" % ("YES" if ok else "NO -- INVESTIGATE"))

    # summary
    log.h("SUMMARY OF COUNTS PER STEP")
    table = ["%-5s %-22s %10s %10s  %s" % ("Step", "Unit", "Before", "After", "Note")]
    for s, u, b, a, n in log.steps:
        table.append("%-5s %-22s %10s %10s  %s" % (s, u, b, a, n))
    for t in table:
        log.add(t)
    with open(os.path.join(args.outdir, "cleaning_log.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(log.lines) + "\n")

    print("\nSUMMARY OF COUNTS PER STEP")
    print("\n".join(table))
    print("\nOriginal file hash before/after/copy identical:", ok, h_before)
    rng = random.Random(20240601)
    print("\n20 random rows from culinary_lexicon.csv:")
    for r in rng.sample(lex, min(20, len(lex))):
        print("  ", to_row(r))
    print("\n10 random rows from unchanged_spellings.csv:")
    for r in rng.sample(unch, min(10, len(unch))):
        print("  ", list(r))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
