"""
linkagex_fixes.py  --  drop-in fixes for nlp_service.py

Put this file next to nlp_service.py and import from it. Each section fixes
something that is visible in your 10 test outputs.

  1. AMOUNTS + LOSS  (outputs 2, 4, 5, 8, 9, 10)
       - amount_lost used max(amounts)  ->  "₹2,500 then ₹4,000" gave ₹4,000
       - "lost around ₹7,000", "₹9,100 was taken from my account",
         "transactions ... made without my permission" were not recognised as
         loss, so Outcome stayed empty while the amount box said "lost"
       - the loss rules were written 3 separate times and disagreed; there is
         now ONE list (LOSS_EVIDENCE_PATTERNS) used everywhere

  2. ORGANIZATIONS   (output 1 shows "Google Pay, Google")
       - longest-match with overlap suppression
       - WhatsApp/Telegram/Instagram... are CHANNELS, not organisations
         (they were showing up in both)
       - "Aadhaar" no longer maps to UIDAI (a victim mentioning their Aadhaar
         card is not mentioning the organisation)

  3. SEMANTIC FALLBACK for crime category  (5 of 10 outputs: "Not provided")
       - only runs when the rules found nothing
       - always marks the result for human review

  4. GROQ SAFETY    (your file's LLM step)
       - the LLM was allowed a DIFFERENT category list than your rules
         ("Online Financial Fraud" vs "Financial Fraud"), so the same crime
         could get two labels that never match during linkage
       - identifiers are masked before any complaint text leaves the machine

HOW TO WIRE IT IN (inside extract_complaint_information):

    from linkagex_fixes import (analyze_amounts, extract_organizations_v2,
                                classify_crime_hybrid)

    crime_category, crime_subcategory, crime_all_matches = \\
        classify_crime_hybrid(text, classify_crime)

    organizations = extract_organizations_v2(text)

    amt = analyze_amounts(text)            # replaces detected_amount / max()
    amount_lost   = amt["amount_lost"]
    loss_occurred = amt["loss_occurred"]
    contextual_details["amount_details"] = amt["amount_details"]
    if loss_occurred is True and "Financial Loss" not in outcomes:
        outcomes.append("Financial Loss")
    if loss_occurred is False:
        outcomes = [o for o in outcomes if o != "Financial Loss"]
"""

import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

# =====================================================================
# 1. AMOUNTS + LOSS
# =====================================================================

_CUR = r"(?:₹|\brs\.?|\binr)"

# ONE source of truth for "money actually left the victim's account".
LOSS_EVIDENCE_PATTERNS = [
    rf"\blost\b[^.]{{0,40}}?{_CUR}\s*[\d,]+",
    r"\blost\b[^.]{0,30}\b(?:money|amount|funds|savings)\b",
    rf"{_CUR}\s*[\d,]+(?:\.\d+)?\s*(?:\w+\s+){{0,3}}?(?:was|were|has been|had been|got)\s+"
    r"(?:debited|deducted|taken|withdrawn|charged|transferred|stolen|siphoned|cut)\b",
    rf"\b(?:debited|deducted|withdrawn|charged)\b[^.]{{0,30}}{_CUR}\s*[\d,]+",
    r"\b(?:amount|money|funds?|payment)\b[^.]{0,30}\b(?:was|were)\s+"
    r"(?:debited|deducted|taken|withdrawn|stolen|transferred)\b",
    r"\b(?:was|were)\s+(?:made|taken|debited|deducted)\s+from\s+my\s+(?:bank\s+)?account\b",
    r"\b(?:transactions?|payments?|debits?|withdrawals?)\b[^.]{0,60}"
    r"\bwithout my (?:permission|consent|knowledge)\b",
    r"\bunauthori[sz]ed\s+(?:transactions?|payments?|debits?|withdrawals?|charges)\b",
    rf"\b(?:i|we)\s+(?:have\s+|had\s+)?(?:paid|transferred|sent|deposited|invested)\b[^.]{{0,40}}{_CUR}\s*[\d,]+",
    r"\bafter\s+(?:i|we)\s+paid\b",
    r"\bmoney was stolen\b|\bfunds were stolen\b|\bsiphoned\b",
    r"\bfinancial loss\b|\btotal loss\b",
    r"\bfraudulent\s+transactions?\b",
]

NO_LOSS_PATTERNS = [
    r"\bno money was (?:lost|deducted|debited)\b",
    r"\bno financial loss\b",
    r"\bno amount was (?:deducted|debited|lost)\b",
    r"\bnothing was (?:deducted|debited)\b",
    r"\b(?:did not|didn't|never) lose any money\b",
    r"\bno loss occurred\b",
    r"\b(?:i|we) (?:did not|didn't|have not|haven't|never) (?:pay|paid|transfer|transferred|send|sent)\b",
]

# "the victim really did pay", for when the amount was only named as a demand.
ACTUAL_PAYMENT_CUES = [
    r"\b(?:i|we)\s+(?:have\s+|had\s+)?(?:paid|transferred|sent|deposited)\b",
    r"\bpayment was made\b",
    r"\bpaid the amount\b",
    r"\bmade (?:the|a) payment\b",
]

_LOSS_RE = [re.compile(p, re.IGNORECASE) for p in LOSS_EVIDENCE_PATTERNS]
_NO_LOSS_RE = [re.compile(p, re.IGNORECASE) for p in NO_LOSS_PATTERNS]
_PAYMENT_RE = [re.compile(p, re.IGNORECASE) for p in ACTUAL_PAYMENT_CUES]


def has_loss_evidence(text: str) -> bool:
    return any(r.search(text) for r in _LOSS_RE)


def has_no_loss_evidence(text: str) -> bool:
    return any(r.search(text) for r in _NO_LOSS_RE)


def has_actual_payment_cue(text: str) -> bool:
    return any(r.search(text) for r in _PAYMENT_RE)


_AMT_SYMBOL_RE = re.compile(
    r"(?:₹|\bRs\.?|\bINR)\s*([\d,]+(?:\.\d+)?)(?:\s*(lakhs?|lacs?|crores?|cr|k)\b)?",
    re.IGNORECASE,
)
_AMT_WORD_RE = re.compile(r"\b([\d,]+(?:\.\d+)?)\s*(?:rupees|rs)\b", re.IGNORECASE)
_MULT = {"lakh": 1e5, "lac": 1e5, "crore": 1e7, "cr": 1e7, "k": 1e3}
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z₹])")


def _num(raw: str, unit: Optional[str] = None) -> Optional[float]:
    try:
        v = float(raw.replace(",", ""))
    except ValueError:
        return None
    return v * _MULT.get(unit.lower().rstrip("s"), 1) if unit else v


def _sentence_span(text: str, start: int, end: int) -> Tuple[int, int]:
    s = 0
    for m in _SENT_SPLIT.finditer(text):
        if m.end() <= start:
            s = m.end()
        elif m.start() >= end:
            return s, m.start()
    return s, len(text)


# cues that sit to the LEFT of an amount ---------------------------------
_LEFT_LOSS = [re.compile(p, re.IGNORECASE) for p in [
    r"\b(?:i|we)\s+(?:\w+\s+){0,3}?(?:paid|transferred|sent|deposited|invested|spent|lost|gave|wired|remitted)\b",
    r"\blost\b",
    r"\b(?:i|we)\s+(?:\w+\s+){0,2}made\s+(?:a\s+|the\s+)?payments?\b",
]]
_LEFT_DEMAND = [re.compile(p, re.IGNORECASE) for p in [
    r"\b(?:asked|told|requested|demanded|instructed|wanted|required)\b[^.]{0,40}?\b(?:pay|transfer|send|deposit|invest|give)\b",
    r"\b(?:demanded|demanding|asking for|asked for|requested)\b",
]]
_LEFT_MENTION = [re.compile(p, re.IGNORECASE) for p in [
    r"\b(?:prize|reward|lottery|cashback|salary|stipend|balance|limit|worth|valued at|won)\b",
]]
_NEGATION = re.compile(r"\b(?:not|never|no|didn't|haven't|hadn't)\b", re.IGNORECASE)

# cues to the RIGHT of an amount (same sentence, ~90 chars) -----------------
_RIGHT_LOSS = [re.compile(p, re.IGNORECASE) for p in [
    r"^\W*(?:\w+\s+){0,4}?(?:was|were|has been|had been|got)\s+"
    r"(?:debited|deducted|taken|withdrawn|charged|transferred|stolen|siphoned|cut)\b",
    r"^\W*(?:\w+\s+){0,3}?(?:was|were)\s+made\s+from\s+my\b",
    r"^\W*(?:\w+\s+){0,3}?(?:was|were)\s+lost\b",
    r"\bwithout my (?:permission|consent|knowledge|authori[sz]ation)\b",
    r"\bunauthori[sz]ed\b",
]]
_RIGHT_DEMAND = [re.compile(p, re.IGNORECASE) for p in [
    r"^\W*(?:\w+\s+){0,2}?(?:was|were)\s+(?:demanded|requested|asked)\b",
]]
_RIGHT_TOTAL = re.compile(r"^\W*(?:\w+\s+){0,3}?(?:in total|altogether|in all|in sum)\b", re.IGNORECASE)
_LEFT_TOTAL = re.compile(r"\btotal(?:\s+(?:loss|amount))?\s*(?:of|was|is|:)?\s*$", re.IGNORECASE)


def _nearest(window: str, patterns: List[re.Pattern]) -> Optional[int]:
    """Distance from the END of the nearest matching cue to the end of `window`."""
    best = None
    for p in patterns:
        for m in p.finditer(window):
            if _NEGATION.search(m.group(0)):
                continue  # "I have not paid" is not a payment
            d = len(window) - m.end()
            if best is None or d < best:
                best = d
    return best


def _classify_amount(left: str, right: str, sentence: str) -> Tuple[str, bool]:
    """-> (role, is_total). role in {"lost", "demanded", "mentioned"}."""
    is_total = bool(_RIGHT_TOTAL.search(right) or _LEFT_TOTAL.search(left))

    if any(p.search(right) for p in _RIGHT_LOSS):
        return "lost", is_total
    if any(p.search(right) for p in _RIGHT_DEMAND):
        return "demanded", is_total

    cands = {
        "lost": _nearest(left, _LEFT_LOSS),
        "demanded": _nearest(left, _LEFT_DEMAND),
        "mentioned": _nearest(left, _LEFT_MENTION),
    }
    cands = {k: v for k, v in cands.items() if v is not None}
    if cands:
        return min(cands, key=cands.get), is_total

    # No local cue: a sentence that itself reports an unauthorised debit
    if any(r.search(sentence) for r in _LOSS_RE):
        return "lost", is_total
    return "mentioned", is_total


_MEANING = {
    "lost": "Amount lost / debited",
    "demanded": "Amount requested / demanded; payment not established",
    "mentioned": "Amount mentioned; purpose or payment status unclear",
}


def analyze_amounts(text: str) -> Dict[str, Any]:
    """
    Classify EACH amount by its own context, then total the losses.

      "I transferred ₹2,500 and then ₹4,000"  -> lost 2,500 + lost 4,000 = 6,500
      "asked me to pay ₹2,500"                -> demanded (not a loss yet)
      "I lost around ₹7,000 in total"         -> lost, flagged as a total
      "₹9,100 was taken from my account"      -> lost

    Returns amount_details (same keys your UI already reads, plus "role"),
    amount_lost, amount_lost_basis, loss_occurred.
    """
    mentions: List[Tuple[int, int, float]] = []
    for m in _AMT_SYMBOL_RE.finditer(text):
        v = _num(m.group(1), m.group(2))
        if v is not None:
            mentions.append((m.start(), m.end(), v))
    for m in _AMT_WORD_RE.finditer(text):
        if any(s <= m.start() < e for s, e, _ in mentions):
            continue
        v = _num(m.group(1))
        if v is not None:
            mentions.append((m.start(), m.end(), v))
    mentions.sort()

    details: List[Dict[str, Any]] = []
    for start, end, value in mentions:
        s, e = _sentence_span(text, start, end)
        left = text[max(s, start - 80):start]
        right = text[end:min(e, end + 90)]
        sentence = text[s:e].strip()
        role, is_total = _classify_amount(left, right, sentence)
        if is_total and role == "mentioned" and has_loss_evidence(sentence):
            role = "lost"
        details.append({
            "value": value, "role": role, "is_total": is_total,
            "meaning": _MEANING[role], "context": sentence,
            "evidence_status": "supported_by_complaint",
        })

    no_loss = has_no_loss_evidence(text)
    lost_vals = list(dict.fromkeys(d["value"] for d in details if d["role"] == "lost"))
    totals = [d["value"] for d in details if d["role"] == "lost" and d["is_total"]]
    demanded = list(dict.fromkeys(d["value"] for d in details if d["role"] == "demanded"))

    amount_lost: Optional[float] = None
    basis: Optional[str] = None
    if no_loss:
        pass
    elif totals:
        amount_lost, basis = max(totals), "explicit_total"
    elif lost_vals:
        amount_lost, basis = float(sum(lost_vals)), "explicit"
    elif len(demanded) == 1 and has_actual_payment_cue(text):
        # They named one demanded amount and later say "I paid": probable,
        # not certain. Marked so the citizen is asked to confirm.
        amount_lost, basis = demanded[0], "inferred_from_demand"
        for d in details:
            if d["role"] == "demanded":
                d["meaning"] = "Amount requested and (apparently) paid; please confirm"

    if no_loss:
        loss_occurred: Optional[bool] = False
    elif amount_lost is not None or has_loss_evidence(text):
        loss_occurred = True
    else:
        loss_occurred = None

    return {
        "amount_details": details,
        "amount_lost": amount_lost,
        "amount_lost_basis": basis,
        "loss_occurred": loss_occurred,
    }


# =====================================================================
# 2. ORGANIZATIONS
# =====================================================================

# Channels / platforms: they belong in channels + platform_or_service.
PLATFORM_NAMES = {"whatsapp", "instagram", "facebook", "telegram", "skype", "twitter", "meta"}

ORG_ALIASES: Dict[str, str] = {
    "HDFC Bank": "HDFC Bank", "HDFC": "HDFC Bank",
    "ICICI Bank": "ICICI Bank", "ICICI": "ICICI Bank",
    "Axis Bank": "Axis Bank", "Bank of Baroda": "Bank of Baroda",
    "Punjab National Bank": "Punjab National Bank", "PNB": "Punjab National Bank",
    "State Bank of India": "SBI", "SBI": "SBI",
    "Kotak Mahindra Bank": "Kotak Mahindra Bank", "Canara Bank": "Canara Bank",
    "Union Bank": "Union Bank", "Yes Bank": "Yes Bank",
    "Google Pay": "Google Pay", "GPay": "Google Pay", "PhonePe": "PhonePe",
    "Paytm": "Paytm", "Amazon Pay": "Amazon Pay", "Freecharge": "Freecharge",
    "Amazon": "Amazon", "Flipkart": "Flipkart", "Myntra": "Myntra",
    "Swiggy": "Swiggy", "Zomato": "Zomato",
    "Microsoft": "Microsoft", "Google": "Google",
    "FedEx": "FedEx", "DHL": "DHL", "Blue Dart": "Blue Dart", "India Post": "India Post",
    "Reserve Bank of India": "RBI", "RBI": "RBI", "UIDAI": "UIDAI",
    "Income Tax Department": "Income Tax Department", "Income Tax": "Income Tax Department",
    "Narcotics Control Bureau": "Narcotics Control Bureau",
    "Enforcement Directorate": "Enforcement Directorate",
    "CBI": "CBI", "TRAI": "TRAI",
}
_ORG_ORDER = sorted(ORG_ALIASES, key=len, reverse=True)


def extract_organizations_v2(text: str, include_platforms: bool = False) -> List[str]:
    """Longest match wins; 'Google' inside 'Google Pay' is not reported twice."""
    taken: List[Tuple[int, int]] = []
    found: List[Tuple[int, str]] = []
    for alias in _ORG_ORDER:
        for m in re.finditer(r"\b" + re.escape(alias) + r"\b", text, re.IGNORECASE):
            if any(m.start() < e and m.end() > s for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append((m.start(), ORG_ALIASES[alias]))
    found.sort()
    out: List[str] = []
    for _, name in found:
        if name not in out:
            out.append(name)
    return out


# =====================================================================
# 3. SEMANTIC FALLBACK FOR CRIME CATEGORY
# =====================================================================
# Runs ONLY if the rules found nothing. Category/subcategory names below are
# the ones your rule taxonomy already uses, so labels stay comparable.
# (Financial Fraud / Authority-Courier Impersonation is NEW: add it to your
#  evaluation label mapping before you score against the CSV labels.)

SEMANTIC_PROTOTYPES: Dict[Tuple[str, str], List[str]] = {
    ("Financial Fraud", "UPI/Payment Fraud"): [
        "A caller told me to approve a collect request on Google Pay or PhonePe and money was deducted from my account.",
        "I scanned a QR code or approved a UPI request and money was debited without my intending to pay.",
        "I transferred money to different UPI IDs on someone's instructions.",
    ],
    ("Financial Fraud", "Credit Card Fraud"): [
        "Unauthorized transactions appeared on my credit card that I never made.",
    ],
    ("Financial Fraud", "Cryptocurrency Scams"): [
        "I invested in a crypto trading platform and could not withdraw my money.",
    ],
    ("Financial Fraud", "Investment/Loan Scams"): [
        "Someone on Telegram asked me to pay in instalments for a task or investment and promised returns, then asked for more money.",
        "I was promised guaranteed returns or a loan and paid fees but received nothing.",
    ],
    ("Financial Fraud", "Phishing/Fake Websites"): [
        "I clicked a link in an SMS, entered my card or bank details and money was debited.",
    ],
    ("Financial Fraud", "Authority/Courier Impersonation"): [
        "A caller claiming to be from customs, police or a courier company said a parcel in my name had illegal items and demanded money.",
        "Someone posing as a police officer said I was under digital arrest and told me to transfer money.",
    ],
    ("E-Commerce & Job Fraud", "Fake Job Offers"): [
        "I was offered a job on WhatsApp and asked to pay a registration or training fee.",
    ],
    ("E-Commerce & Job Fraud", "Fake Shopping Websites"): [
        "I ordered from an online store or seller who took my payment and never delivered.",
    ],
    ("Ransomware & Malware", "Malware"): [
        "I installed an app or APK from a link and afterwards transactions were made from my account without my permission.",
        "A malicious app on my phone read my messages and gave someone access to my bank.",
    ],
    ("Social Media Crimes", "Fake Accounts/Impersonation"): [
        "Someone created a fake profile with my name and photos.",
    ],
    ("Hacking & Unauthorized Access", "Account Takeover"): [
        "Someone got into my account and changed the password so I could not log in.",
    ],
    ("Online Harassment & Threats", "Threatening Messages"): [
        "Someone is threatening me and demanding money or they will share my photos.",
    ],
}

_MODEL: Any = None
_PROTO_VECS: Any = None
_PROTO_KEYS: List[Tuple[str, str]] = []


def _get_model():
    """Lazy-load; returns None (with one warning) if sentence-transformers is missing."""
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as exc:  # ImportError, offline, etc.
            warnings.warn(f"Semantic fallback disabled: {exc}")
            _MODEL = False
    return _MODEL or None


def _prototype_matrix():
    global _PROTO_VECS, _PROTO_KEYS
    model = _get_model()
    if model is None:
        return None, []
    if _PROTO_VECS is None:
        sentences, keys = [], []
        for key, examples in SEMANTIC_PROTOTYPES.items():
            for s in examples:
                sentences.append(s)
                keys.append(key)
        _PROTO_VECS = model.encode(sentences, normalize_embeddings=True)
        _PROTO_KEYS = keys
    return _PROTO_VECS, _PROTO_KEYS


def explain_semantic(text: str, top_n: int = 3) -> List[Tuple[str, str, float]]:
    """Top matches with scores. Run this on your 10 test complaints to pick a threshold."""
    import numpy as np
    model = _get_model()
    vecs, keys = _prototype_matrix()
    if model is None or vecs is None:
        return []
    q = model.encode([text], normalize_embeddings=True)[0]
    sims = vecs @ q
    best: Dict[Tuple[str, str], float] = {}
    for key, s in zip(keys, sims):
        best[key] = max(best.get(key, -1.0), float(s))
    ranked = sorted(best.items(), key=lambda kv: -kv[1])[:top_n]
    return [(k[0], k[1], round(v, 3)) for k, v in ranked]


def classify_crime_semantic(text: str, threshold: float = 0.40, margin: float = 0.03):
    """
    Returns (category, subcategory, score). (None, None, score) when the best
    match is below `threshold`, or when it barely beats the runner-up
    (`margin`): an honest "unclear" instead of a guess.
    NOTE: 0.40 / 0.03 are starting points, not tuned values; calibrate them.
    """
    top = explain_semantic(text, top_n=2)
    if not top:
        return None, None, 0.0
    cat, sub, score = top[0]
    if score < threshold:
        return None, None, score
    if len(top) > 1 and (score - top[1][2]) < margin and top[1][0] != cat:
        return None, None, score
    return cat, sub, score


def classify_crime_hybrid(text: str, rule_classifier):
    """
    rule_classifier = your existing classify_crime. Rules first (high precision),
    semantic only when the rules returned nothing. Semantic results carry
    source="semantic(...)" so the pipeline can force human review.
    """
    category, subcategory, matches = rule_classifier(text)
    if category:
        return category, subcategory, matches
    cat, sub, score = classify_crime_semantic(text)
    if cat:
        return cat, sub, [{"category": cat, "subcategory": sub,
                           "source": f"semantic(score={score:.2f})"}]
    return None, None, []


def came_from_semantic(crime_all_matches: List[Dict[str, str]]) -> bool:
    """Use in determine_needs_review: a semantic guess must always be confirmed."""
    return any(str(m.get("source", "")).startswith("semantic") for m in (crime_all_matches or []))


# =====================================================================
# 4. GROQ SAFETY
# =====================================================================

def get_taxonomy(rule_patterns: Optional[Dict[str, Dict[str, list]]] = None) -> Dict[str, List[str]]:
    """Category -> subcategories, from YOUR rule taxonomy plus the semantic prototypes."""
    tax: Dict[str, List[str]] = {}
    for cat, subs in (rule_patterns or {}).items():
        tax.setdefault(cat, []).extend(subs.keys())
    for cat, sub in SEMANTIC_PROTOTYPES:
        tax.setdefault(cat, [])
        if sub not in tax[cat]:
            tax[cat].append(sub)
    return tax


def validate_taxonomy(category: Optional[str], subcategory: Optional[str],
                      taxonomy: Dict[str, List[str]]) -> Tuple[Optional[str], Optional[str]]:
    """Drop any LLM label that is not in the closed taxonomy (no free-text labels)."""
    if category not in taxonomy:
        return None, None
    if subcategory not in taxonomy[category]:
        return category, None
    return category, subcategory


def build_groq_category_instruction(taxonomy: Dict[str, List[str]]) -> str:
    lines = [f"- {c}: {', '.join(subs)}" for c, subs in taxonomy.items()]
    return ("Choose crime_category and crime_subcategory ONLY from this list "
            "(category: allowed subcategories). Use null if none fits:\n" + "\n".join(lines))


_MASKS = [
    (re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.I), "[URL]"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"), "[EMAIL]"),
    (re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._-]{2,}@[A-Za-z][A-Za-z0-9]{1,20}\b"), "[UPI]"),
    (re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"), "[IFSC]"),
    (re.compile(r"\b0x[a-fA-F0-9]{40}\b"), "[WALLET]"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP]"),
    (re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"), "[PHONE]"),
    (re.compile(r"(?<!\d)\d{9,18}(?!\d)"), "[NUMBER]"),
]


def mask_identifiers_for_llm(text: str) -> str:
    """Replace identifiers with placeholders BEFORE sending text to an external API."""
    for pattern, token in _MASKS:
        text = pattern.sub(token, text)
    return text


# =====================================================================
# SELF-TESTS  (python linkagex_fixes.py)
# =====================================================================

def _run_self_tests() -> None:
    # ---- amounts: the exact sentences from your outputs -----------------
    cases = [
        ("I approved it and ₹8,500 was deducted from my account.", 8500.0, True),
        ("After I installed it, two transactions of ₹3,000 and ₹2,000 were made without my permission.", 5000.0, True),
        ("Soon after, ₹12,000 was debited from my account.", 12000.0, True),
        ("I lost around ₹7,000 in total.", 7000.0, True),
        ("I received a message saying that a payment of ₹4,999 was made from my account.", 4999.0, True),
        ("I followed his instructions on PhonePe and lost ₹6,200.", 6200.0, True),
        ("After doing so, ₹9,100 was taken from my account.", 9100.0, True),
        ("I transferred ₹2,500 and then ₹4,000 to different UPI IDs.", 6500.0, True),
        ("I paid ₹3,000 followed by ₹5,000.", 8000.0, True),
    ]
    for text, expected, loss in cases:
        r = analyze_amounts(text)
        assert r["amount_lost"] == expected and r["loss_occurred"] is loss, (text, r)

    # demand only -> NOT a loss
    r = analyze_amounts("They asked me to pay ₹2,500 as a registration fee.")
    assert r["amount_lost"] is None and r["loss_occurred"] is None
    assert r["amount_details"][0]["role"] == "demanded"

    # demand + later "I paid it" -> inferred, flagged
    r = analyze_amounts("They asked me to pay ₹2,500 as a registration fee. I paid it through UPI.")
    assert r["amount_lost"] == 2500.0 and r["amount_lost_basis"] == "inferred_from_demand"

    # explicit total beats summing the parts
    r = analyze_amounts("I paid ₹3,000 and ₹5,000, a total loss of ₹8,000.")
    assert r["amount_lost"] == 8000.0, r

    # explicit no-loss and negated payment
    assert analyze_amounts("I got an OTP I did not request. No money was lost.")["loss_occurred"] is False
    assert analyze_amounts("I have not paid ₹5,000 yet.")["amount_lost"] is None

    # prize amounts are not losses
    r = analyze_amounts("They said I won a prize of ₹5,00,000 and I paid ₹1,500 as tax.")
    assert r["amount_lost"] == 1500.0, r

    # lakh/crore
    assert analyze_amounts("I lost Rs 5 lakh.")["amount_lost"] == 500000.0

    # ---- organizations ---------------------------------------------------
    assert extract_organizations_v2("I approved the request on Google Pay.") == ["Google Pay"]
    assert extract_organizations_v2("Contacted me on WhatsApp and Telegram.") == []
    orgs = extract_organizations_v2("My Aadhaar and passport were found by FedEx; paid from ICICI Bank to Punjab National Bank. CBI Mumbai.")
    assert orgs == ["FedEx", "ICICI Bank", "Punjab National Bank", "CBI"], orgs

    # ---- groq safety -----------------------------------------------------
    masked = mask_identifiers_for_llm(
        "Call 9988233057 or mail bad@gmail.com, pay fraud@ybl, a/c 3335002100021930 IFSC PUNB0436400 at http://x.xyz/a")
    for leaked in ("9988233057", "bad@gmail.com", "fraud@ybl", "3335002100021930", "PUNB0436400", "x.xyz"):
        assert leaked not in masked, (leaked, masked)

    tax = get_taxonomy({"Financial Fraud": {"UPI/Payment Fraud": []}})
    assert validate_taxonomy("Online Financial Fraud", "x", tax) == (None, None)
    assert validate_taxonomy("Financial Fraud", "Made Up Type", tax) == ("Financial Fraud", None)
    assert validate_taxonomy("Financial Fraud", "UPI/Payment Fraud", tax) == ("Financial Fraud", "UPI/Payment Fraud")

    # ---- semantic fallback: PLUMBING ONLY (stand-in embedder) -----------
    # The real MiniLM model cannot be downloaded in my test sandbox, so this
    # checks control flow (rules first, semantic only on None, threshold,
    # margin, review flag) with a toy bag-of-words embedder. It says nothing
    # about real accuracy: calibrate on your own complaints with explain_semantic().
    import numpy as np

    class _Toy:
        def encode(self, texts, normalize_embeddings=True):
            out = []
            for t in texts:
                v = np.zeros(512)
                for w in re.findall(r"[a-z]+", t.lower()):
                    v[hash(w) % 512] += 1.0
                n = np.linalg.norm(v)
                out.append(v / n if n else v)
            return np.array(out)

    global _MODEL, _PROTO_VECS
    _MODEL, _PROTO_VECS = _Toy(), None

    rules_hit = lambda t: ("Ransomware & Malware", "Ransomware Attack", [{"source": "rule"}])
    rules_miss = lambda t: (None, None, [])

    assert classify_crime_hybrid("anything", rules_hit)[0] == "Ransomware & Malware"   # rules win
    cat, sub, m = classify_crime_hybrid(
        "A caller claiming to be from customs said a parcel in my name had illegal items and demanded money",
        rules_miss)
    assert (cat, sub) == ("Financial Fraud", "Authority/Courier Impersonation"), (cat, sub)
    assert came_from_semantic(m)
    assert classify_crime_hybrid("Something strange happened yesterday", rules_miss)[0] is None

    _MODEL, _PROTO_VECS = None, None
    print("All fix self-tests passed (semantic part = plumbing only).")


if __name__ == "__main__":
    _run_self_tests()