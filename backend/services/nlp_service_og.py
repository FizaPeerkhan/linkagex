"""
LinkageX - NLP Complaint Structuring Service

Input:
    Raw cybercrime complaint text

Output:
    Structured incident used by:
    - Citizen complaint flow
    - Missing-information flow
    - Case repository
    - Linkage engine

This version combines:
- Claude v3 role-aware extraction
- LinkageX incident schema additions
- Indian cybercrime-specific patterns
- Attacker/victim identifier separation
- Attack variant detection
- Event type detection
- Loss detection
- Platform/service detection
- Communication direction
- Confidence and review flags

IMPORTANT:
This is a rule-based baseline.
It should later be evaluated against a human-labelled gold dataset
and compared with a learned Transformer-based classifier.
"""

import re
import warnings
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# OPTIONAL SPACY
# ============================================================

try:
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = None
        warnings.warn(
            "spaCy model 'en_core_web_sm' not installed. "
            "Run: python -m spacy download en_core_web_sm\n"
            "Location extraction will use the Indian gazetteer only."
        )

except ImportError:
    nlp = None
    warnings.warn(
        "spaCy is not installed. "
        "Location extraction will use the Indian gazetteer only."
    )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_complaint_text(text: str) -> str:
    """
    Trim and collapse whitespace while preserving:
    - numbers
    - URLs
    - punctuation
    - identifiers
    """

    text = str(text).strip()

    return re.sub(r"\s+", " ", text)


# ============================================================
# GENERIC PATTERN MATCHER
# ============================================================

def extract_pattern_labels(
    text: str,
    pattern_dictionary: Dict[str, list]
) -> list:
    """
    Return labels whose patterns match.

    Labels are returned in order of their first appearance
    in the complaint text.
    """

    hits = []

    for label, patterns in pattern_dictionary.items():

        positions = []

        for pattern in patterns:

            m = re.search(
                pattern,
                text,
                flags=re.IGNORECASE
            )

            if m:
                positions.append(m.start())

        if positions:
            hits.append(
                (
                    min(positions),
                    label
                )
            )

    hits.sort()

    return [
        label
        for _, label in hits
    ]


# ============================================================
# ROLE ATTRIBUTION
# ============================================================

ROLE_MAX_DISTANCE = 80


ATTACKER_CONTEXT_PATTERNS = [

    r"\b(?:called|contacted|messaged|texted|emailed|whatsapped)"
    r"\b[^.]{0,20}\bfrom\b",

    r"\breceived (?:a |an )?"
    r"(?:call|message|sms|email|text)\b"
    r"[^.]{0,20}\bfrom\b",

    r"\b(?:caller|fraudster|scammer|suspect|accused|culprit|"
    r"beneficiary|recipient|payee|mule|impersonator)(?:'s)?\b",

    r"\bthe (?:number|contact|id)\b"
    r"[^.]{0,15}\b(?:was|is)\b",

    r"\b(?:sent|transferred|paid|deposited|credited|remitted|"
    r"send|transfer|pay)\b"
    r"[^.]{0,40}?\bto\b",

    r"\bto (?:the )?(?:following )?"
    r"(?:account|a/c|upi|wallet)\b",

    r"\b(?:provided|shared|given|sent) by\b",

    r"\bgiven to me\b",

    r"\btheir\b",

    r"\basked me to (?:pay|transfer|send)\b",
]


VICTIM_CONTEXT_PATTERNS = [

    r"\bmy (?:own )?(?:\w+ ){0,2}"
    r"(?:number|phone|mobile|contact|email|e-mail|upi|account|"
    r"a/c|wallet|id|profile|handle|username)\b",

    r"\breach me\b",

    r"\bcontact me (?:at|on)\b",

    r"\bi can be reached\b",

    r"\bcomplainant(?:'s)?\b",

    r"\bvictim(?:'s)?\b",

    r"\b(?:debited|deducted|withdrawn) from\b",
]


INFRA_ATTACKER_CUES = [

    r"\b(?:link|clicked|opened|visited|redirected|fake|phishing|"
    r"suspicious|malicious|website|site|url|portal)\b"
]


SOCIAL_ATTACKER_CUES = [
    r"\bfake\b",
    r"\bimpersonat\w*\b",
    r"\bcreated\b",
]


WALLET_ATTACKER_CUES = [
    r"\bwallet\b"
]


_SENT_SPLIT = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\"'(])"
)


def _sentence_bounds(
    text: str,
    start: int,
    end: int
) -> Tuple[int, int]:
    """
    Return character bounds of the sentence
    containing the specified identifier.
    """

    s = 0

    for m in _SENT_SPLIT.finditer(text):

        if m.end() <= start:
            s = m.end()

        elif m.start() >= end:
            return s, m.start()

    return s, len(text)


def _nearest_cue_distance(
    sentence: str,
    patterns: List[str],
    rel_start: int,
    rel_end: int
) -> Optional[int]:

    best = None

    for pattern in patterns:

        for m in re.finditer(
            pattern,
            sentence,
            flags=re.IGNORECASE
        ):

            if m.end() <= rel_start:

                d = rel_start - m.end()

            elif m.start() >= rel_end:

                d = m.start() - rel_end

            else:

                d = 0

            if (
                d <= ROLE_MAX_DISTANCE
                and (
                    best is None
                    or d < best
                )
            ):
                best = d

    return best


def _attribute_role(
    text: str,
    start: int,
    end: int,
    extra_attacker: Optional[List[str]] = None
) -> str:
    """
    Attribute identifier role.

    Possible values:
        attacker
        victim
        unclear

    Nearest cue wins.

    If there is no reliable cue,
    return unclear rather than guessing.
    """

    s, e = _sentence_bounds(
        text,
        start,
        end
    )

    sentence = text[s:e]

    rs = start - s
    re_ = end - s

    attacker_patterns = (
        ATTACKER_CONTEXT_PATTERNS
        + (extra_attacker or [])
    )

    attacker_distance = _nearest_cue_distance(
        sentence,
        attacker_patterns,
        rs,
        re_
    )

    victim_distance = _nearest_cue_distance(
        sentence,
        VICTIM_CONTEXT_PATTERNS,
        rs,
        re_
    )

    if (
        attacker_distance is None
        and victim_distance is None
    ):
        return "unclear"

    if victim_distance is None:
        return "attacker"

    if attacker_distance is None:
        return "victim"

    if attacker_distance < victim_distance:
        return "attacker"

    if victim_distance < attacker_distance:
        return "victim"

    return "unclear"


def _merge_roles(
    a: str,
    b: str
) -> str:

    if a == b:
        return a

    if a == "unclear":
        return b

    if b == "unclear":
        return a

    if "legitimate" in (a, b):
        return "legitimate"

    # attacker vs victim conflict
    return "unclear"


def _extract_with_role(
    text: str,
    pattern: str,
    group: int = 0,
    transform=None,
    validator=None,
    flags: int = re.IGNORECASE,
    extra_attacker: Optional[List[str]] = None,
    scan_text: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Extract identifiers and attach a role.

    scan_text can be a scrubbed copy of the text,
    while role attribution still uses the original text.
    """

    scan = (
        text
        if scan_text is None
        else scan_text
    )

    found: Dict[str, str] = {}

    for m in re.finditer(
        pattern,
        scan,
        flags=flags
    ):

        raw = m.group(group)

        if raw is None:
            continue

        value = (
            transform(raw)
            if transform
            else raw
        )

        if validator and not validator(value):
            continue

        role = _attribute_role(
            text,
            m.start(group),
            m.end(group),
            extra_attacker
        )

        if value in found:

            found[value] = _merge_roles(
                found[value],
                role
            )

        else:

            found[value] = role

    return [
        {
            "value": value,
            "role": role
        }
        for value, role in found.items()
    ]


def _merge_lists(
    *lists: List[Dict[str, str]]
) -> List[Dict[str, str]]:

    merged: Dict[str, str] = {}

    for lst in lists:

        for item in lst:

            value = item["value"]
            role = item["role"]

            if value in merged:

                merged[value] = _merge_roles(
                    merged[value],
                    role
                )

            else:

                merged[value] = role

    return [
        {
            "value": value,
            "role": role
        }
        for value, role in merged.items()
    ]


def _blank(
    text: str,
    pattern: str,
    flags: int = re.IGNORECASE
) -> str:
    """
    Replace matches with same-length spaces.

    This preserves character offsets for
    role attribution.
    """

    return re.sub(
        pattern,
        lambda m: " " * len(m.group(0)),
        text,
        flags=flags
    )


# ============================================================
# AMOUNT EXTRACTION
# ============================================================

_AMOUNT_SYMBOL_RE = re.compile(
    r"(?:₹|\bRs\.?|\bINR)\s*"
    r"([\d,]+(?:\.\d+)?)"
    r"(?:\s*(lakhs?|lacs?|crores?|cr|k)\b)?",
    re.IGNORECASE
)


_AMOUNT_WORD_RE = re.compile(
    r"\b([\d,]+(?:\.\d+)?)\s*"
    r"(?:rupees|rs)\b",
    re.IGNORECASE
)


_MULTIPLIERS = {
    "lakh": 1e5,
    "lac": 1e5,
    "crore": 1e7,
    "cr": 1e7,
    "k": 1e3,
}


def _to_number(
    raw: str,
    unit: Optional[str] = None
) -> Optional[float]:

    try:

        value = float(
            raw.replace(",", "")
        )

    except ValueError:

        return None

    if unit:

        key = (
            unit.lower()
            .rstrip("s")
        )

        value *= _MULTIPLIERS.get(
            key,
            1
        )

    return value


def extract_amounts(
    text: str
) -> list:
    """
    Examples:

        ₹25,000
        Rs. 30,000
        INR 50000
        Rs 5 lakh
        25,000 rupees
    """

    amounts = []

    for m in _AMOUNT_SYMBOL_RE.finditer(text):

        number = _to_number(
            m.group(1),
            m.group(2)
        )

        if number is not None:
            amounts.append(number)

    for m in _AMOUNT_WORD_RE.finditer(text):

        number = _to_number(
            m.group(1)
        )

        if number is not None:
            amounts.append(number)

    return list(
        dict.fromkeys(amounts)
    )


# ============================================================
# CHANNELS
# ============================================================

CHANNEL_PATTERNS = {

    "WhatsApp": [
        r"\bwhatsapp\b"
    ],

    "Instagram": [
        r"\binstagram\b",
        r"\binsta\b"
    ],

    "Facebook": [
        r"\bfacebook\b",
        r"\bfb\b"
    ],

    "Twitter": [
        r"\btwitter\b",
        r"\bx\.com\b"
    ],

    "Telegram": [
        r"\btelegram\b"
    ],

    "Skype": [
        r"\bskype\b"
    ],

    "Email": [
        r"\bemail\b",
        r"\be-mail\b",
        r"\bemailed\b"
    ],

    "SMS": [
        r"\bsms\b",
        r"\btext message\b",
        r"\btexted me\b"
    ],

    "Phone Call": [
        r"\bphone call\b",
        r"\bcalled me\b",
        r"\bcall from\b",
        r"\breceived a call\b",
        r"\btelephone\b"
    ],

    "Website": [
        r"\bwebsite\b",
        r"\bweb site\b",
        r"\bwebpage\b"
    ],

    "Mobile App": [
        r"\bmobile app\b",
        r"\bapplication\b",
        r"\bapp\b"
    ],
}


_SPECIFIC_PLATFORMS = [
    "WhatsApp",
    "Instagram",
    "Facebook",
    "Twitter",
    "Telegram",
    "Skype"
]


def extract_channels(
    text: str
) -> list:

    t = text.lower()

    detected = []

    for channel in (
        _SPECIFIC_PLATFORMS
        + [
            "Email",
            "SMS",
            "Phone Call",
            "Website"
        ]
    ):

        if any(
            re.search(
                pattern,
                t
            )
            for pattern
            in CHANNEL_PATTERNS[channel]
        ):
            detected.append(channel)

    if (
        not any(
            p in detected
            for p in _SPECIFIC_PLATFORMS
        )
        and any(
            re.search(
                p,
                t
            )
            for p
            in CHANNEL_PATTERNS["Mobile App"]
        )
    ):
        detected.append("Mobile App")

    return detected


# ============================================================
# PAYMENT METHODS
# ============================================================

PAYMENT_METHOD_PATTERNS = {

    "UPI": [
        r"\bpaid\b.{0,50}\bupi\b",
        r"\bpayment\b.{0,50}\bupi\b",
        r"\btransferred\b.{0,50}\bupi\b",
        r"\bupi payment\b",
        r"\busing upi\b",
        r"\bthrough upi\b",
        r"\bvia upi\b",
        r"\bupi id\b.{0,50}\bpaid\b",
        r"\bupi id\b.{0,50}\bpayment\b",
        r"\bphonepe\b.{0,50}\bpaid\b",
        r"\bgoogle pay\b.{0,50}\bpaid\b",
        r"\bgpay\b.{0,50}\bpaid\b",
        r"\bpaytm\b.{0,50}\bpaid\b",
    ],

    "Credit Card": [
        r"\bpaid\b.{0,50}\bcredit card\b",
        r"\bpayment\b.{0,50}\bcredit card\b",
        r"\busing (?:my )?credit card\b",
        r"\bthrough (?:my )?credit card\b",
    ],

    "Debit Card": [
        r"\bpaid\b.{0,50}\bdebit card\b",
        r"\bpayment\b.{0,50}\bdebit card\b",
        r"\busing (?:my )?debit card\b",
        r"\bthrough (?:my )?debit card\b",
    ],

    "Bank Transfer": [
        r"\bbank transfer\b",
        r"\btransferred\b.{0,50}\bto (?:my )?bank\b",
        r"\baccount transfer\b",
        r"\btransferred the money\b.{0,50}\bbank\b",
        r"\b(?:via|through|using|by) "
        r"(?:rtgs|neft|imps)\b",
    ],

    "Cash": [
        r"\bpaid\b.{0,30}\bcash\b",
        r"\bpaid in cash\b",
        r"\bgave\b.{0,30}\bcash\b"
    ],

    "Cryptocurrency": [
        r"\bpaid\b.{0,50}\bbitcoin\b",
        r"\bpaid\b.{0,50}\bcrypto\b",
        r"\btransferred\b.{0,50}\bcrypto\b",
        r"\busing bitcoin\b",
        r"\busing cryptocurrency\b",
        r"\busing crypto\b",
        r"\bethereum payment\b",
    ],

    "Digital Wallet": [
        r"\bpaid\b.{0,50}\bdigital wallet\b",
        r"\bpayment\b.{0,50}\bdigital wallet\b",
        r"\busing (?:a )?digital wallet\b",
        r"\busing (?:an )?e-wallet\b",
    ],
}


def extract_payment_methods(
    text: str
) -> list:

    return extract_pattern_labels(
        text,
        PAYMENT_METHOD_PATTERNS
    )


# ============================================================
# IDENTIFIERS
# ============================================================

# ------------------------------------------------------------
# PHONE
# ------------------------------------------------------------

_PHONE_RE = (
    r"(?<!\d)"
    r"(?:\+?91[\s-]?)?"
    r"[6-9]\d{4}[\s-]?\d{5}"
    r"(?!\d)"
)


def _norm_phone(
    value: str
) -> str:

    return re.sub(
        r"\D",
        "",
        value
    )[-10:]


def extract_phones(
    text: str
) -> List[Dict[str, str]]:

    return _extract_with_role(
        text,
        _PHONE_RE,
        transform=_norm_phone
    )


# ------------------------------------------------------------
# EMAIL
# ------------------------------------------------------------

_EMAIL_RE = (
    r"(?<![A-Za-z0-9._%+-])"
    r"[A-Za-z0-9._%+-]+@"
    r"[A-Za-z0-9-]+"
    r"(?:\.[A-Za-z0-9-]+)*"
    r"\.[A-Za-z]{2,}"
)


def extract_emails(
    text: str
) -> List[Dict[str, str]]:

    return _extract_with_role(
        text,
        _EMAIL_RE,
        transform=str.lower
    )


# ------------------------------------------------------------
# UPI
# ------------------------------------------------------------

KNOWN_UPI_SUFFIXES = {

    "ybl",
    "okhdfcbank",
    "oksbi",
    "okicici",
    "okaxis",
    "paytm",
    "ibl",
    "axl",
    "apl",
    "upi",
    "yapl",
    "airtel",
    "fbl",
    "idfcbank",
    "jio",
    "kotak",
    "rbl",
    "uco",
    "waaxis",
    "waicici",
    "wahdfcbank",
    "wasbi",
    "yesbank",
    "boi",
    "cnrb",
    "dbs",
    "federal",
    "indus",
    "axisbank",
    "sbi",
    "icici",
    "hdfcbank",
    "pnb",
    "aubank",
    "ikwik",
    "freecharge",
}


_UPI_RE = (
    r"(?<![A-Za-z0-9._%+-])"
    r"[A-Za-z0-9._-]{2,}"
    r"@[A-Za-z][A-Za-z0-9]{1,20}"
    r"\b(?!\.[A-Za-z])"
)


def _plausible_upi(
    value: str
) -> bool:

    return (
        "@"
        in value
        and value.rsplit(
            "@",
            1
        )[-1].lower()
        in KNOWN_UPI_SUFFIXES
    )


def extract_upi_ids(
    text: str
) -> List[Dict[str, str]]:

    return _extract_with_role(
        text,
        _UPI_RE,
        transform=str.lower,
        validator=_plausible_upi
    )


# ------------------------------------------------------------
# URL / DOMAIN
# ------------------------------------------------------------

_URL_RE = (
    r"(?:https?://|www\.)"
    r"[^\s<>\"']+"
)


_DOMAIN_RE = (
    r"(?<![A-Za-z0-9@._-])"
    r"(?:[A-Za-z0-9-]+\.)+"
    r"(?:co\.in|gov\.in|com|in|org|net|info|xyz|online|"
    r"site|top|club|shop|store|app|link|cc|me|io|biz|live|vip)"
    r"(?![A-Za-z0-9-])"
)


KNOWN_LEGIT_DOMAINS = {

    "sbi.co.in",
    "onlinesbi.sbi",
    "hdfcbank.com",
    "icicibank.com",
    "axisbank.com",
    "pnbindia.in",

    "google.com",
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "microsoft.com",

    "whatsapp.com",
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "telegram.org",

    "amazon.in",
    "amazon.com",
    "flipkart.com",
    "paytm.com",
    "phonepe.com",
    "npci.org.in",

    "rbi.org.in",
    "cybercrime.gov.in",
    "uidai.gov.in",
    "incometax.gov.in",
}


def _clean_url(
    value: str
) -> str:

    return value.rstrip(
        ".,!?;:)]}'\""
    )


def _domain_of(
    url: str
) -> str:

    domain = re.sub(
        r"^(?:https?://)?(?:www\.)?",
        "",
        url,
        flags=re.IGNORECASE
    )

    return re.split(
        r"[/?#:]",
        domain,
        maxsplit=1
    )[0].lower()


def _is_legit_domain(
    domain: str
) -> bool:

    d = domain.lower()

    if (
        d == "gov.in"
        or d.endswith(".gov.in")
    ):
        return True

    return any(
        d == legitimate
        or d.endswith("." + legitimate)
        for legitimate
        in KNOWN_LEGIT_DOMAINS
    )


def extract_urls(
    text: str
) -> List[Dict[str, str]]:

    items = _extract_with_role(
        text,
        _URL_RE,
        transform=_clean_url,
        extra_attacker=INFRA_ATTACKER_CUES
    )

    for item in items:

        if _is_legit_domain(
            _domain_of(item["value"])
        ):
            item["role"] = "legitimate"

    return items


def extract_domains(
    text: str
) -> List[Dict[str, str]]:

    from_urls = [

        {
            "value": _domain_of(
                item["value"]
            ),
            "role": item["role"]
        }

        for item
        in extract_urls(text)
    ]

    scrubbed = _blank(
        _blank(
            text,
            _EMAIL_RE
        ),
        _URL_RE
    )

    standalone = _extract_with_role(
        text,
        _DOMAIN_RE,
        transform=str.lower,
        extra_attacker=INFRA_ATTACKER_CUES,
        scan_text=scrubbed
    )

    for item in standalone:

        if item["value"].startswith("www."):
            item["value"] = item["value"][4:]

    merged = _merge_lists(
        from_urls,
        standalone
    )

    for item in merged:

        if _is_legit_domain(
            item["value"]
        ):
            item["role"] = "legitimate"

    return merged


# ------------------------------------------------------------
# BANK ACCOUNT / IFSC / TRANSACTION ID
# ------------------------------------------------------------

_BANK_ACCOUNT_RE = (
    r"\b(?:bank\s+)?"
    r"(?:account|a/c|acct)"
    r"(?:\s*(?:number|no\.?|num|#))?"
    r"\s*(?:is|was|[:\-])?"
    r"\s*(\d{9,18})(?!\d)"
)


_IFSC_RE = (
    r"\b[A-Z]{4}0[A-Z0-9]{6}\b"
)


_TXN_RE = (
    r"\b(?:transaction|txn|trans|reference|ref|utr)"
    r"\s*(?:id|no\.?|number|num)?"
    r"\s*(?:is|was|[:\-#])?"
    r"\s*((?=[A-Za-z0-9]*\d)"
    r"[A-Za-z0-9]{8,30})\b"
)


def extract_bank_accounts(
    text: str
) -> List[Dict[str, str]]:

    return _extract_with_role(
        text,
        _BANK_ACCOUNT_RE,
        group=1
    )


def extract_ifsc_codes(
    text: str
) -> List[Dict[str, str]]:

    return _extract_with_role(
        text,
        _IFSC_RE,
        flags=0
    )


def extract_transaction_ids(
    text: str
) -> list:
    """
    Transaction IDs are victim-side payment references.

    They should NOT be used as attacker linkage identifiers.
    """

    ids = [

        m.group(1)

        for m in re.finditer(
            _TXN_RE,
            text,
            flags=re.IGNORECASE
        )
    ]

    return list(
        dict.fromkeys(ids)
    )


# ------------------------------------------------------------
# IP / CRYPTO WALLETS
# ------------------------------------------------------------

_IP_RE = (
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
)


_ETH_RE = (
    r"\b0x[a-fA-F0-9]{40}\b"
)


_BTC_RE = (
    r"\b(?:bc1[a-z0-9]{25,60}|"
    r"[13][a-km-zA-HJ-NP-Z1-9]{25,34})\b"
)


def extract_ip_addresses(
    text: str
) -> List[Dict[str, str]]:

    return _extract_with_role(
        text,
        _IP_RE,
        extra_attacker=INFRA_ATTACKER_CUES
    )


def _mixed_alnum(
    value: str
) -> bool:

    return (
        bool(re.search(r"[A-Za-z]", value))
        and
        bool(re.search(r"\d", value))
    )


def extract_crypto_wallets(
    text: str
) -> List[Dict[str, str]]:

    ethereum = _extract_with_role(
        text,
        _ETH_RE,
        transform=str.lower,
        flags=0,
        extra_attacker=WALLET_ATTACKER_CUES
    )

    bitcoin = _extract_with_role(
        text,
        _BTC_RE,
        flags=0,
        validator=_mixed_alnum,
        extra_attacker=WALLET_ATTACKER_CUES
    )

    return _merge_lists(
        ethereum,
        bitcoin
    )


# ------------------------------------------------------------
# SOCIAL MEDIA IDs
# ------------------------------------------------------------

_SOCIAL = (
    r"(?:instagram|insta|facebook|fb|telegram|"
    r"twitter|snapchat|skype)"
)


_SOCIAL_LABELLED_RE = (
    rf"\b{_SOCIAL}\b\s*"
    r"(?:id|username|user name|handle)"
    r"\s*(?:is|was|[:\-])?"
    r"\s*@?"
    r"([A-Za-z0-9._]{3,30})\b"
)


_SOCIAL_AT_RE = (
    rf"\b{_SOCIAL}\b"
    r"[^.\n@]{0,25}?"
    r"(?<![A-Za-z0-9._])"
    r"@([A-Za-z0-9._]{3,30})\b"
)


def extract_social_media_ids(
    text: str
) -> List[Dict[str, str]]:

    labelled = _extract_with_role(
        text,
        _SOCIAL_LABELLED_RE,
        group=1,
        transform=str.lower,
        validator=lambda value: bool(
            re.search(
                r"[\d._]",
                value
            )
        ),
        extra_attacker=SOCIAL_ATTACKER_CUES
    )

    at_handle = _extract_with_role(
        text,
        _SOCIAL_AT_RE,
        group=1,
        transform=str.lower,
        extra_attacker=SOCIAL_ATTACKER_CUES
    )

    return _merge_lists(
        labelled,
        at_handle
    )
# ============================================================
# ORGANIZATIONS
# ============================================================

ORGANIZATION_NORMALIZATION = {

    "HDFC": "HDFC Bank",
    "HDFC Bank": "HDFC Bank",

    "ICICI": "ICICI Bank",
    "ICICI Bank": "ICICI Bank",

    "PNB": "Punjab National Bank",
    "Punjab National Bank": "Punjab National Bank",

    "SBI": "SBI",
    "State Bank of India": "SBI",

    "Amazon Pay": "Amazon Pay",
    "Amazon": "Amazon",

    "PhonePe": "PhonePe",
    "Paytm": "Paytm",
    "Google Pay": "Google Pay",
    "Freecharge": "Freecharge",
    "Flipkart": "Flipkart",
    "Swiggy": "Swiggy",

    "FedEx": "FedEx",
    "DHL": "DHL",
    "Axis Bank": "Axis Bank",
    "Bank of Baroda": "Bank of Baroda",

    "RBI": "RBI",
    "Reserve Bank of India": "RBI",
    "CBI": "CBI",
    "UIDAI": "UIDAI",
    "Aadhaar": "UIDAI",
    "Income Tax Department": "Income Tax Department",
    "Income Tax": "Income Tax Department",

    "Microsoft": "Microsoft",
    "Google": "Google",
    "Meta": "Meta",
    "Instagram": "Instagram",
    "Facebook": "Facebook",
    "Telegram": "Telegram",
    "WhatsApp": "WhatsApp",
    "Skype": "Skype",

    "Amazon": "Amazon",
    "Flipkart": "Flipkart",
    "Myntra": "Myntra",
    "Swiggy": "Swiggy",
    "Zomato": "Zomato",
}


def extract_organizations(
    text: str
) -> list:

    detected = []

    sorted_orgs = sorted(
        ORGANIZATION_NORMALIZATION.keys(),
        key=len,
        reverse=True
    )

    for organization in sorted_orgs:

        pattern = (
            r"\b"
            + re.escape(organization)
            + r"\b"
        )

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        ):

            canonical_name = (
                ORGANIZATION_NORMALIZATION[
                    organization
                ]
            )

            if canonical_name not in detected:

                detected.append(
                    canonical_name
                )

    return detected


# ============================================================
# LOCATIONS
# ============================================================

LOCATION_CONTEXT_PATTERNS = [

    r"\bin ([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b",

    r"\bat ([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b",

    r"\bfrom ([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b",

    r"\bnear ([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b",

    r"\blocated in ([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b",

    r"\bbased in ([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b",

    r"\bincident happened in ([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b",

    r"\bincident occurred in ([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b",
]


LOCATION_FALSE_POSITIVES = {

    "netflix",
    "amazon",
    "paytm",
    "phonepe",
    "facebook",
    "instagram",
    "twitter",
    "telegram",
    "whatsapp",
    "skype",
    "fedex",
}


INDIAN_STATES_UTS = [

    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chhattisgarh",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
    "Delhi",
    "Jammu and Kashmir",
    "Ladakh",
    "Chandigarh",
    "Puducherry",
]


INDIAN_CITIES = [

    "Mumbai",
    "Navi Mumbai",
    "Pune",
    "Nagpur",
    "Thane",
    "Nashik",
    "Kalyan",
    "New Delhi",
    "Bengaluru",
    "Bangalore",
    "Chennai",
    "Hyderabad",
    "Kolkata",
    "Ahmedabad",
    "Surat",
    "Vadodara",
    "Rajkot",
    "Jaipur",
    "Lucknow",
    "Kanpur",
    "Indore",
    "Bhopal",
    "Patna",
    "Ranchi",
    "Noida",
    "Gurgaon",
    "Gurugram",
    "Ghaziabad",
    "Faridabad",
    "Ludhiana",
    "Amritsar",
    "Kochi",
    "Thiruvananthapuram",
    "Coimbatore",
    "Visakhapatnam",
    "Vijayawada",
    "Bhubaneswar",
    "Guwahati",
    "Dehradun",
    "Raipur",
    "Mysuru",
    "Mangaluru",
]


_GAZETTEER = sorted(
    set(
        INDIAN_STATES_UTS
        + INDIAN_CITIES
    ),
    key=len,
    reverse=True
)


# Do not treat the city mentioned as part of a fake authority
# as the victim's location.
_AUTHORITY_SUFFIX = (
    r"(?!\s+(?:police|crime\s+branch|cyber\s*"
    r"(?:cell|crime)|customs|court|high\s+court|cbi))"
)


def _gazetteer_locations(
    text: str
) -> List[str]:

    found = []
    taken = []

    for name in _GAZETTEER:

        pattern = (
            r"\b"
            + re.escape(name)
            + r"\b"
            + _AUTHORITY_SUFFIX
        )

        for m in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE
        ):

            # Already covered by a longer location
            if any(
                m.start() >= start
                and m.end() <= end
                for start, end
                in taken
            ):
                continue

            taken.append(
                (
                    m.start(),
                    m.end()
                )
            )

            found.append(
                (
                    m.start(),
                    name
                )
            )

    found.sort()

    return [
        name
        for _, name
        in found
    ]


def extract_locations(
    text: str
) -> list:
    """
    Extract Indian cities/states from:
    1. Gazetteer
    2. spaCy GPE/LOC entities supported by
       explicit location context
    """

    locations = _gazetteer_locations(
        text
    )

    if nlp is not None:

        captures = set()

        for pattern in LOCATION_CONTEXT_PATTERNS:

            for m in re.finditer(
                pattern,
                text
            ):

                captures.add(
                    m.group(1)
                )

        doc = nlp(text)

        for ent in doc.ents:

            candidate = ent.text.strip()

            if (
                ent.label_ in (
                    "GPE",
                    "LOC"
                )
                and candidate.lower()
                not in LOCATION_FALSE_POSITIVES
                and candidate in captures
            ):

                locations.append(
                    candidate
                )

    seen = set()
    output = []

    for location in locations:

        key = location.lower()

        if key not in seen:

            seen.add(key)

            output.append(
                location
            )

    return output


# ============================================================
# DECEPTION
# ============================================================

DECEPTION_PATTERNS = {

    "Fake": [

        r"\bfake\b",
        r"\bforged\b",
        r"\bfraudulent\b",
        r"\bfake website\b",
        r"\bfake account\b",
        r"\bfake profile\b",
        r"\bfake offer\b",
    ],

    "Impersonation": [

        r"\bpretending to be\b",
        r"\bposing as\b",
        r"\bclaimed to be\b",
        r"\bimpersonat(?:e|ed|ing)\b",
        r"\bpretended to be\b",
    ],

    "Deceptive Offer": [

        r"\bpromised\b",
        r"\bguaranteed\b",
        r"\bprocessing fee\b",
        r"\bprize\b",
        r"\blottery\b",
        r"\bhigh returns\b",
        r"\b300% returns\b",
        r"\bregistration fee\b",
        r"\bguaranteed returns\b",
    ],

    "Scam": [

        r"\bscam\b",
        r"\bscammer\b",
        r"\bscamming\b",
        r"\bponzi scheme\b",
    ],
}


def extract_deception(
    text: str
) -> list:

    return extract_pattern_labels(
        text,
        DECEPTION_PATTERNS
    )


# ============================================================
# VICTIM ACTION
# ============================================================

VICTIM_ACTION_PATTERNS = {

    "Paid": [

        r"\bi paid\b",
        r"\bi have paid\b",
        r"\bi had paid\b",
        r"\bwe paid\b",
        r"\bpayment was made\b",
        r"\bmade the payment\b",
        r"\bmade a payment\b",
        r"\bpaid the amount\b",
        r"\bpaid ₹",
        r"\bpaid rs\.?",
        r"\bpaid inr\b",
    ],

    "Transferred": [

        r"\bi transferred\b",
        r"\bi have transferred\b",
        r"\bi had transferred\b",
        r"\bwe transferred\b",
        r"\btransferred the amount\b",
        r"\btransferred the money\b",
        r"\bsent the money\b",
        r"\bi sent ₹",
        r"\bi sent rs\.?",
        r"\bi sent inr\b",
    ],

    "Shared OTP": [

        r"\bshared\b.{0,30}\botp\b",
        r"\bgave\b.{0,30}\botp\b",
        r"\bprovided\b.{0,30}\botp\b",
        r"\btold them my otp\b",
    ],

    "Entered Details": [

        r"\bentered\b.{0,40}\bdetails\b",
        r"\bentered my\b",
        r"\bprovided my\b.{0,40}\bdetails\b",
        r"\bfilled in\b.{0,40}\bdetails\b",
    ],

    "Clicked Link": [

        r"\bclicked\b.{0,20}\blink\b",
        r"\bclicked on\b",
        r"\bclicked the link\b",
    ],

    "Invested": [

        r"\bi invested\b",
        r"\bi had invested\b",
        r"\bwe invested\b",
        r"\binvested ₹",
        r"\binvested rs\.?",
        r"\binvested inr\b",
    ],

    "Approved": [

        r"\bi approved\b",
        r"\bapproved the payment\b",
        r"\bi authorized the payment\b",
    ],

    "Scanned QR": [

        r"\bscanned\b.{0,20}\bqr\b",
        r"\bscan\b.{0,20}\bqr\b",
    ],

    "Downloaded": [

        r"\bi downloaded\b",
        r"\bi installed\b",
        r"\bdownloaded the\b",
        r"\binstalled the\b",
    ],

    "Shared Credentials": [

        r"\bshared\b.{0,30}\bpassword\b",
        r"\bshared\b.{0,30}\bpin\b",
        r"\bshared\b.{0,30}\bcredentials\b",
        r"\bprovided\b.{0,30}\bpassword\b",
        r"\bprovided\b.{0,30}\bcredentials\b",
    ],
}


_DEMAND_PATTERNS = [

    r"\basked me to pay\b",
    r"\basked us to pay\b",
    r"\bdemanded payment\b",
    r"\bdemanded ₹",
    r"\bdemanded rs\.?",
    r"\bdemanded inr\b",
    r"\basked for payment\b",
    r"\basked for ₹",
    r"\basked for rs\.?",
    r"\basked for inr\b",
    r"\bunless paid\b",
    r"\bpay or\b",
    r"\bpay otherwise\b",
]


_ACTUAL_PAYMENT_PATTERNS = [

    r"\bi paid\b",
    r"\bwe paid\b",
    r"\bpayment was made\b",
    r"\bmade the payment\b",
    r"\bpaid the amount\b",
]


def extract_victim_actions(
    text: str
) -> list:

    detected = extract_pattern_labels(
        text,
        VICTIM_ACTION_PATTERNS
    )

    t = text.lower()

    demand_only = any(
        re.search(
            pattern,
            t
        )
        for pattern
        in _DEMAND_PATTERNS
    )

    actual_payment = any(
        re.search(
            pattern,
            t
        )
        for pattern
        in _ACTUAL_PAYMENT_PATTERNS
    )

    # A payment demand alone does not mean
    # that the victim actually paid.
    if (
        demand_only
        and not actual_payment
        and "Paid" in detected
    ):

        detected.remove(
            "Paid"
        )

    return detected


# ============================================================
# ATTACKER ACTION
# ============================================================

ATTACKER_ACTION_PATTERNS = {

    "Account Takeover": [

        r"\baccount was taken over\b",
        r"\btook over my account\b",
        r"\baccount takeover\b",
        r"\btaken over my account\b",
        r"\bgained control of my account\b",
        r"\bgained access to my account\b",
    ],

    "Identity Misuse": [

        r"\bused my identity\b",
        r"\bmisused my identity\b",
        r"\bidentity was misused\b",
        r"\bused my aadhaar\b",
        r"\bmisused my aadhaar\b",
        r"\bused my pan\b",
        r"\bmisused my pan\b",
    ],

    "Hacked": [

        r"\bhacked\b",
        r"\bhacking\b",
        r"\bunauthorized access\b",
        r"\bserver was compromised\b",
        r"\bsystem was compromised\b",
        r"\bcomputer was compromised\b",
    ],

    "Threatened": [

        r"\bthreatened\b",
        r"\bthreatening\b",
        r"\bthreat\b",
        r"\bblackmail(?:ed|ing)?\b",
        r"\bdelete everything unless\b",
        r"\bunless paid\b",
    ],

    "Posted": [

        r"\bposted\b",
        r"\buploaded\b",
        r"\bshared\b.{0,30}\bonline\b",
        r"\bpublished\b",
    ],

    "Stole": [

        r"\bstole\b",
        r"\bstolen\b",
        r"\btheft\b",
        r"\bstole my\b",
    ],

    "Requested Payment": [

        r"\basked me to pay\b",
        r"\basked us to pay\b",
        r"\basked for payment\b",
        r"\basked for a payment\b",
        r"\brequested payment\b",
        r"\bdemanded payment\b",
        r"\bdemanded a payment\b",
        r"\basked for\b.{0,30}\bprocessing fee\b",
        r"\bdemanded\b.{0,30}\bprocessing fee\b",
    ],
}


def extract_attacker_actions(
    text: str
) -> list:

    detected = extract_pattern_labels(
        text,
        ATTACKER_ACTION_PATTERNS
    )

    # Specific actions beat generic actions.

    if (
        "Account Takeover" in detected
        and "Hacked" in detected
    ):

        detected.remove(
            "Hacked"
        )

    if (
        "Identity Misuse" in detected
        and "Stole" in detected
    ):

        detected.remove(
            "Stole"
        )

    return detected


# ============================================================
# OUTCOMES
# ============================================================

OUTCOME_PATTERNS = {

    "Financial Loss": [

        r"\blost\b.{0,30}\bmoney\b",
        r"\blost ₹",
        r"\blost rs\.?",
        r"\blost inr\b",
        r"\btotal loss\b",
        r"\bfinancial loss\b",
        r"\bmoney was deducted\b",
        r"\bamount was deducted\b",
        r"\bfunds were deducted\b",
        r"\baccount was debited\b",
        r"\baccount was charged\b",
        r"\bamount was debited\b",
        r"\bmoney was debited\b",
        r"\bunauthori[sz]ed charges\b",
        r"\bunauthori[sz]ed transactions?\b",
        r"\bfraudulent transactions?\b",
        r"\bunauthori[sz]ed payment\b",
        r"\bmoney was stolen\b",
        r"\bfunds were stolen\b",
    ],

    "Account Compromised": [

        r"\baccount was compromised\b",
        r"\baccount compromised\b",
        r"\bmy account was hacked\b",
        r"\baccount was hacked\b",
        r"\bsecurity settings\b",
        r"\bsomeone gained access to my account\b",
    ],

    "Account Blocked": [

        r"\baccount\b.{0,20}\bblocked\b",
        r"\bblocked my account\b",
        r"\baccount has been blocked\b",
    ],

    "Scammer Stopped Responding": [

        r"\bstopped responding\b",
        r"\bstopped replying\b",
        r"\bstopped answering\b",
        r"\bno longer responds?\b",
        r"\bthey disappeared\b",
        r"\bscammer disappeared\b",
    ],

    "Product/Service Not Received": [

        r"\bnever received\b",
        r"\breceived nothing\b",
        r"\bproduct\b.{0,30}\bnot received\b",
        r"\bservice\b.{0,30}\bnot received\b",
        r"\bdid not receive\b",
        r"\bnot delivered\b",
        r"\bfailed to deliver\b",
    ],

    "Data Exposed": [

        r"\bdata was exposed\b",
        r"\bdata exposed\b",
        r"\bdata was leaked\b",
        r"\bdata leaked\b",
        r"\bpersonal information\b.{0,30}\bleaked\b",
        r"\bpersonal information\b.{0,30}\bexposed\b",
        r"\bdetails\b.{0,30}\bleaked\b",
        r"\binformation\b.{0,40}\bleaked online\b",
        r"\bcustomer data\b.{0,40}\bexposed\b",
    ],

    "Data Encrypted": [

        r"\bfiles are encrypted\b",
        r"\bfiles were encrypted\b",
        r"\bdata is encrypted\b",
        r"\bdata was encrypted\b",
        r"\bdatabase is encrypted\b",
        r"\bdatabase was encrypted\b",
        r"\bdatabase encrypted\b",
        r"\bdocuments are locked\b",
        r"\bfiles are locked\b",
    ],

    "Device Infected": [

        r"\bdevice\b.{0,30}\binfected\b",
        r"\bcomputer\b.{0,30}\binfected\b",
        r"\blaptop\b.{0,30}\binfected\b",
        r"\bphone\b.{0,30}\binfected\b",
        r"\bmalware\b",
        r"\bvirus\b.{0,30}\binfected\b",
    ],

    "Identity Theft": [

        r"\bidentity theft\b",
        r"\bidentity was stolen\b",
        r"\bused my identity\b",
        r"\bmisused my identity\b",
        r"\bmisused my aadhaar\b",
        r"\bused my aadhaar\b",
        r"\bmisused my pan\b",
        r"\bused my pan\b",
    ],
}


def extract_outcomes(
    text: str
) -> list:

    return extract_pattern_labels(
        text,
        OUTCOME_PATTERNS
    )


# ============================================================
# MODUS OPERANDI
# ============================================================

MO_PATTERNS = {

    "Fake Processing Fee": [
        r"\bprocessing fee\b"
    ],

    "Fake Investment Returns": [
        r"\bguaranteed returns\b",
        r"\bhigh returns\b",
        r"\b300% returns\b"
    ],

    "KYC Verification Scam": [
        r"\bkyc\b",
        r"\bverify\b.{0,30}\baccount\b"
    ],

    "Fake Loan Offer": [
        r"\bloan\b.{0,40}\bprocessing fee\b"
    ],

    "Fake Shopping": [
        r"\bfake shopping\b",
        r"\bsite disappeared\b",
        r"\bproduct\b.{0,30}\bnever received\b"
    ],

    "QR Payment Fraud": [
        r"\bqr code\b.{0,50}\bdeducted\b"
    ],

    "OTP Scam": [
        r"\botp\b"
    ],

    "Ransomware Encryption": [
        r"\bransomware\b",
        r"\bfiles are encrypted\b",
        r"\bdatabase is encrypted\b"
    ],
}


def extract_mo(
    text: str
) -> list:

    return extract_pattern_labels(
        text,
        MO_PATTERNS
    )


# ============================================================
# CRIME CLASSIFICATION
# ============================================================

_SOC = (
    r"(?:facebook|instagram|twitter|telegram|"
    r"whatsapp|snapchat)"
)


PRIORITY_RULES = [

    (
        "Ransomware & Malware",
        "Ransomware Attack",
        r"\bransomware\b|"
        r"\bfiles?\s+(?:are|were)\s+encrypted\b|"
        r"\bdatabase\s+(?:is|was)\s+encrypted\b|"
        r"\bdata\s+(?:is|was)\s+encrypted\b"
    ),

    (
        "Ransomware & Malware",
        "Malware",
        r"\bmalware\b|"
        r"\btrojan\b|"
        r"\bspyware\b|"
        r"\bkeylogger\b|"
        r"\bvirus\b"
    ),

    (
        "Social Media Crimes",
        "Morphed Images/Deepfakes",
        r"\bdeepfake\b|"
        r"\bmorphed images?\b|"
        r"\bmorphed photos?\b|"
        r"\bsynthetic video\b"
    ),

    (
        "Social Media Crimes",
        "Fake Accounts/Impersonation",
        r"\bfake profile\b|"
        r"\bfake account\b|"
        r"\bcreated a fake\b|"
        r"\bimpersonat(?:e|ed|ing)\b|"
        r"\bpretend(?:ing|ed) to be me\b"
    ),

    (
        "Social Media Crimes",
        "Email/Social Media Hacking",
        rf"\b(?:hacked|hack|compromised|taken over)\b"
        rf".{{0,50}}\b{_SOC}\b|"
        rf"\b{_SOC}\b"
        rf".{{0,50}}\b(?:hacked|hack|compromised|taken over)\b"
    ),

    (
        "Hacking & Unauthorized Access",
        "Account Takeover",
        r"\baccount was taken over\b|"
        r"\btook over my account\b|"
        r"\btaken over by someone\b|"
        r"\baccount takeover\b|"
        r"\bgained control of my account\b|"
        r"\bgained access to my account\b"
    ),

    (
        "Hacking & Unauthorized Access",
        "Identity Theft",
        r"\bidentity theft\b|"
        r"\bidentity was stolen\b|"
        r"\bidentity was misused\b|"
        r"\bused my identity\b|"
        r"\bmisused my identity\b|"
        r"\bused my aadhaar\b|"
        r"\bmisused my aadhaar\b|"
        r"\bused my pan\b|"
        r"\bmisused my pan\b"
    ),

    (
        "Hacking & Unauthorized Access",
        "Data Breach",
        r"\bdata breach\b|"
        r"\bdata was breached\b|"
        r"\bdata was exposed\b|"
        r"\bcustomer data\b.{0,50}\bexposed\b|"
        r"\bpersonal information\b.{0,50}\bexposed\b|"
        r"\bdata leak\b|"
        r"\bdata was leaked\b"
    ),

    (
        "Financial Fraud",
        "Credit Card Fraud",
        r"\bcredit card\b.{0,60}"
        r"\b(?:unauthori[sz]ed|fraudulent|charges|"
        r"transactions?|used|stolen)\b"
    ),

    (
        "Financial Fraud",
        "Cryptocurrency Scams",
        r"\b(?:cryptocurrency|crypto|bitcoin|ethereum)\b"
        r".{0,80}"
        r"\b(?:invested|investment|trading|exchange|"
        r"scam|fraud|stole|stolen|lost|transferred|wallet)\b"
        r"|"
        r"\b(?:invested|investment|transferred|lost|"
        r"stole|stolen)\b.{0,80}"
        r"\b(?:cryptocurrency|crypto|bitcoin|ethereum)\b"
    ),

    (
        "Financial Fraud",
        "Online Gambling/Betting",
        r"\bonline gambling\b|"
        r"\bonline casino\b|"
        r"\bonline betting\b|"
        r"\bcricket betting\b|"
        r"\bbetting app\b|"
        r"\bgambling app\b|"
        r"\bcasino\b|"
        r"\bbetting\b"
    ),

    (
        "E-Commerce & Job Fraud",
        "Education/Course Scams",
        r"\bstudy material\b|"
        r"\bskill development course\b|"
        r"\bonline degree\b|"
        r"\beducation course\b|"
        r"\bonline course\b|"
        r"\btraining course\b|"
        r"\bcourse fee\b"
    ),

    (
        "E-Commerce & Job Fraud",
        "Fake Shopping Websites",
        r"\bfake shopping\b|"
        r"\bfake online store\b|"
        r"\bfake product\b|"
        r"\bonline seller\b.{0,60}\bnot received\b|"
        r"\bproduct\b.{0,60}\bnot received\b"
    ),

    (
        "E-Commerce & Job Fraud",
        "Non-Delivery of Goods",
        r"\bnever received\b|"
        r"\bnot delivered\b|"
        r"\bdid not receive\b|"
        r"\bfailed to deliver\b"
    ),

    (
        "Financial Fraud",
        "Phishing/Fake Websites",
        r"\bphishing\b|"
        r"\bfake website\b|"
        r"\bfraudulent website\b|"
        r"\bsuspicious link\b"
    ),

    (
        "Financial Fraud",
        "Investment/Loan Scams",
        r"\binvestment scam\b|"
        r"\bfake investment\b|"
        r"\bloan scam\b|"
        r"\bfake loan\b|"
        r"\bloan offer\b|"
        r"\bpersonal loan\b|"
        r"\bprocessing fee\b.{0,50}\bloan\b|"
        r"\bloan\b.{0,50}\bprocessing fee\b|"
        r"\bguaranteed returns\b|"
        r"\bhigh returns\b"
    ),

    (
        "Financial Fraud",
        "UPI/Payment Fraud",
        r"\bupi payment\b|"
        r"\bupi fraud\b|"
        r"\bupi scam\b|"
        r"\bpayment request\b|"
        r"\bqr code\b.{0,50}\bpayment\b|"
        r"\bqr payment\b"
    ),

    (
        "Online Harassment & Threats",
        "Stalking",
        r"\bstalking\b"
    ),

    (
        "Online Harassment & Threats",
        "Cyberbullying",
        r"\bcyberbullying\b"
    ),

    (
        "Online Harassment & Threats",
        "Cyber Defamation",
        r"\bcyber defamation\b"
    ),

    (
        "Online Harassment & Threats",
        "Threatening Messages",
        r"\bthreatening messages?\b|"
        r"\bthreatened me\b|"
        r"\bblackmail\b"
    ),
]


_COMPILED_RULES = [

    (
        category,
        subcategory,
        re.compile(
            pattern,
            re.IGNORECASE
        )
    )

    for category,
    subcategory,
    pattern

    in PRIORITY_RULES
]


# ============================================================
# FALLBACK CRIME PATTERNS
# ============================================================

CRIME_PATTERNS = {

    "Financial Fraud": {

        "UPI/Payment Fraud": [

            r"\bupi\b",
            r"\bqr code\b",
            r"\bpayment request\b",
            r"\bamount was deducted\b",
            r"\bmoney was deducted\b",
        ],

        "Credit Card Fraud": [

            r"\bcredit card\b",
            r"\bcard details\b",
        ],

        "Cryptocurrency Scams": [

            r"\bbitcoin\b",
            r"\bcrypto\b",
            r"\bcryptocurrency\b",
            r"\bethereum\b",
        ],

        "Investment/Loan Scams": [

            r"\binvestment\b",
            r"\binvested\b",
            r"\bloan\b",
            r"\bprocessing fee\b",
            r"\bhigh returns\b",
        ],

        "Phishing/Fake Websites": [

            r"\bverification link\b",
            r"\bclick(?:ed)?\b.{0,30}\blink\b",
            r"\bemail\b.{0,50}"
            r"\b(?:account|bank|kyc|verification)\b",
            r"\bgovernment subsidy\b",
            r"\bsubsidy scheme\b",
            r"\bregistration fee\b",
        ],

        "Online Gambling/Betting": [

            r"\billegal lottery\b",
            r"\blottery schemes?\b",
        ],
    },

    "E-Commerce & Job Fraud": {

        "Fake Shopping Websites": [

            r"\bfake store\b",
            r"\binstagram seller\b",
            r"\bblocked me\b.{0,40}\bpayment\b",
        ],

        "Fake Job Offers": [

            r"\bjob offer\b",
            r"\bfake job\b",
            r"\bonline job\b",
            r"\bjob scam\b",
            r"\bemployment\b.{0,40}\bpayment\b",
            r"\bjob\b.{0,40}\bprocessing fee\b",
            r"\bbackground verification\b",
        ],

        "Education/Course Scams": [

            r"\bcoaching\b",
            r"\bcertification\b",
            r"\beducation\b.{0,40}\bpaid\b",
            r"\bcourse\b.{0,40}\bpaid\b",
        ],

        "Non-Delivery of Goods": [

            r"\bnot received\b",
            r"\bnon[- ]delivery\b",
            r"\bseller took the money\b",
            r"\bstopped responding\b",
            r"\breceived a brick\b",
            r"\bdifferent from what was advertised\b",
            r"\bused and damaged items\b",
        ],
    },

    "Social Media Crimes": {

        "Fake Accounts/Impersonation": [

            r"\bfake instagram\b",
            r"\bfake facebook\b",
            r"\bprofile using my name\b",
        ],

        "Morphed Images/Deepfakes": [

            r"\bfake video\b",
            r"\bedited image\b",
        ],
    },

    "Hacking & Unauthorized Access": {

        "Account Takeover": [

            r"\baccount was hacked\b",
        ],

        "Data Breach": [

            r"\bdata was compromised\b",
            r"\bdata\b.{0,30}\bleaked\b",
            r"\bdata\b.{0,30}\bexposed\b",
            r"\bcompany database\b",
            r"\bdatabase was hacked\b",
        ],

        "Identity Theft": [

            r"\bpan card\b.{0,40}\bmisused\b",
            r"\baadhaar\b.{0,40}\bmisused\b",
        ],

        "Email/Social Media Hacking": [

            r"\bhacked my "
            r"(?:gmail|email|facebook|instagram|social media)\b",
        ],
    },

    "Online Harassment & Threats": {

        "Threatening Messages": [

            r"\bthreatened\b",
            r"\bthreat\b",
            r"\bthreatening\b.{0,50}\bmorphed\b",
            r"\bupload\b.{0,50}\bobscene images?\b",
        ],

        "Stalking": [

            r"\bstalked\b",
            r"\bstalker\b",
            r"\bcalling repeatedly\b",
            r"\bobtained my "
            r"(?:phone number|address)\b",
        ],

        "Cyber Defamation": [

            r"\bfalse rumou?rs?\b",
            r"\bdefam(?:e|ation|atory)\b",
            r"\brumou?rs? about me\b",
        ],
    },

    "Others": {

        "Copyright Violation": [

            r"\bcopyright\b",
            r"\bcopied and sold\b",
            r"\bintellectual property\b",
        ],

        "Illegal Content Sharing": [

            r"\bpirated (?:movies?|content)\b",
            r"\billegal content\b",
            r"\bleaked exam papers?\b",
            r"\banswer keys\b",
        ],

        "Misinformation Spreading": [

            r"\bmisinformation\b",
            r"\bfalse information\b",
        ],

        "Miscellaneous": [

            r"\bmultiple types of cybercrimes\b",
            r"\bvarious cybercrimes\b",
            r"\bother cybercrime\b",
            r"\borganized cybercrime\b",
            r"\bneed investigation\b",
        ],
    },
}


CATEGORY_TIEBREAK_PRIORITY = [

    "Ransomware & Malware",
    "Social Media Crimes",
    "Hacking & Unauthorized Access",
    "Online Harassment & Threats",
    "E-Commerce & Job Fraud",
    "Financial Fraud",
    "Others",
]


def classify_crime(
    text: str
) -> Tuple[
    Optional[str],
    Optional[str],
    List[Dict[str, str]]
]:
    """
    Return:

        primary category
        primary subcategory
        all matching rules

    Strong rules have priority.

    If no strong rule fires,
    fallback rules are scored.
    """

    t = text.lower()

    strong = [

        {
            "category": category,
            "subcategory": subcategory,
            "source": "rule"
        }

        for category,
        subcategory,
        regex

        in _COMPILED_RULES

        if regex.search(t)
    ]

    if strong:

        return (
            strong[0]["category"],
            strong[0]["subcategory"],
            strong
        )

    scored = []

    for category, subcategories in CRIME_PATTERNS.items():

        for subcategory, patterns in subcategories.items():

            score = sum(

                1

                for pattern
                in patterns

                if re.search(
                    pattern,
                    t,
                    flags=re.IGNORECASE
                )
            )

            if score:

                scored.append(
                    (
                        score,
                        category,
                        subcategory
                    )
                )

    if not scored:

        return (
            None,
            None,
            []
        )

    scored.sort(
        key=lambda x: (
            -x[0],
            CATEGORY_TIEBREAK_PRIORITY.index(
                x[1]
            )
            if x[1]
            in CATEGORY_TIEBREAK_PRIORITY
            else 99
        )
    )

    matches = [

        {
            "category": category,
            "subcategory": subcategory,
            "source": f"fallback(score={score})"
        }

        for score,
        category,
        subcategory
        in scored
    ]

    return (
        matches[0]["category"],
        matches[0]["subcategory"],
        matches
    )
# ============================================================
# ATTACK VARIANT
# ============================================================

ATTACK_VARIANT_PATTERNS = {

    "Fake Internship Scam": [
        r"\bfake internship\b",
        r"\binternship offer\b",
        r"\binternship\b.{0,60}\bregistration fee\b",
        r"\binternship\b.{0,60}\bprocessing fee\b",
        r"\binternship\b.{0,60}\bpayment\b",
        r"\bpaid internship\b.{0,60}\bfee\b",
    ],

    "Fake Job Scam": [
        r"\bfake job\b",
        r"\bfake job offer\b",
        r"\bonline job\b",
        r"\bjob offer\b.{0,60}\bfee\b",
        r"\bjob\b.{0,60}\bregistration fee\b",
        r"\bjob\b.{0,60}\bprocessing fee\b",
        r"\bjob scam\b",
    ],

    "Unsolicited OTP Scam": [
        r"\bunrequested otp\b",
        r"\bunsolicited otp\b",
        r"\breceived an otp\b.{0,50}\bwithout\b",
        r"\botp\b.{0,50}\bwithout requesting\b",
        r"\bsuspicious otp\b",
        r"\botp bombing\b",
    ],

    "Vishing": [
        r"\bvishing\b",
        r"\bfraudulent phone call\b",
        r"\bfraudulent call\b",
        r"\bscam call\b",
        r"\bcaller claimed to be\b",
        r"\bcaller pretending to be\b",
    ],

    "Smishing": [
        r"\bsmishing\b",
        r"\bfraudulent sms\b",
        r"\bfraudulent text message\b",
        r"\bsuspicious sms\b",
        r"\bsms.{0,40}\bphishing\b",
    ],

    "SIM Swap": [
        r"\bsim swap\b",
        r"\bsim swapping\b",
        r"\bsim was replaced\b",
        r"\bduplicate sim\b",
        r"\bduplicate sim card\b",
        r"\bphone stopped working\b.{0,80}\bsim\b",
    ],

    "Digital Arrest Scam": [
        r"\bdigital arrest\b",
        r"\bunder digital arrest\b",
        r"\bvideo call\b.{0,80}\bpolice\b",
        r"\bpolice\b.{0,80}\bvideo call\b",
        r"\bcyber crime officer\b.{0,80}\bvideo call\b",
    ],

    "Fake E-Challan": [
        r"\bfake e[- ]?challan\b",
        r"\be[- ]?challan scam\b",
        r"\btraffic challan\b.{0,60}\bfake\b",
        r"\bfake traffic fine\b",
    ],

    "Fake KYC Scam": [
        r"\bfake kyc\b",
        r"\bkyc scam\b",
        r"\bkyc update\b",
        r"\bkyc verification\b",
        r"\baccount.{0,30}\bkyc\b.{0,50}\blink\b",
    ],

    "Fake Customer Support": [
        r"\bfake customer support\b",
        r"\bfake customer care\b",
        r"\bfake support number\b",
        r"\bcustomer care scam\b",
        r"\bpretending to be customer support\b",
    ],

    "Fake Refund Scam": [
        r"\bfake refund\b",
        r"\brefund scam\b",
        r"\brefund\b.{0,50}\bqr code\b",
        r"\brefund\b.{0,50}\bupi\b",
        r"\brefund\b.{0,50}\botp\b",
    ],

    "Fake Courier Scam": [
        r"\bfake courier\b",
        r"\bcourier scam\b",
        r"\bcourier call\b",
        r"\bcourier\b.{0,60}\bcustoms\b",
        r"\bpackage\b.{0,60}\bcustoms\b",
        r"\bfedex scam\b",
        r"\bfedex\b.{0,80}\bpolice\b",
    ],

    "Government/Police Impersonation": [
        r"\bpretending to be police\b",
        r"\bpretending to be a police officer\b",
        r"\bclaimed to be police\b",
        r"\bclaimed to be a police officer\b",
        r"\bcrime branch\b.{0,50}\bcalled\b",
        r"\bcyber crime officer\b",
        r"\bcbi officer\b",
        r"\bgovernment officer\b.{0,50}\bcalled\b",
        r"\bpolice officer\b.{0,50}\bvideo call\b",
    ],

    "Fake Investment Group": [
        r"\bfake investment group\b",
        r"\binvestment group\b.{0,50}\btelegram\b",
        r"\binvestment group\b.{0,50}\bwhatsapp\b",
        r"\btelegram investment\b",
        r"\bwhatsapp investment\b",
    ],

    "QR Payment Fraud": [
        r"\bqr code scam\b",
        r"\bqr payment fraud\b",
        r"\bscan the qr\b.{0,50}\bpayment\b",
        r"\bqr code\b.{0,50}\bmoney\b",
    ],

    "Remote Access Scam": [
        r"\bremote access\b",
        r"\bremote desktop\b",
        r"\banydesk\b",
        r"\bteamviewer\b",
        r"\bremote control\b",
    ],

    "Malicious APK": [
        r"\bmalicious apk\b",
        r"\bfake apk\b",
        r"\bdownloaded an apk\b",
        r"\bapk\b.{0,50}\bmalware\b",
        r"\bapk file\b.{0,50}\bscam\b",
    ],

    "Deepfake Impersonation": [
        r"\bdeepfake\b",
        r"\bdeep fake\b",
        r"\bfake video call\b",
        r"\bai[- ]generated video\b",
        r"\bai[- ]generated voice\b",
        r"\bcloned voice\b",
        r"\bvoice cloning\b",
    ],

    "Fake Social Media Profile": [
        r"\bfake social media profile\b",
        r"\bfake profile\b",
        r"\bfake instagram profile\b",
        r"\bfake facebook profile\b",
        r"\bcreated a fake profile\b",
        r"\bcreated a fake account\b",
    ],

    "Account Takeover": [
        r"\baccount takeover\b",
        r"\baccount was taken over\b",
        r"\btook over my account\b",
        r"\bgained access to my account\b",
        r"\bpassword was changed\b.{0,50}\baccount\b",
    ],

    "Blackmail/Extortion": [
        r"\bblackmail\b",
        r"\bblackmailed\b",
        r"\bextortion\b",
        r"\bextorted\b",
        r"\bpay or\b.{0,60}\bphotos?\b",
        r"\bpay or\b.{0,60}\bvideo\b",
    ],

    "Ransomware": [
        r"\bransomware\b",
        r"\bfiles?\b.{0,30}\bencrypted\b",
        r"\bdatabase\b.{0,30}\bencrypted\b",
    ],

    "Fake Shopping Website": [
        r"\bfake shopping website\b",
        r"\bfake online store\b",
        r"\bfake ecommerce\b",
        r"\bfake product website\b",
    ],

    "Non-Delivery Scam": [
        r"\bpaid\b.{0,60}\bnever received\b",
        r"\bpaid\b.{0,60}\bnot delivered\b",
        r"\bproduct\b.{0,60}\bnever arrived\b",
    ],
}


def extract_attack_variants(
    text: str
) -> list:

    return extract_pattern_labels(
        text,
        ATTACK_VARIANT_PATTERNS
    )


# ============================================================
# EVENT TYPE
# ============================================================

def determine_event_type(
    text: str,
    outcomes: list,
    victim_actions: list,
    amount_lost: Optional[float]
) -> str:
    """
    Event type is intentionally separate from crime category.

    Examples:
        suspicious OTP with no loss
            -> SUSPICIOUS_ACTIVITY

        attempted phishing
            -> ATTEMPTED_FRAUD

        completed financial fraud
            -> REPORTED_CRIME

        malware/ransomware incident
            -> SECURITY_INCIDENT
    """

    t = text.lower()

    security_signals = [

        r"\bransomware\b",
        r"\bmalware\b",
        r"\bvirus\b",
        r"\bspyware\b",
        r"\bkeylogger\b",
        r"\bdata breach\b",
        r"\bdata was exposed\b",
        r"\bsystem was compromised\b",
        r"\bserver was compromised\b",
    ]

    suspicious_signals = [

        r"\bsuspicious\b",
        r"\bunwanted otp\b",
        r"\bunsolicited otp\b",
        r"\bunknown caller\b",
        r"\bsuspicious message\b",
        r"\bsuspicious link\b",
        r"\bsuspicious sms\b",
        r"\breceived an otp\b.{0,40}\bwithout\b",
    ]

    attempted_signals = [

        r"\battempted\b",
        r"\btried to scam\b",
        r"\btried to defraud\b",
        r"\btried to access\b",
        r"\battempted fraud\b",
        r"\bphishing attempt\b",
    ]

    if any(
        re.search(
            p,
            t,
            flags=re.IGNORECASE
        )
        for p in security_signals
    ):

        return "SECURITY_INCIDENT"

    if any(
        re.search(
            p,
            t,
            flags=re.IGNORECASE
        )
        for p in suspicious_signals
    ) and not outcomes and not victim_actions:

        return "SUSPICIOUS_ACTIVITY"

    if any(
        re.search(
            p,
            t,
            flags=re.IGNORECASE
        )
        for p in attempted_signals
    ):

        return "ATTEMPTED_FRAUD"

    if (
        amount_lost is not None
        or "Financial Loss" in outcomes
        or "Account Compromised" in outcomes
        or victim_actions
    ):

        return "REPORTED_CRIME"

    return "REPORTED_CRIME"


# ============================================================
# PLATFORM / SERVICE
# ============================================================

PLATFORM_PATTERNS = {

    "WhatsApp": [
        r"\bwhatsapp\b"
    ],

    "Instagram": [
        r"\binstagram\b",
        r"\binsta\b"
    ],

    "Facebook": [
        r"\bfacebook\b",
        r"\bfb\b"
    ],

    "Telegram": [
        r"\btelegram\b"
    ],

    "Skype": [
        r"\bskype\b"
    ],

    "PhonePe": [
        r"\bphonepe\b"
    ],

    "Google Pay": [
        r"\bgoogle pay\b",
        r"\bgpay\b"
    ],

    "Paytm": [
        r"\bpaytm\b"
    ],

    "Amazon": [
        r"\bamazon\b"
    ],

    "Flipkart": [
        r"\bflipkart\b"
    ],

    "FedEx": [
        r"\bfedex\b"
    ],

    "AnyDesk": [
        r"\banydesk\b"
    ],

    "TeamViewer": [
        r"\bteamviewer\b"
    ],

    "Skype": [
        r"\bskype\b"
    ],

    "Banking App": [
        r"\bbanking app\b",
        r"\bmobile banking\b"
    ],

    "UPI": [
        r"\bupi\b"
    ],

    "Website": [
        r"\bwebsite\b",
        r"\bweb site\b",
        r"\bportal\b"
    ],
}


def extract_platforms(
    text: str
) -> list:

    return extract_pattern_labels(
        text,
        PLATFORM_PATTERNS
    )


# ============================================================
# COMMUNICATION DIRECTION
# ============================================================

def determine_communication_direction(
    text: str
) -> Optional[str]:

    t = text.lower()

    victim_received = [

        r"\breceived a call\b",
        r"\breceived a message\b",
        r"\breceived an sms\b",
        r"\breceived an email\b",
        r"\bcalled me\b",
        r"\bmessaged me\b",
        r"\bcontacted me\b",
        r"\bsent me a link\b",
    ]

    victim_initiated = [

        r"\bi contacted\b",
        r"\bi called\b",
        r"\bi messaged\b",
        r"\bi contacted customer support\b",
        r"\bi called customer care\b",
        r"\bi searched for\b.{0,50}\bsupport\b",
    ]

    if any(
        re.search(
            p,
            t,
            flags=re.IGNORECASE
        )
        for p in victim_received
    ):

        return "ATTACKER_TO_VICTIM"

    if any(
        re.search(
            p,
            t,
            flags=re.IGNORECASE
        )
        for p in victim_initiated
    ):

        return "VICTIM_TO_ATTACKER"

    return None


# ============================================================
# CURRENCY
# ============================================================

def detect_currency(
    text: str
) -> Optional[str]:

    t = text.lower()

    if (
        "₹" in text
        or re.search(
            r"\b(?:rs\.?|inr|rupees?)\b",
            t
        )
    ):

        return "INR"

    if re.search(
        r"\b(?:usd|dollars?)\b",
        t
    ):

        return "USD"

    if re.search(
        r"\b(?:eur|euros?)\b",
        t
    ):

        return "EUR"

    if re.search(
        r"\b(?:gbp|pounds?)\b",
        t
    ):

        return "GBP"

    return None


# ============================================================
# LOSS DETECTION
# ============================================================

def determine_loss_occurred(
    text: str,
    outcomes: list,
    amount_lost: Optional[float]
) -> Optional[bool]:

    t = text.lower()

    explicit_loss = [

        r"\blost\b",
        r"\bfinancial loss\b",
        r"\bamount was deducted\b",
        r"\bmoney was deducted\b",
        r"\bfunds were deducted\b",
        r"\baccount was debited\b",
        r"\bmoney was stolen\b",
        r"\bfunds were stolen\b",
        r"\bunauthori[sz]ed transaction\b",
        r"\bfraudulent transaction\b",
    ]

    no_loss = [

        r"\bno money was lost\b",
        r"\bno financial loss\b",
        r"\bno amount was deducted\b",
        r"\bdid not lose any money\b",
        r"\bno loss occurred\b",
    ]

    if any(
        re.search(
            p,
            t,
            flags=re.IGNORECASE
        )
        for p in no_loss
    ):

        return False

    if (
        amount_lost is not None
        or "Financial Loss" in outcomes
        or any(
            re.search(
                p,
                t,
                flags=re.IGNORECASE
            )
            for p in explicit_loss
        )
    ):

        return True

    return None


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(
    incident: Dict[str, Any]
) -> str:
    """
    Simple extraction confidence.

    This is NOT a calibrated probability.

    It represents how much explicit evidence
    was found by the deterministic extractor.
    """

    score = 0

    if incident.get(
        "crime_category"
    ):
        score += 2

    if incident.get(
        "crime_subcategory"
    ):
        score += 2

    if incident.get(
        "attack_variant"
    ):
        score += 1

    if incident.get(
        "modus_operandi"
    ):
        score += 1

    if incident.get(
        "deception_method"
    ):
        score += 1

    if incident.get(
        "victim_action"
    ):
        score += 1

    if incident.get(
        "attacker_action"
    ):
        score += 1

    if incident.get(
        "outcome"
    ):
        score += 1

    if incident.get(
        "channel"
    ):
        score += 1

    if incident.get(
        "amount_lost"
    ) is not None:
        score += 1

    if incident.get(
        "phone_numbers"
    ):
        score += 1

    if incident.get(
        "urls"
    ):
        score += 1

    if score >= 9:
        return "HIGH"

    if score >= 5:
        return "MEDIUM"

    return "LOW"


# ============================================================
# REQUIRED / MISSING FIELDS
# ============================================================

REQUIRED_INCIDENT_FIELDS = [

    "crime_category",
    "crime_subcategory",
    "modus_operandi",
    "deception_method",
    "victim_action",
    "attacker_action",
    "outcome",
    "channel",
]


def get_missing_fields(
    incident: Dict[str, Any]
) -> List[str]:

    missing = []

    for field in REQUIRED_INCIDENT_FIELDS:

        value = incident.get(
            field
        )

        if (
            value is None
            or value == ""
            or value == []
        ):

            missing.append(
                field
            )

    return missing


# ============================================================
# REVIEW LOGIC
# ============================================================

def determine_needs_review(
    incident: Dict[str, Any],
    missing_fields: List[str]
) -> bool:

    critical_missing = {

        "crime_category",
        "crime_subcategory",
        "channel",
    }

    if critical_missing.intersection(
        missing_fields
    ):

        return True

    if incident.get(
        "confidence"
    ) == "LOW":

        return True

    if (
        incident.get(
            "crime_all_matches"
        )
        and len(
            incident["crime_all_matches"]
        ) > 4
    ):

        return True

    return False


# ============================================================
# MASTER EXTRACTION
# ============================================================

def extract_complaint_information(
    text: str
) -> Dict[str, Any]:

    text = normalize_complaint_text(
        text
    )

    (
        crime_category,
        crime_subcategory,
        crime_all_matches
    ) = classify_crime(
        text
    )

    modus_operandi = extract_mo(
        text
    )

    deception = extract_deception(
        text
    )

    victim_action = extract_victim_actions(
        text
    )

    attacker_action = extract_attacker_actions(
        text
    )

    outcomes = extract_outcomes(
        text
    )

    channels = extract_channels(
        text
    )

    platforms = extract_platforms(
        text
    )

    amounts = extract_amounts(
        text
    )

    payment_methods = extract_payment_methods(
        text
    )

    phones = extract_phones(
        text
    )

    emails = extract_emails(
        text
    )

    upi_ids = extract_upi_ids(
        text
    )

    bank_accounts = extract_bank_accounts(
        text
    )

    ifsc_codes = extract_ifsc_codes(
        text
    )

    urls = extract_urls(
        text
    )

    domains = extract_domains(
        text
    )

    ip_addresses = extract_ip_addresses(
        text
    )

    crypto_wallets = extract_crypto_wallets(
        text
    )

    social_media_ids = extract_social_media_ids(
        text
    )

    transaction_ids = extract_transaction_ids(
        text
    )

    organizations = extract_organizations(
        text
    )

    locations = extract_locations(
        text
    )

    attack_variants = extract_attack_variants(
        text
    )

    amount_lost = (
        max(amounts)
        if amounts
        else None
    )

    currency = detect_currency(
        text
    )

    loss_occurred = determine_loss_occurred(
        text,
        outcomes,
        amount_lost
    )

    event_type = determine_event_type(
        text,
        outcomes,
        victim_action,
        amount_lost
    )

    communication_direction = (
        determine_communication_direction(
            text
        )
    )

    # --------------------------------------------------------
    # MASTER INCIDENT OBJECT
    # --------------------------------------------------------

    incident = {

        # ----------------------------------------------------
        # Core classification
        # ----------------------------------------------------

        "crime_category":
            crime_category,

        "crime_subcategory":
            crime_subcategory,

        "crime_all_matches":
            crime_all_matches,

        "attack_variant":
            attack_variants,

        "modus_operandi":
            modus_operandi,

        "deception_method":
            deception,

        "victim_action":
            victim_action,

        "attacker_action":
            attacker_action,

        "outcome":
            outcomes,

        # ----------------------------------------------------
        # Event
        # ----------------------------------------------------

        "event_type":
            event_type,

        # ----------------------------------------------------
        # Communication
        # ----------------------------------------------------

        "channel":
            channels[0]
            if channels
            else None,

        "channels":
            channels,

        "platform_or_service":
            platforms,

        "communication_direction":
            communication_direction,

        # ----------------------------------------------------
        # Financial
        # ----------------------------------------------------

        "payment_method":
            payment_methods,

        "amount_lost":
            amount_lost,

        "amounts":
            amounts,

        "currency":
            currency,

        "loss_occurred":
            loss_occurred,

        # ----------------------------------------------------
        # Identifiers
        # ----------------------------------------------------

        "transaction_id":
            transaction_ids,

        "transaction_ids":
            transaction_ids,

        "bank_account":
            bank_accounts,

        "bank_accounts":
            bank_accounts,

        "ifsc_code":
            ifsc_codes,

        "ifsc_codes":
            ifsc_codes,

        "phone_numbers":
            phones,

        "phones":
            phones,

        "email_addresses":
            emails,

        "emails":
            emails,

        "upi_ids":
            upi_ids,

        "urls":
            urls,

        "domains":
            domains,

        "social_media_ids":
            social_media_ids,

        "ip_addresses":
            ip_addresses,

        "crypto_wallets":
            crypto_wallets,

        # ----------------------------------------------------
        # Entities
        # ----------------------------------------------------

        "organizations":
            organizations,

        "locations":
            locations,

        # ----------------------------------------------------
        # Compatibility aliases
        # ----------------------------------------------------

        "deception":
            deception,

    }

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    incident["confidence"] = (
        calculate_confidence(
            incident
        )
    )

    # --------------------------------------------------------
    # Missing fields
    # --------------------------------------------------------

    missing_fields = get_missing_fields(
        incident
    )

    incident["missing_fields"] = (
        missing_fields
    )

    incident["needs_review"] = (
        determine_needs_review(
            incident,
            missing_fields
        )
    )

    return incident


# ============================================================
# ANALYZE COMPLAINT
# ============================================================

def analyze_complaint(
    text: str
) -> Dict[str, Any]:
    """
    Public service function.

    Keeps the API structure used by LinkageX.
    """

    normalized_text = normalize_complaint_text(
        text
    )

    incident = extract_complaint_information(
        normalized_text
    )

    return {

        "complaint_text":
            normalized_text,

        "incident":
            incident,

        "missing_fields":
            incident.get(
                "missing_fields",
                []
            ),

        "needs_review":
            incident.get(
                "needs_review",
                True
            ),

    }


# ============================================================
# BACKWARD-COMPATIBILITY HELPERS
# ============================================================

def get_missing_fields_legacy(
    incident: Dict[str, Any]
) -> List[str]:
    """
    Compatibility wrapper for older backend code.
    """

    return get_missing_fields(
        incident
    )


# ============================================================
# PRETTY PRINT
# ============================================================

def print_analysis(
    result: Dict[str, Any]
) -> None:

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LINKAGEX NLP ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        "\nComplaint:"
    )

    print(
        result.get(
            "complaint_text",
            ""
        )
    )

    print(
        "\nStructured Incident:"
    )

    incident = result.get(
        "incident",
        {}
    )

    for key, value in incident.items():

        print(
            f"{key}: {value}"
        )

    print(
        "\nMissing Fields:"
    )

    print(
        result.get(
            "missing_fields",
            []
        )
    )

    print(
        "\nNeeds Review:"
    )

    print(
        result.get(
            "needs_review",
            True
        )
    )

    print(
        "\n"
        + "=" * 70
    )


# ============================================================
# SELF TESTS
# ============================================================

def run_self_tests():

    print(
        "\nRunning LinkageX NLP self-tests..."
    )

    # --------------------------------------------------------
    # Test 1: Fake internship
    # --------------------------------------------------------

    text1 = (
        "I received a WhatsApp message about a fake internship "
        "at a company. They asked me to pay a registration fee "
        "of Rs 2500. I transferred the amount through UPI and "
        "then they stopped responding."
    )

    result1 = analyze_complaint(
        text1
    )

    assert result1[
        "incident"
    ]["crime_category"] is not None

    assert (
        "Fake Internship Scam"
        in result1[
            "incident"
        ]["attack_variant"]
    )

    assert (
        result1[
            "incident"
        ]["amount_lost"] == 2500.0
    )

    assert (
        result1[
            "incident"
        ]["loss_occurred"] is True
    )

    # --------------------------------------------------------
    # Test 2: Suspicious OTP
    # --------------------------------------------------------

    text2 = (
        "I suddenly received multiple OTP messages "
        "from my bank even though I had not requested "
        "any transaction. No money was lost."
    )

    result2 = analyze_complaint(
        text2
    )

    assert (
        result2[
            "incident"
        ]["event_type"]
        == "SUSPICIOUS_ACTIVITY"
    )

    assert (
        result2[
            "incident"
        ]["loss_occurred"]
        is False
    )

    # --------------------------------------------------------
    # Test 3: Digital arrest
    # --------------------------------------------------------

    text3 = (
        "A caller claimed to be a police officer and "
        "said I was under digital arrest. He asked me "
        "to stay on a video call and transfer money."
    )

    result3 = analyze_complaint(
        text3
    )

    assert (
        "Digital Arrest Scam"
        in result3[
            "incident"
        ]["attack_variant"]
    )

    assert (
        "Government/Police Impersonation"
        in result3[
            "incident"
        ]["attack_variant"]
    )

    # --------------------------------------------------------
    # Test 4: Account takeover
    # --------------------------------------------------------

    text4 = (
        "Someone gained unauthorized access to my Instagram "
        "account and changed my password."
    )

    result4 = analyze_complaint(
        text4
    )

    assert (
        "Account Takeover"
        in result4[
            "incident"
        ]["attack_variant"]
    )

    assert (
        "Account Takeover"
        in result4[
            "incident"
        ]["attacker_action"]
    )

    # --------------------------------------------------------
    # Test 5: Identifier extraction
    # --------------------------------------------------------

    text5 = (
        "The caller contacted me from 9988233057 and "
        "asked me to transfer money to account "
        "3335002100021930 using IFSC PUNB0436400. "
        "The payment was made through RTGS."
    )

    result5 = analyze_complaint(
        text5
    )

    incident5 = result5[
        "incident"
    ]

    assert (
        "9988233057"
        in [
            x["value"]
            for x in incident5[
                "phone_numbers"
            ]
        ]
    )

    assert (
        "3335002100021930"
        in [
            x["value"]
            for x in incident5[
                "bank_accounts"
            ]
        ]
    )

    assert (
        "PUNB0436400"
        in [
            x["value"]
            for x in incident5[
                "ifsc_codes"
            ]
        ]
    )

    assert (
        "Bank Transfer"
        in incident5[
            "payment_method"
        ]
    )

    print(
        "All self-tests passed."
    )


# ============================================================
# SAMPLE HYDERABAD COMPLAINT TEST
# ============================================================

HYDERABAD_SAMPLE = """
I am a resident of Hyderabad 500057. I received a call from
9988233057 on my number 9989966455. The caller claimed that he
was from FedEx courier Mumbai Andheri East and said that a package
in my name containing my Aadhaar, passports, debit cards, laptop,
shoes and MDMA had been found by customs.

He then connected with me through a fake Skype video call using
live:.cid.62206428ed2de72c and displayed a fake MUMBAI CYBER CRIME
logo. The person claimed to be a Sub Inspector from Mumbai Crime
Branch.

He asked me for my bank details, balances, fixed deposits,
job details and personal information. He showed me a forged FIR
number 2212186 and claimed that it was related to CBI Mumbai.
The document was signed with the name Dr Balsing Rajput IPS.

I was then coerced into transferring 90 percent of my bank balance.
I transferred Rs 12,49,633 through RTGS from my ICICI Bank account
to Punjab National Bank account 3335002100021930 with IFSC
PUNB0436400 in Mumbai Andheri East.

After confirming the transfer, the scammers disconnected the call.
"""


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_self_tests()

    print(
        "\nRunning Hyderabad sample..."
    )

    sample_result = analyze_complaint(
        HYDERABAD_SAMPLE
    )

    print_analysis(
        sample_result
    )
