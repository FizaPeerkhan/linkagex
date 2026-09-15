"""
LinkageX - NLP Complaint Structuring Service

Production version of the NLP complaint extraction pipeline.

Input:
    Raw cybercrime complaint text

Output:
    Structured incident information used by:
    - Citizen complaint flow
    - Missing-information flow
    - Investigator case repository
    - Case linkage service
"""

import re
from typing import Any, Dict

import spacy


# ============================================================
# spaCy MODEL
# ============================================================

try:
    nlp = spacy.load("en_core_web_sm")
except OSError as exc:
    raise RuntimeError(
        "spaCy model 'en_core_web_sm' is not installed. "
        "Run: python -m spacy download en_core_web_sm"
    ) from exc


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_complaint_text(text: str) -> str:
    """
    Basic normalization while preserving important entities
    such as numbers, URLs and punctuation.
    """
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ============================================================
# GENERIC PATTERN MATCHER
# ============================================================

def extract_pattern_labels(text: str, pattern_dictionary: Dict[str, list]) -> list:
    """
    Return all labels whose patterns are detected in the text.
    """
    detected = []

    for label, patterns in pattern_dictionary.items():
        for pattern in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                detected.append(label)
                break

    return detected


# ============================================================
# AMOUNT EXTRACTION
# ============================================================

def extract_amounts(text: str) -> list:
    """
    Extract monetary amounts.

    Examples:
        ₹25,000
        Rs. 30,000
        Rs 30000
        INR 50,000
    """

    pattern = r"(?:₹|Rs\.?|INR)\s*([\d,]+(?:\.\d+)?)"

    matches = re.findall(
        pattern,
        text,
        flags=re.IGNORECASE
    )

    amounts = []

    for value in matches:
        value = value.replace(",", "")

        try:
            amounts.append(float(value))
        except ValueError:
            continue

    return amounts


# ============================================================
# CHANNEL EXTRACTION
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
    ]
}


def extract_channels(text: str) -> list:

    text_lower = text.lower()

    detected = []

    specific_platforms = [
        "WhatsApp",
        "Instagram",
        "Facebook",
        "Twitter",
        "Telegram"
    ]

    # Specific platforms
    for channel in specific_platforms:

        for pattern in CHANNEL_PATTERNS[channel]:

            if re.search(pattern, text_lower):
                detected.append(channel)
                break

    # Other channels
    for channel in [
        "Email",
        "SMS",
        "Phone Call",
        "Website"
    ]:

        for pattern in CHANNEL_PATTERNS[channel]:

            if re.search(pattern, text_lower):
                detected.append(channel)
                break

    # Generic mobile app
    if (
        not any(
            platform in detected
            for platform in specific_platforms
        )
        and any(
            re.search(pattern, text_lower)
            for pattern in CHANNEL_PATTERNS["Mobile App"]
        )
    ):
        detected.append("Mobile App")

    return detected


# ============================================================
# PAYMENT METHOD EXTRACTION
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
        r"\bpaytm\b.{0,50}\bpaid\b"
    ],

    "Credit Card": [
        r"\bpaid\b.{0,50}\bcredit card\b",
        r"\bpayment\b.{0,50}\bcredit card\b",
        r"\busing (?:my )?credit card\b",
        r"\bthrough (?:my )?credit card\b"
    ],

    "Debit Card": [
        r"\bpaid\b.{0,50}\bdebit card\b",
        r"\bpayment\b.{0,50}\bdebit card\b",
        r"\busing (?:my )?debit card\b",
        r"\bthrough (?:my )?debit card\b"
    ],

    "Bank Transfer": [
        r"\bbank transfer\b",
        r"\btransferred\b.{0,50}\bto (?:my )?bank\b",
        r"\baccount transfer\b",
        r"\btransferred the money\b.{0,50}\bbank\b"
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
        r"\bethereum payment\b"
    ],

    "Digital Wallet": [
        r"\bpaid\b.{0,50}\bdigital wallet\b",
        r"\bpayment\b.{0,50}\bdigital wallet\b",
        r"\busing (?:a )?digital wallet\b",
        r"\busing (?:an )?e-wallet\b"
    ]
}


def extract_payment_methods(text: str) -> list:

    detected = []

    for method, patterns in PAYMENT_METHOD_PATTERNS.items():

        for pattern in patterns:

            if re.search(
                pattern,
                text,
                flags=re.IGNORECASE
            ):
                detected.append(method)
                break

    return detected


# ============================================================
# PHONE NUMBER EXTRACTION
# ============================================================

def extract_phones(text: str) -> list:

    pattern = r"(?:\+91[\s-]?)?[6-9]\d{9}\b"

    return re.findall(pattern, text)


# ============================================================
# EMAIL EXTRACTION
# ============================================================

def extract_emails(text: str) -> list:

    pattern = (
        r"\b[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )

    return re.findall(pattern, text)


# ============================================================
# UPI ID EXTRACTION
# ============================================================

def extract_upi_ids(text: str) -> list:

    pattern = r"\b[A-Za-z0-9._-]+@[A-Za-z0-9.-]+\b"

    candidates = re.findall(pattern, text)

    emails = set(
        extract_emails(text)
    )

    return [
        value
        for value in candidates
        if value not in emails
    ]


# ============================================================
# URL EXTRACTION
# ============================================================

def extract_urls(text: str) -> list:

    pattern = r"https?://[^\s]+"

    urls = re.findall(pattern, text)

    return [
        url.rstrip(".,!?;:)]}")
        for url in urls
    ]


# ============================================================
# ORGANIZATION EXTRACTION
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
    "Swiggy": "Swiggy"
}


def extract_organizations(text: str) -> list:

    detected = []

    sorted_orgs = sorted(
        ORGANIZATION_NORMALIZATION.keys(),
        key=len,
        reverse=True
    )

    for organization in sorted_orgs:

        pattern = r"\b" + re.escape(organization) + r"\b"

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
                detected.append(canonical_name)

    return detected


# ============================================================
# LOCATION EXTRACTION
# ============================================================

LOCATION_CONTEXT_PATTERNS = [

    r"\bin ([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b",

    r"\bat ([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b",

    r"\bfrom ([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b",

    r"\bnear ([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b",

    r"\blocated in ([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b",

    r"\bbased in ([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b",

    r"\bincident happened in ([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b",

    r"\bincident occurred in ([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b"
]


def extract_locations(text: str) -> list:

    doc = nlp(text)

    locations = []

    candidates = [
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ in ["GPE", "LOC"]
    ]

    for candidate in candidates:

        candidate_lower = candidate.lower()

        # Remove common false positives
        if candidate_lower in {
            "netflix",
            "amazon",
            "paytm",
            "phonepe",
            "facebook",
            "instagram",
            "twitter",
            "telegram",
            "whatsapp"
        }:
            continue

        contextual = False

        for pattern in LOCATION_CONTEXT_PATTERNS:

            if (
                re.search(pattern, text)
                and candidate in text
            ):
                contextual = True
                break

        if contextual:
            locations.append(candidate)

    return list(
        dict.fromkeys(locations)
    )


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
        r"\bfake offer\b"
    ],

    "Impersonation": [
        r"\bpretending to be\b",
        r"\bposing as\b",
        r"\bclaimed to be\b",
        r"\bimpersonat(?:e|ed|ing)\b",
        r"\bpretended to be\b"
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
        r"\bguaranteed returns\b"
    ],

    "Scam": [
        r"\bscam\b",
        r"\bscammer\b",
        r"\bscamming\b",
        r"\bponzi scheme\b"
    ]
}


def extract_deception(text: str) -> list:

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
        r"\bpaid inr\b"
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
        r"\bi sent inr\b"
    ],

    "Shared OTP": [
        r"\bshared\b.{0,30}\botp\b",
        r"\bgave\b.{0,30}\botp\b",
        r"\bprovided\b.{0,30}\botp\b",
        r"\btold them my otp\b"
    ],

    "Entered Details": [
        r"\bentered\b.{0,40}\bdetails\b",
        r"\bentered my\b",
        r"\bprovided my\b.{0,40}\bdetails\b",
        r"\bfilled in\b.{0,40}\bdetails\b"
    ],

    "Clicked Link": [
        r"\bclicked\b.{0,20}\blink\b",
        r"\bclicked on\b",
        r"\bclicked the link\b"
    ],

    "Invested": [
        r"\bi invested\b",
        r"\bi had invested\b",
        r"\bwe invested\b",
        r"\binvested ₹",
        r"\binvested rs\.?",
        r"\binvested inr\b"
    ],

    "Approved": [
        r"\bi approved\b",
        r"\bapproved the payment\b",
        r"\bi authorized the payment\b"
    ],

    "Scanned QR": [
        r"\bscanned\b.{0,20}\bqr\b",
        r"\bscan\b.{0,20}\bqr\b"
    ],

    "Downloaded": [
        r"\bi downloaded\b",
        r"\bi installed\b",
        r"\bdownloaded the\b",
        r"\binstalled the\b"
    ],

    "Shared Credentials": [
        r"\bshared\b.{0,30}\bpassword\b",
        r"\bshared\b.{0,30}\bpin\b",
        r"\bshared\b.{0,30}\bcredentials\b",
        r"\bprovided\b.{0,30}\bpassword\b",
        r"\bprovided\b.{0,30}\bcredentials\b"
    ]
}


def extract_victim_actions(text: str) -> list:

    detected = extract_pattern_labels(
        text,
        VICTIM_ACTION_PATTERNS
    )

    text_lower = text.lower()

    # Payment demand ≠ actual payment
    demand_patterns = [

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
        r"\bpay otherwise\b"
    ]

    actual_payment_patterns = [

        r"\bi paid\b",
        r"\bwe paid\b",
        r"\bpayment was made\b",
        r"\bmade the payment\b",
        r"\bpaid the amount\b"
    ]

    demand_only = any(
        re.search(pattern, text_lower)
        for pattern in demand_patterns
    )

    actual_payment = any(
        re.search(pattern, text_lower)
        for pattern in actual_payment_patterns
    )

    if demand_only and not actual_payment:

        if "Paid" in detected:
            detected.remove("Paid")

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
        r"\bgained access to my account\b"
    ],

    "Identity Misuse": [
        r"\bused my identity\b",
        r"\bmisused my identity\b",
        r"\bidentity was misused\b",
        r"\bused my aadhaar\b",
        r"\bmisused my aadhaar\b",
        r"\bused my pan\b",
        r"\bmisused my pan\b"
    ],

    "Hacked": [
        r"\bhacked\b",
        r"\bhacking\b",
        r"\bunauthorized access\b",
        r"\bserver was compromised\b",
        r"\bsystem was compromised\b",
        r"\bcomputer was compromised\b"
    ],

    "Threatened": [
        r"\bthreatened\b",
        r"\bthreatening\b",
        r"\bthreat\b",
        r"\bblackmail(?:ed|ing)?\b",
        r"\bdelete everything unless\b",
        r"\bunless paid\b"
    ],

    "Posted": [
        r"\bposted\b",
        r"\buploaded\b",
        r"\bshared\b.{0,30}\bonline\b",
        r"\bpublished\b"
    ],

    "Stole": [
        r"\bstole\b",
        r"\bstolen\b",
        r"\btheft\b",
        r"\bstole my\b"
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
    r"\bdemanded\b.{0,30}\bprocessing fee\b"
]
}


def extract_attacker_actions(text: str) -> list:

    detected = extract_pattern_labels(
        text,
        ATTACKER_ACTION_PATTERNS
    )

    # Specific action gets priority over generic action
    if "Account Takeover" in detected and "Hacked" in detected:
        detected.remove("Hacked")

    if "Identity Misuse" in detected and "Stole" in detected:
        detected.remove("Stole")

    return detected


# ============================================================
# OUTCOME
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

        r"\bunauthorized charges\b",
        r"\bunauthorised charges\b",
        r"\bunauthorized transaction\b",
        r"\bunauthorised transaction\b",
        r"\bfraudulent transaction\b",
        r"\bfraudulent transactions\b",
        r"\bunauthorized payment\b",
        r"\bunauthorised payment\b",

        r"\bmoney was stolen\b",
        r"\bfunds were stolen\b"
    ],

    "Account Compromised": [
        r"\baccount was compromised\b",
        r"\baccount compromised\b",
        r"\bmy account was hacked\b",
        r"\baccount was hacked\b",
        r"\bsecurity settings\b",
        r"\bsomeone gained access to my account\b"
    ],

    "Account Blocked": [
        r"\baccount\b.{0,20}\bblocked\b",
        r"\bblocked my account\b",
        r"\baccount has been blocked\b"
    ],

    "Scammer Stopped Responding": [
    r"\bstopped responding\b",
    r"\bstopped replying\b",
    r"\bstopped answering\b",
    r"\bno longer responds?\b",
    r"\bthey stopped responding\b",
    r"\bthey stopped replying\b",
    r"\bthey disappeared\b",
    r"\bscammer disappeared\b"
],

    "Product/Service Not Received": [
        r"\bnever received\b",
        r"\breceived nothing\b",
        r"\bproduct\b.{0,30}\bnot received\b",
        r"\bservice\b.{0,30}\bnot received\b",
        r"\bdid not receive\b",
        r"\bnot delivered\b",
        r"\bfailed to deliver\b"
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
        r"\bcustomer data\b.{0,40}\bexposed\b"
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
        r"\bfiles are locked\b"
    ],

    "Device Infected": [
        r"\bdevice\b.{0,30}\binfected\b",
        r"\bcomputer\b.{0,30}\binfected\b",
        r"\blaptop\b.{0,30}\binfected\b",
        r"\bphone\b.{0,30}\binfected\b",
        r"\bmalware\b",
        r"\bvirus\b.{0,30}\binfected\b"
    ],

    "Identity Theft": [
        r"\bidentity theft\b",
        r"\bidentity was stolen\b",
        r"\bused my identity\b",
        r"\bmisused my identity\b",
        r"\bmisused my aadhaar\b",
        r"\bused my aadhaar\b",
        r"\bmisused my pan\b",
        r"\bused my pan\b"
    ]
}


def extract_outcomes(text: str) -> list:

    return extract_pattern_labels(
        text,
        OUTCOME_PATTERNS
    )


# ============================================================
# CRIME CLASSIFICATION
# ============================================================

CRIME_PATTERNS = {

    "Ransomware & Malware": {

        "Ransomware Attack": [
            r"\bransomware\b",
            r"\bransom demand\b",
            r"\bfiles are encrypted\b",
            r"\bdatabase is encrypted\b",
            r"\bfiles? (?:were|was|are) (?:locked|encrypted)\b"
        ],

        "Malware": [
            r"\bmalware\b",
            r"\bvirus\b",
            r"\btrojan\b",
            r"\bspyware\b",
            r"\bkeylogger\b",
            r"\bmalicious software\b"
        ]
    },

    "Financial Fraud": {

        "UPI/Payment Fraud": [
            r"\bupi\b",
            r"\bqr code\b",
            r"\bpayment request\b",
            r"\bpayment fraud\b",
            r"\bamount was deducted\b",
            r"\bmoney was deducted\b"
        ],

        "Credit Card Fraud": [
            r"\bcredit card\b",
            r"\bcard details\b",
            r"\bcredit card fraud\b"
        ],

        "Cryptocurrency Scams": [
            r"\bbitcoin\b",
            r"\bcrypto\b",
            r"\bcryptocurrency\b",
            r"\bethereum\b",
            r"\bcrypto wallet\b"
        ],

        "Investment/Loan Scams": [
            r"\binvestment\b",
            r"\binvested\b",
            r"\bloan\b",
            r"\bprocessing fee\b",
            r"\bguaranteed returns\b",
            r"\bhigh returns\b"
        ],

        "Phishing/Fake Websites": [
            r"\bphishing\b",
            r"\bfake website\b",
            r"\bfraudulent website\b",
            r"\bverification link\b",
            r"\bsuspicious link\b",
            r"\bclick(?:ed)?\b.{0,30}\blink\b",
            r"\bemail\b.{0,50}\b(account|bank|kyc|verification)\b",
            r"\bgovernment subsidy\b",
            r"\bsubsidy scheme\b",
            r"\bregistration fee\b"
        ],

        "Online Gambling/Betting": [
            r"\bonline gambling\b",
            r"\bonline casino\b",
            r"\bonline betting\b",
            r"\bcricket betting\b",
            r"\bbetting app\b",
            r"\bgambling app\b",
            r"\bcasino\b",
            r"\billegal lottery\b",
            r"\blottery scheme\b",
            r"\blottery schemes\b"
        ]
    },

    "E-Commerce & Job Fraud": {

        "Fake Shopping Websites": [
            r"\bfake shopping\b",
            r"\bfake shopping website\b",
            r"\bfraudulent shopping\b",
            r"\bfake online store\b",
            r"\bfake store\b",
            r"\binstagram seller\b",
            r"\bblocked me\b.{0,40}\bpayment\b"
        ],

        "Fake Job Offers": [
            r"\bjob offer\b",
            r"\bfake job\b",
            r"\bonline job\b",
            r"\bjob scam\b",
            r"\bemployment\b.{0,40}\bpayment\b",
            r"\bjob\b.{0,40}\bprocessing fee\b",
            r"\bbackground verification\b",
            r"\bjob\b.{0,50}\bbackground verification\b"
        ],

        "Education/Course Scams": [
            r"\bonline course\b",
            r"\bskill development course\b",
            r"\bstudy material\b",
            r"\bcoaching\b",
            r"\bcertification\b",
            r"\bonline degree\b",
            r"\beducation\b.{0,40}\bpaid\b",
            r"\bcourse\b.{0,40}\bpaid\b"
        ],

        "Non-Delivery of Goods": [
            r"\bnever received\b",
            r"\bnot received\b",
            r"\bdid not receive\b",
            r"\bfailed to deliver\b",
            r"\bnon[- ]delivery\b",
            r"\bseller took the money\b",
            r"\bstopped responding\b",
            r"\breceived a brick\b",
            r"\bdifferent from what was advertised\b",
            r"\bused and damaged items\b",
            r"\bcompletely different\b.{0,30}\badvertised\b"
        ]
    },

    "Social Media Crimes": {

        "Fake Accounts/Impersonation": [
            r"\bfake profile\b",
            r"\bfake account\b",
            r"\bfake instagram\b",
            r"\bfake facebook\b",
            r"\bimpersonat(?:e|ed|ing)\b",
            r"\bprofile using my name\b"
        ],

        "Morphed Images/Deepfakes": [
            r"\bdeepfake\b",
            r"\bmorphed image\b",
            r"\bmorphed images\b",
            r"\bfake video\b",
            r"\bedited image\b"
        ]
    },

    "Hacking & Unauthorized Access": {

        "Account Takeover": [
            r"\baccount was taken over\b",
            r"\btook over my account\b",
            r"\baccount takeover\b",
            r"\baccount was hacked\b"
        ],

        "Data Breach": [
            r"\bdata breach\b",
            r"\bdata was compromised\b",
            r"\bcustomer data\b.{0,40}\bexposed\b",
            r"\bpersonal information\b.{0,40}\bexposed\b",
            r"\bdata\b.{0,30}\bleaked\b",
            r"\bdata\b.{0,30}\bexposed\b",
            r"\bcompany database\b",
            r"\bdatabase was hacked\b",
            r"\binformation\b.{0,40}\bleaked online\b"
        ],

        "Identity Theft": [
            r"\bidentity theft\b",
            r"\bused my identity\b",
            r"\bmisused my identity\b",
            r"\bpan card\b.{0,40}\bmisused\b",
            r"\bpan card\b.{0,50}\baccount\b",
            r"\baadhaar\b.{0,40}\bmisused\b"
        ],

        "Email/Social Media Hacking": [
            r"\bhacked my gmail\b",
            r"\bhacked my email\b",
            r"\bhacked my facebook\b",
            r"\bhacked my instagram\b",
            r"\bhacked my social media\b"
        ]
    },

    "Online Harassment & Threats": {

        "Threatening Messages": [
            r"\bthreatened\b",
            r"\bthreatening message\b",
            r"\bthreatening messages\b",
            r"\bblackmail\b",
            r"\bthreat\b",
            r"\bthreatening\b.{0,50}\bmorphed\b",
            r"\bmorphed\b.{0,50}\bimages?\b.{0,50}\bpay\b",
            r"\bupload\b.{0,50}\bobscene images?\b"
        ],

        "Stalking": [
            r"\bstalking\b",
            r"\bstalked\b",
            r"\bstalker\b",
            r"\bcalling repeatedly\b",
            r"\brepeatedly\b.{0,40}\bcalling\b",
            r"\bobtained my phone number\b",
            r"\bobtained my address\b"
        ],

        "Cyberbullying": [
            r"\bcyberbullying\b",
            r"\bcyber bullying\b",
            r"\bbullying online\b"
        ],

        "Cyber Defamation": [
            r"\bfalse rumors?\b",
            r"\bfalse rumours?\b",
            r"\bdefam(?:e|ation|atory)\b",
            r"\brumors? about me\b",
            r"\brumours? about me\b"
        ]
    },

    "Others": {

        "Copyright Violation": [
            r"\bcopyright\b",
            r"\bcopied and sold\b",
            r"\bsoftware code\b.{0,40}\bcopied\b",
            r"\bintellectual property\b"
        ],

        "Illegal Content Sharing": [
            r"\bpirated movies?\b",
            r"\bpirated content\b",
            r"\billegal content\b",
            r"\bshared pirated\b",
            r"\bleaked exam papers\b",
            r"\bleaked exam paper\b",
            r"\banswer keys\b"
        ],

        "Misinformation Spreading": [
            r"\bmisinformation\b",
            r"\bfalse information\b",
            r"\bfalse rumors?\b",
            r"\bfalse rumours?\b"
        ],

        "Miscellaneous": [
            r"\bmultiple types of cybercrimes\b",
            r"\bvarious cybercrimes\b",
            r"\bother cybercrime\b",
            r"\borganized cybercrime\b",
            r"\bcybercrime activities\b",
            r"\bneed investigation\b"
        ]
    }
}


def classify_crime(text: str):

    t = text.lower()

    # --------------------------------------------------------
    # 1. RANSOMWARE
    # --------------------------------------------------------

    if re.search(
        r"\bransomware\b"
        r"|\bfiles?\s+(?:are|were)\s+encrypted\b"
        r"|\bdatabase\s+(?:is|was)\s+encrypted\b"
        r"|\bdata\s+(?:is|was)\s+encrypted\b",
        t
    ):
        return (
            "Ransomware & Malware",
            "Ransomware Attack"
        )

    # --------------------------------------------------------
    # 2. MALWARE
    # --------------------------------------------------------

    if re.search(
        r"\bmalware\b"
        r"|\btrojan\b"
        r"|\bspyware\b"
        r"|\bkeylogger\b"
        r"|\bvirus\b",
        t
    ):
        return (
            "Ransomware & Malware",
            "Malware"
        )

    # --------------------------------------------------------
    # 3. DEEPFAKE
    # --------------------------------------------------------

    if re.search(
        r"\bdeepfake\b"
        r"|\bmorphed images?\b"
        r"|\bmorphed photos?\b"
        r"|\bsynthetic video\b",
        t
    ):
        return (
            "Social Media Crimes",
            "Morphed Images/Deepfakes"
        )

    # --------------------------------------------------------
    # 4. FAKE ACCOUNT / IMPERSONATION
    # --------------------------------------------------------

    if re.search(
        r"\bfake profile\b"
        r"|\bfake account\b"
        r"|\bcreated a fake\b"
        r"|\bimpersonat(?:e|ed|ing)\b"
        r"|\bpretending to be me\b"
        r"|\bpretended to be me\b",
        t
    ):
        return (
            "Social Media Crimes",
            "Fake Accounts/Impersonation"
        )

    # --------------------------------------------------------
    # 5. SOCIAL MEDIA HACKING
    # --------------------------------------------------------

    social_platforms = (
        r"facebook|instagram|twitter|telegram|"
        r"whatsapp|snapchat"
    )

    if re.search(
        r"\b(?:hacked|hack|compromised|taken over)\b"
        r".{0,50}\b(?:" + social_platforms + r")\b"
        r"|"
        r"\b(?:" + social_platforms + r")\b"
        r".{0,50}\b(?:hacked|hack|compromised|taken over)\b",
        t
    ):
        return (
            "Social Media Crimes",
            "Email/Social Media Hacking"
        )

    # --------------------------------------------------------
    # 6. ACCOUNT TAKEOVER
    # --------------------------------------------------------

    if re.search(
        r"\baccount was taken over\b"
        r"|\btook over my account\b"
        r"|\btaken over by someone\b"
        r"|\baccount takeover\b"
        r"|\bgained control of my account\b"
        r"|\bgained access to my account\b",
        t
    ):
        return (
            "Hacking & Unauthorized Access",
            "Account Takeover"
        )

    # --------------------------------------------------------
    # 7. IDENTITY THEFT
    # --------------------------------------------------------

    if re.search(
        r"\bidentity theft\b"
        r"|\bidentity was stolen\b"
        r"|\bidentity was misused\b"
        r"|\bused my identity\b"
        r"|\bmisused my identity\b"
        r"|\bused my aadhaar\b"
        r"|\bmisused my aadhaar\b"
        r"|\bused my pan\b"
        r"|\bmisused my pan\b",
        t
    ):
        return (
            "Hacking & Unauthorized Access",
            "Identity Theft"
        )

    # --------------------------------------------------------
    # 8. DATA BREACH
    # --------------------------------------------------------

    if re.search(
        r"\bdata breach\b"
        r"|\bdata was breached\b"
        r"|\bdata was exposed\b"
        r"|\bcustomer data\b.{0,50}\bexposed\b"
        r"|\bpersonal information\b.{0,50}\bexposed\b"
        r"|\bdata leak\b"
        r"|\bdata was leaked\b",
        t
    ):
        return (
            "Hacking & Unauthorized Access",
            "Data Breach"
        )

    # --------------------------------------------------------
    # 9. CREDIT CARD FRAUD
    # --------------------------------------------------------

    if re.search(
        r"\bcredit card\b"
        r".{0,60}"
        r"\b(?:unauthorized|unauthorised|fraudulent|charges|"
        r"transactions?|used|stolen)\b",
        t
    ):
        return (
            "Financial Fraud",
            "Credit Card Fraud"
        )

    # --------------------------------------------------------
    # 10. CRYPTOCURRENCY
    # --------------------------------------------------------

    if re.search(
        r"\b(?:cryptocurrency|crypto|bitcoin|ethereum)\b"
        r".{0,80}"
        r"\b(?:invested|investment|trading|exchange|"
        r"scam|fraud|stole|stolen|lost|transferred|wallet)\b"
        r"|"
        r"\b(?:invested|investment|transferred|lost|stole|stolen)\b"
        r".{0,80}"
        r"\b(?:cryptocurrency|crypto|bitcoin|ethereum)\b",
        t
    ):
        return (
            "Financial Fraud",
            "Cryptocurrency Scams"
        )

    # --------------------------------------------------------
    # 11. ONLINE GAMBLING
    # --------------------------------------------------------

    if re.search(
        r"\bonline gambling\b"
        r"|\bonline casino\b"
        r"|\bonline betting\b"
        r"|\bcricket betting\b"
        r"|\bbetting app\b"
        r"|\bgambling app\b"
        r"|\bcasino\b"
        r"|\bbetting\b",
        t
    ):
        return (
            "Financial Fraud",
            "Online Gambling/Betting"
        )

    # --------------------------------------------------------
    # 12. EDUCATION / COURSE SCAM
    # --------------------------------------------------------

    if re.search(
        r"\bstudy material\b"
        r"|\bskill development course\b"
        r"|\bonline degree\b"
        r"|\beducation course\b"
        r"|\bonline course\b"
        r"|\btraining course\b"
        r"|\bcourse fee\b",
        t
    ):
        return (
            "E-Commerce & Job Fraud",
            "Education/Course Scams"
        )

    # --------------------------------------------------------
    # 13. FAKE SHOPPING
    # --------------------------------------------------------

    if re.search(
        r"\bfake shopping\b"
        r"|\bfake shopping website\b"
        r"|\bfake online store\b"
        r"|\bfake product\b"
        r"|\bonline seller\b.{0,60}\bnot received\b"
        r"|\bproduct\b.{0,60}\bnot received\b",
        t
    ):
        return (
            "E-Commerce & Job Fraud",
            "Fake Shopping Websites"
        )

    # --------------------------------------------------------
    # 14. NON-DELIVERY
    # --------------------------------------------------------

    if re.search(
        r"\bnever received\b"
        r"|\bnot delivered\b"
        r"|\bdid not receive\b"
        r"|\bfailed to deliver\b",
        t
    ):
        return (
            "E-Commerce & Job Fraud",
            "Non-Delivery of Goods"
        )

    # --------------------------------------------------------
    # 15. PHISHING
    # --------------------------------------------------------

    if re.search(
        r"\bphishing\b"
        r"|\bfake website\b"
        r"|\bfraudulent website\b"
        r"|\bsuspicious link\b"
        r"|\bphishing link\b",
        t
    ):
        return (
            "Financial Fraud",
            "Phishing/Fake Websites"
        )

    # --------------------------------------------------------
    # 16. INVESTMENT / LOAN
    # --------------------------------------------------------

    if re.search(
         r"\binvestment scam\b"
         r"|\bfake investment\b"
         r"|\bloan scam\b"
         r"|\bfake loan\b"
         r"|\bloan offer\b"
         r"|\bpersonal loan\b"
         r"|\bprocessing fee\b.{0,50}\bloan\b"
         r"|\bloan\b.{0,50}\bprocessing fee\b"
         r"|\bguaranteed returns\b"
         r"|\bhigh returns\b",
         t
    ):
        return (
            "Financial Fraud",
            "Investment/Loan Scams"
        )

    # --------------------------------------------------------
    # 17. UPI / PAYMENT
    # --------------------------------------------------------

    if re.search(
        r"\bupi payment\b"
        r"|\bupi fraud\b"
        r"|\bupi scam\b"
        r"|\bpayment request\b"
        r"|\bqr code\b.{0,50}\bpayment\b"
        r"|\bqr payment\b",
        t
    ):
        return (
            "Financial Fraud",
            "UPI/Payment Fraud"
        )

    # --------------------------------------------------------
    # 18. THREATS / HARASSMENT
    # --------------------------------------------------------

    if re.search(
        r"\bthreatening messages?\b"
        r"|\bthreatened me\b"
        r"|\bblackmail\b"
        r"|\bcyberbullying\b"
        r"|\bstalking\b"
        r"|\bcyber defamation\b",
        t
    ):

        if re.search(r"\bstalking\b", t):
            return (
                "Online Harassment & Threats",
                "Stalking"
            )

        if re.search(r"\bcyberbullying\b", t):
            return (
                "Online Harassment & Threats",
                "Cyberbullying"
            )

        if re.search(r"\bcyber defamation\b", t):
            return (
                "Online Harassment & Threats",
                "Cyber Defamation"
            )

        return (
            "Online Harassment & Threats",
            "Threatening Messages"
        )

    # --------------------------------------------------------
    # 19. GENERIC FALLBACK
    # --------------------------------------------------------

    matches = []

    for category, subcategories in CRIME_PATTERNS.items():

        for subcategory, patterns in subcategories.items():

            for pattern in patterns:

                if re.search(
                    pattern,
                    t,
                    flags=re.IGNORECASE
                ):

                    matches.append(
                        (category, subcategory)
                    )

                    break

    if matches:

        priority = [
            "Ransomware & Malware",
            "Social Media Crimes",
            "Hacking & Unauthorized Access",
            "Online Harassment & Threats",
            "E-Commerce & Job Fraud",
            "Financial Fraud",
            "Others"
        ]

        matches.sort(
            key=lambda x:
                priority.index(x[0])
                if x[0] in priority
                else len(priority)
        )

        return matches[0]

    return None, None


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
    ]
}


def extract_mo(text: str) -> list:

    return extract_pattern_labels(
        text,
        MO_PATTERNS
    )


# ============================================================
# MASTER EXTRACTION FUNCTION
# ============================================================

def extract_complaint_information(text: str) -> Dict[str, Any]:

    crime_category, crime_subcategory = classify_crime(text)

    return {

        "crime_category": crime_category,

        "crime_subcategory": crime_subcategory,

        "modus_operandi":
            extract_mo(text),

        "deception":
            extract_deception(text),

        "victim_action":
            extract_victim_actions(text),

        "attacker_action":
            extract_attacker_actions(text),

        "outcome":
            extract_outcomes(text),

        "channels":
            extract_channels(text),

        "amounts":
            extract_amounts(text),

        "payment_method":
            extract_payment_methods(text),

        "phones":
            extract_phones(text),

        "emails":
            extract_emails(text),

        "upi_ids":
            extract_upi_ids(text),

        "urls":
            extract_urls(text),

        "organizations":
            extract_organizations(text),

        "locations":
            extract_locations(text)
    }


# ============================================================
# MISSING FIELD DETECTION
# ============================================================

def is_missing(value: Any) -> bool:

    if value is None:
        return True

    if isinstance(value, str):
        return value.strip() == ""

    if isinstance(value, (list, tuple, set)):
        return len(value) == 0

    return False


def get_missing_fields(incident: Dict[str, Any]) -> list:

    required_fields = [

        "crime_category",
        "crime_subcategory",
        "modus_operandi",
        "deception",
        "victim_action",
        "attacker_action",
        "outcome",
        "channels",
        "amounts",
        "payment_method"
    ]

    missing_fields = []

    for field in required_fields:

        if (
            field not in incident
            or is_missing(incident[field])
        ):
            missing_fields.append(field)

    return missing_fields


# ============================================================
# PUBLIC SERVICE FUNCTION
# ============================================================

def analyze_complaint(text: str) -> Dict[str, Any]:

    """
    Main production entry point.

    Takes raw complaint text and returns:
        complaint_text
        incident
        missing_fields
    """

    if not isinstance(text, str):
        raise TypeError(
            "Complaint text must be a string."
        )

    text = normalize_complaint_text(text)

    if not text:
        raise ValueError(
            "Complaint text cannot be empty."
        )

    incident = extract_complaint_information(text)

    return {

        "complaint_text": text,

        "incident": incident,

        "missing_fields":
            get_missing_fields(incident)
    }


# ============================================================
# SERVICE READY
# ============================================================

print("LinkageX NLP service loaded successfully.")