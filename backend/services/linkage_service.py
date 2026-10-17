import ast
import os
import math
import re
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

STRUCTURED_DATA_PATH = os.environ.get(
    "LINKAGEX_DATASET_PATH",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
        "LINKAGEX_MASTER_CASES_V2.csv"
    )
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# FIELDS
# ============================================================

TEXT_FIELDS = [
    "complaint_text",
    "modus_operandi",
    "deception",
    "victim_action",
    "attacker_action",
    "outcome"
]

LIST_FIELDS = [
    "modus_operandi", "deception", "victim_action", "attacker_action", "outcome",
    "channels", "phones", "emails", "upi_ids", "urls", "organizations", "locations",
    "payment_method", "attack_variant", "attack_pattern", "platform_or_service",
    "transaction_ids", "bank_accounts", "ifsc_codes", "social_accounts", "ip_addresses",
    "crypto_wallets", "person_references"
]


# ============================================================
# BASE WEIGHTS
# ============================================================
#
# More weight is given to signals that can be more distinctive.
# Generic behavioural/context fields receive lower weights.
#
# These are NOT probabilities.
# They are model-design weights used for ranking.
# ============================================================

BASE_WEIGHTS = {
    # Semantic and crime scores are fixed portions of the final score.
    "semantic": 0.35,
    "crime": 0.15,
    # These are relative weights within the structured-evidence component.
    "modus_operandi": 0.15,
    "deception": 0.08,
    "victim_action": 0.05,
    "attacker_action": 0.05,
    "outcome": 0.03,
    "channel": 0.05,
    "payment": 0.05,
    "organization": 0.07,
    "phone": 0.10,
    "email": 0.08,
    "upi": 0.12,
    "url": 0.10,
    "location": 0.02,
    "attack_variant": 0.08,
    "attack_pattern": 0.10,
    "platform": 0.03,
    "transaction_id": 0.12,
    "bank_account": 0.10,
    "ifsc": 0.07,
    "social_account": 0.08,
    "ip_address": 0.10,
    "crypto_wallet": 0.12,
    "person_reference": 0.08,
}


# ============================================================
# RELATIONSHIP THRESHOLDS
# ============================================================

POTENTIAL_RELATIONSHIP_THRESHOLD = 0.70
SHARED_PATTERN_THRESHOLD = 0.45
WEAK_SIMILARITY_THRESHOLD = 0.25


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

print("Loading LinkageX historical cases...")

df = pd.read_csv(STRUCTURED_DATA_PATH, low_memory=False)

# Adapt LINKAGEX_MASTER_CASES_V2.csv to the canonical field names
# used by the existing scoring/evidence pipeline. Keep source columns.
COLUMN_ALIASES = {
    "complaint_id": "case_id",  # legacy compatibility; case_id remains the displayed ID
    "deception": "deception_method",
    "channels": "channel",
    "phones": "phone",
    "emails": "email",
    "upi_ids": "upi",
    "organizations": "organization",
    "transaction_ids": "transaction_id",
    "bank_accounts": "bank_account",
    "ifsc_codes": "ifsc_code",
    "social_accounts": "social_account",
    "ip_addresses": "ip_address",
    "crypto_wallets": "crypto_wallet",
    "person_references": "person_reference",
}

for canonical, source in COLUMN_ALIASES.items():
    if canonical not in df.columns and source in df.columns:
        df[canonical] = df[source]

# Preserve both full URLs and their domains as searchable indicators.
if "urls" not in df.columns:
    url_values = df["url"] if "url" in df.columns else pd.Series("", index=df.index)
    domain_values = df["domain"] if "domain" in df.columns else pd.Series("", index=df.index)
    df["urls"] = (url_values.fillna("").astype(str) + ";" + domain_values.fillna("").astype(str)).str.strip(";")

# The master schema distinguishes incident location from locations mentioned
# in the complaint. Retain both as location indicators for linkage.
if "locations" not in df.columns:
    mentioned = df["locations_mentioned"] if "locations_mentioned" in df.columns else pd.Series("", index=df.index)
    incident_location = df["location"] if "location" in df.columns else pd.Series("", index=df.index)
    df["locations"] = (mentioned.fillna("").astype(str) + ";" + incident_location.fillna("").astype(str)).str.strip(";")

print(f"Historical cases loaded: {len(df)}")


# ============================================================
# SAFE LIST PARSER
# ============================================================

def parse_list(value):
    """
    Safely converts stored list values into normalized sets.

    Handles:
    - Python lists
    - string representations of lists
    - empty values
    - numpy arrays
    - plain strings
    """

    if value is None:
        return set()

    if pd.isna(value) if not isinstance(value, (list, tuple, set)) else False:
        return set()

    if isinstance(value, (list, tuple, set)):
        return {
            str(x).strip().lower()
            for x in value
            if str(x).strip()
        }

    text = str(value).strip()

    if not text or text.lower() in {
        "nan",
        "none",
        "null",
        "[]"
    }:
        return set()

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, (list, tuple, set)):
            values = [str(x).strip().lower() for x in parsed if str(x).strip()]
        elif parsed is None:
            return set()
        else:
            values = [str(parsed).strip().lower()]

    except Exception:
        # Master CSV stores multi-value fields as semicolon-separated text.
        values = [part.strip().lower() for part in text.split(";") if part.strip()]

    # Also split semicolon-delimited strings parsed as quoted literals.
    flattened = []
    for value in values:
        flattened.extend(part.strip().lower() for part in value.split(";") if part.strip())
    return set(flattened)


# ============================================================
# NORMALIZE HISTORICAL DATA
# ============================================================

def normalize_channel_values(values):
    aliases = {
        "phone call": "phone",
        "telephone call": "phone",
        "voice call": "phone",
        "text message": "sms",
        "text messages": "sms",
        "short message service": "sms",
    }
    return {aliases.get(value.strip().lower(), value.strip().lower()) for value in values if str(value).strip()}


for field in LIST_FIELDS:
    if field in df.columns:
        df[field + "_set"] = df[field].apply(parse_list)
        if field == "channels":
            df[field + "_set"] = df[field + "_set"].apply(normalize_channel_values)


def normalize_text(value):

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return " ".join(
        str(value).lower().split()
    )


df["complaint_text_normalized"] = (
    df["complaint_text"]
    .apply(normalize_text)
)


# ============================================================
# CREATE LINKAGE TEXT
# ============================================================

def create_linkage_text(row):
    """Build a semantic representation from complaint and normalized fields."""
    parts = [
        row.get("complaint_text", ""),
        row.get("crime_category", ""),
        row.get("crime_subcategory", ""),
        row.get("attack_variant", ""),
        row.get("attack_pattern", ""),
        row.get("modus_operandi", ""),
        row.get("deception", row.get("deception_method", "")),
        row.get("victim_action", ""),
        row.get("attacker_action", ""),
        row.get("outcome", ""),
        row.get("channels", row.get("channel", "")),
        row.get("platform_or_service", ""),
        row.get("payment_method", ""),
    ]
    return " ".join(
        str(part).strip()
        for part in parts
        if part is not None and str(part).strip() and str(part).lower() != "nan"
    )


df["linkage_text"] = df.apply(
    create_linkage_text,
    axis=1
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


# ============================================================
# PRE-COMPUTE HISTORICAL EMBEDDINGS
# ============================================================

print("Creating historical embeddings...")

historical_embeddings = embedding_model.encode(
    df["linkage_text"].tolist(),
    normalize_embeddings=True,
    show_progress_bar=True
)

print(
    "Embedding matrix:",
    historical_embeddings.shape
)


# ============================================================
# BUILD VALUE FREQUENCY INDEX
# ============================================================
#
# This is the key research improvement.
#
# A signal appearing in many historical cases is less distinctive.
# A signal appearing in very few cases is more distinctive.
#
# Example:
#
# WhatsApp -> many cases -> low specificity
# Fake Processing Fee -> fewer cases -> higher specificity
# Specific UPI ID -> very few cases -> very high specificity
# ============================================================

VALUE_DOCUMENT_FREQUENCY = {}

for field in LIST_FIELDS:

    frequency = {}

    if field + "_set" not in df.columns:
        VALUE_DOCUMENT_FREQUENCY[field] = frequency
        continue

    for values in df[field + "_set"]:

        for value in values:
            frequency[value] = (
                frequency.get(value, 0) + 1
            )

    VALUE_DOCUMENT_FREQUENCY[field] = frequency


# ============================================================
# SPECIFICITY
# ============================================================

def signal_specificity(field, value):

    """
    Returns an approximate distinctiveness value between 0 and 1.

    Higher = rarer in the historical repository.

    This is an evidence weighting factor, NOT a probability.
    """

    frequency = VALUE_DOCUMENT_FREQUENCY.get(
        field,
        {}
    )

    document_frequency = frequency.get(
        value,
        0
    )

    total_cases = max(len(df), 1)

    if document_frequency <= 0:
        return 1.0

    # Smoothed inverse-document-frequency style score.
    idf = math.log(
        (total_cases + 1) /
        (document_frequency + 1)
    )

    max_idf = math.log(
        total_cases + 1
    )

    if max_idf == 0:
        return 0.0

    score = idf / max_idf

    return float(
        max(0.0, min(1.0, score))
    )


# ============================================================
# SET SIMILARITY
# ============================================================

def jaccard_similarity(set_a, set_b):

    if not set_a or not set_b:
        return 0.0

    intersection = len(
        set_a & set_b
    )

    union = len(
        set_a | set_b
    )

    if union == 0:
        return 0.0

    return intersection / union


# ============================================================
# SHARED VALUES
# ============================================================

def get_shared_values(
    set_a,
    set_b
):

    if not set_a or not set_b:
        return []

    return sorted(
        set_a & set_b
    )


# ============================================================
# SPECIFICITY-AWARE LIST SIMILARITY
# ============================================================

def specificity_adjusted_similarity(
    field,
    set_a,
    set_b
):

    shared = get_shared_values(
        set_a,
        set_b
    )

    if not shared:
        return 0.0, [], []

    raw_similarity = jaccard_similarity(
        set_a,
        set_b
    )

    specificities = []

    for value in shared:

        specificity = signal_specificity(
            field,
            value
        )

        specificities.append(
            specificity
        )

    # Use the strongest shared signal.
    #
    # This prevents several common signals from
    # overpowering one distinctive signal.
    strongest_specificity = max(
        specificities
    )

    adjusted_score = (
        raw_similarity *
        strongest_specificity
    )

    evidence = []

    for value, specificity in zip(
        shared,
        specificities
    ):

        if specificity >= 0.75:
            level = "Distinctive"

        elif specificity >= 0.40:
            level = "Moderately distinctive"

        else:
            level = "Common"

        evidence.append({
            "value": value,
            "specificity": round(
                specificity,
                4
            ),
            "level": level
        })

    return (
        adjusted_score,
        shared,
        evidence
    )


# ============================================================
# CRIME SIMILARITY
# ============================================================

def crime_similarity(
    query,
    case
):

    query_sub = normalize_text(
        query.get(
            "crime_subcategory",
            ""
        )
    )

    case_sub = normalize_text(
        case.get(
            "crime_subcategory",
            ""
        )
    )

    query_cat = normalize_text(
        query.get(
            "crime_category",
            ""
        )
    )

    case_cat = normalize_text(
        case.get(
            "crime_category",
            ""
        )
    )

    if (
        query_sub
        and case_sub
        and query_sub == case_sub
    ):
        return 1.0

    if (
        query_cat
        and case_cat
        and query_cat == case_cat
    ):
        return 0.5

    return 0.0


# ============================================================
# SIGNAL LABELS
# ============================================================

FIELD_LABELS = {
    "crime": "Crime classification",
    "modus_operandi": "Modus operandi",
    "deception": "Deception pattern",
    "victim_action": "Victim action",
    "attacker_action": "Attacker action",
    "outcome": "Outcome",
    "channel": "Communication channel",
    "payment": "Payment method",
    "organization": "Organization/entity",
    "phone": "Phone number",
    "email": "Email address",
    "upi": "UPI ID",
    "url": "URL/domain",
    "location": "Location",
    "attack_variant": "Attack variant",
    "attack_pattern": "Attack pattern",
    "platform": "Platform/service",
    "transaction_id": "Transaction ID",
    "bank_account": "Bank account",
    "ifsc": "IFSC code",
    "social_account": "Social account",
    "ip_address": "IP address",
    "crypto_wallet": "Cryptocurrency wallet",
    "person_reference": "Person reference",
}


# ============================================================
# LINKAGE SCORE
# ============================================================

def calculate_linkage_score(query, case, semantic_score):
    """Return a weighted ranking score and transparent component evidence.

    Semantic similarity is capped at 35% of the final score. Structured
    fields contribute only when present in the query, but a present query
    field with no match contributes zero rather than disappearing from the
    denominator. This prevents semantic-only cases from receiving inflated
    70-80% scores simply because no structured indicators matched.
    """
    component_scores = {
        "semantic": max(0.0, min(1.0, float(semantic_score))),
        "crime": crime_similarity(query, case),
    }
    evidence_details = {}

    field_mapping = {
        "modus_operandi": "modus_operandi",
        "deception": "deception",
        "victim_action": "victim_action",
        "attacker_action": "attacker_action",
        "outcome": "outcome",
        "channel": "channels",
        "payment": "payment_method",
        "organization": "organizations",
        "phone": "phones",
        "email": "emails",
        "upi": "upi_ids",
        "url": "urls",
        "location": "locations",
        "attack_variant": "attack_variant",
        "attack_pattern": "attack_pattern",
        "platform": "platform_or_service",
        "transaction_id": "transaction_ids",
        "bank_account": "bank_accounts",
        "ifsc": "ifsc_codes",
        "social_account": "social_accounts",
        "ip_address": "ip_addresses",
        "crypto_wallet": "crypto_wallets",
        "person_reference": "person_references",
    }

    # Categorical crime overlap is useful pattern evidence, but not a unique identifier.
    crime_evidence = []
    q_sub = normalize_text(query.get("crime_subcategory", ""))
    c_sub = normalize_text(case.get("crime_subcategory", ""))
    q_cat = normalize_text(query.get("crime_category", ""))
    c_cat = normalize_text(case.get("crime_category", ""))
    if q_sub and c_sub and q_sub == c_sub:
        crime_evidence.append({"value": str(case.get("crime_subcategory")), "specificity": 0.25, "level": "Common"})
    elif q_cat and c_cat and q_cat == c_cat:
        crime_evidence.append({"value": str(case.get("crime_category")), "specificity": 0.15, "level": "Common"})
    evidence_details["crime"] = {
        "raw_similarity": component_scores["crime"],
        "specificity_adjusted": component_scores["crime"],
        "shared_values": [item["value"] for item in crime_evidence],
        "evidence": crime_evidence,
    }

    structured_available_weight = 0.0
    structured_weighted_score = 0.0

    for score_name, field in field_mapping.items():
        query_set = query.get(field + "_set", set())
        case_set = case.get(field + "_set", set())
        raw_score = jaccard_similarity(query_set, case_set)
        adjusted_score, shared, evidence = specificity_adjusted_similarity(field, query_set, case_set)
        component_scores[score_name] = round(raw_score, 4)
        evidence_details[score_name] = {
            "raw_similarity": round(raw_score, 4),
            "specificity_adjusted": round(adjusted_score, 4),
            "shared_values": shared,
            "evidence": evidence,
        }
        if query_set:
            weight = BASE_WEIGHTS[score_name]
            structured_available_weight += weight
            structured_weighted_score += weight * adjusted_score

    structured_score = (
        structured_weighted_score / structured_available_weight
        if structured_available_weight > 0 else 0.0
    )
    final_score = (
        0.35 * component_scores["semantic"]
        + 0.15 * component_scores["crime"]
        + 0.50 * structured_score
    )
    final_score = max(0.0, min(1.0, final_score))

    # Contributions sum to the final score and are useful for auditing ranking.
    contribution_breakdown = {
        "semantic": round(0.35 * component_scores["semantic"], 4),
        "crime": round(0.15 * component_scores["crime"], 4),
    }
    if structured_available_weight > 0:
        for score_name in field_mapping:
            field = field_mapping[score_name]
            if query.get(field + "_set", set()):
                contribution_breakdown[score_name] = round(
                    0.50 * BASE_WEIGHTS[score_name] / structured_available_weight
                    * evidence_details[score_name]["specificity_adjusted"], 4
                )

    return final_score, component_scores, evidence_details, contribution_breakdown



# ============================================================
# RELATIONSHIP CLASSIFICATION
# ============================================================

def classify_relationship(score):

    if score >= POTENTIAL_RELATIONSHIP_THRESHOLD:

        return "Potential relationship"

    elif score >= SHARED_PATTERN_THRESHOLD:

        return "Possible shared pattern"

    elif score >= WEAK_SIMILARITY_THRESHOLD:

        return "Weak similarity"

    return "No significant relationship"


# ============================================================
# EXPLANATION GENERATOR
# ============================================================

def generate_explanation(
    component_scores,
    evidence_details
):

    explanations = []

    distinctive_signals = []
    moderate_signals = []
    common_signals = []

    # --------------------------------------------------------
    # Semantic
    # --------------------------------------------------------

    semantic = component_scores[
        "semantic"
    ]

    if semantic >= 0.70:

        explanations.append(
            "Strong semantic similarity"
        )

    elif semantic >= 0.50:

        explanations.append(
            "Moderate semantic similarity"
        )

    # --------------------------------------------------------
    # Crime
    # --------------------------------------------------------

    crime = component_scores[
        "crime"
    ]

    if crime == 1.0:

        explanations.append(
            "Same crime subcategory"
        )

    elif crime == 0.5:

        explanations.append(
            "Same crime category"
        )

    # --------------------------------------------------------
    # Structured evidence
    # --------------------------------------------------------

    for field, details in evidence_details.items():

        evidence = details[
            "evidence"
        ]

        if not evidence:
            continue

        label = FIELD_LABELS.get(
            field,
            field
        )

        for item in evidence:

            value = item[
                "value"
            ]

            level = item[
                "level"
            ]

            if level == "Distinctive":

                distinctive_signals.append({
                    "field": label,
                    "value": value
                })

            elif level == "Moderately distinctive":

                moderate_signals.append({
                    "field": label,
                    "value": value
                })

            else:

                common_signals.append({
                    "field": label,
                    "value": value
                })

    # --------------------------------------------------------
    # Human-readable explanations
    # --------------------------------------------------------

    for item in distinctive_signals:

        explanations.append(
            f"Distinctive shared {item['field'].lower()}: "
            f"{item['value']}"
        )

    for item in moderate_signals:

        explanations.append(
            f"Shared {item['field'].lower()}: "
            f"{item['value']}"
        )

    # Keep common evidence limited.
    # These are contextual rather than strong linkage evidence.
    for item in common_signals[:4]:

        explanations.append(
            f"Common shared pattern: "
            f"{item['field'].lower()} = "
            f"{item['value']}"
        )

    # --------------------------------------------------------
    # Explicitly identify absence of distinctive evidence
    # --------------------------------------------------------

    if not distinctive_signals:

        explanations.append(
            "No distinctive identifier was shared "
            "between the incidents"
        )

    if not explanations:

        explanations.append(
            "Similarity primarily based on complaint semantics"
        )

    return explanations


# ============================================================
# EVIDENCE SUMMARY
# ============================================================

def build_evidence_summary(
    evidence_details
):

    distinctive = []
    moderate = []
    common = []

    for field, details in evidence_details.items():

        label = FIELD_LABELS.get(
            field,
            field
        )

        for evidence in details.get(
            "evidence",
            []
        ):

            item = {
                "field": label,
                "value": evidence[
                    "value"
                ],
                "specificity": evidence[
                    "specificity"
                ]
            }

            level = evidence[
                "level"
            ]

            if level == "Distinctive":

                distinctive.append(item)

            elif level == "Moderately distinctive":

                moderate.append(item)

            else:

                common.append(item)

    return {
        "distinctive": distinctive,
        "moderate": moderate,
        "common": common
    }


# ============================================================
# MAIN LINKAGE FUNCTION
# ============================================================

def _first_present(mapping, *keys):
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "" and value != []:
            return value
    return None


def _query_value(value, field):
    """Normalize one query field into a list without breaking comma-bearing locations."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]
    output = []
    for item in items:
        text = str(item).strip()
        if not text or text.lower() in {"nan", "none", "null", "[]"}:
            continue
        # Channel lists from the NLP service may be comma-separated, while the
        # master CSV uses semicolons. Do not split locations/organizations on commas.
        if field == "channels":
            output.extend(part.strip() for part in re.split(r"[;,]", text) if part.strip())
        else:
            output.extend(part.strip() for part in text.split(";") if part.strip())
    return output


def find_related_cases(incident, top_k=5):
    """Rank historical master cases with evidence-backed, calibrated scores."""
    if not isinstance(incident, dict):
        raise ValueError("Incident must be a dictionary")

    query = dict(incident)
    aliases = {
        "deception": ("deception_method",),
        "channels": ("channel",),
        "phones": ("phone_numbers", "phone"),
        "emails": ("email_addresses", "email"),
        "upi_ids": ("upi_id", "upi"),
        "urls": ("url", "domains", "domain"),
        "organizations": ("organization",),
        "locations": ("locations_mentioned", "location"),
        "transaction_ids": ("transaction_id", "transaction_ids"),
        "bank_accounts": ("bank_account",),
        "ifsc_codes": ("ifsc_code",),
        "social_accounts": ("social_account",),
        "ip_addresses": ("ip_address",),
        "crypto_wallets": ("crypto_wallet",),
        "person_references": ("person_reference",),
        "platform_or_service": ("platform", "service"),
    }
    for canonical, alternatives in aliases.items():
        current = query.get(canonical)
        if current is None or current == "" or current == []:
            candidate = _first_present(query, *alternatives)
            if candidate is not None:
                query[canonical] = candidate

    # Expand URL values to domains too, so the same host can be linked even
    # when two reports contain different paths on that host.
    raw_urls = _query_value(query.get("urls", []), "urls")
    try:
        from urllib.parse import urlparse
        for raw_url in list(raw_urls):
            candidate = raw_url if "://" in raw_url else "https://" + raw_url
            host = urlparse(candidate).netloc.lower().split(":")[0]
            if host and host not in raw_urls:
                raw_urls.append(host)
    except Exception:
        pass
    query["urls"] = raw_urls

    for field in LIST_FIELDS:
        parsed_values = parse_list(_query_value(query.get(field, []), field))
        query[field + "_set"] = normalize_channel_values(parsed_values) if field == "channels" else parsed_values

    query_text = create_linkage_text(query)
    if not query_text.strip():
        query_text = str(query.get("complaint_text", "") or "")
    if query_text.strip():
        query_embedding = embedding_model.encode([query_text], normalize_embeddings=True)
        semantic_scores = cosine_similarity(query_embedding, historical_embeddings)[0]
    else:
        semantic_scores = [0.0] * len(df)

    query_case_id = str(_first_present(query, "case_id", "complaint_id") or "").strip()
    results = []

    for index, row in df.iterrows():
        case = row.to_dict()
        case_id = str(case.get("case_id") or case.get("complaint_id") or f"CASE_{index}")
        if query_case_id and case_id == query_case_id:
            continue

        score, component_scores, evidence_details, contribution_breakdown = calculate_linkage_score(
            query, case, float(semantic_scores[index])
        )
        relationship = classify_relationship(score)
        explanations = generate_explanation(component_scores, evidence_details)
        evidence_summary = build_evidence_summary(evidence_details)
        has_distinctive_evidence = bool(evidence_summary["distinctive"])

        def clean_value(value):
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return None
            return value.item() if hasattr(value, "item") else value

        complaint = clean_value(case.get("complaint_text"))
        results.append({
            "case_id": case_id,
            "score": round(float(score), 4),
            "relationship": relationship,
            "shared_signals": explanations,
            "component_scores": {key: round(float(value), 4) for key, value in component_scores.items()},
            "contribution_breakdown": contribution_breakdown,
            "evidence": {
                "has_distinctive_evidence": has_distinctive_evidence,
                "distinctive": evidence_summary["distinctive"],
                "moderate": evidence_summary["moderate"],
                "common": evidence_summary["common"],
            },
            "case_summary": {
                "crime_category": clean_value(case.get("crime_category")),
                "crime_subcategory": clean_value(case.get("crime_subcategory")),
                "attack_variant": clean_value(case.get("attack_variant")),
                "attack_pattern": clean_value(case.get("attack_pattern")),
                "incident_date": clean_value(case.get("incident_date")),
                "location": clean_value(case.get("location")),
                "outcome": clean_value(case.get("outcome")),
                "loss_occurred": clean_value(case.get("loss_occurred")),
                "amount_lost": clean_value(case.get("amount_lost")),
                "currency": clean_value(case.get("currency")),
                "channel": clean_value(case.get("channel")),
                "payment_method": clean_value(case.get("payment_method")),
                "incident_cluster_id": clean_value(case.get("incident_cluster_id")),
                "campaign_id": clean_value(case.get("campaign_id")),
                "attacker_entity_id": clean_value(case.get("attacker_entity_id")),
                "evidence_types": clean_value(case.get("evidence_types")),
                "complaint_preview": (str(complaint)[:280] + ("…" if complaint and len(str(complaint)) > 280 else "")) if complaint else None,
            },
        })

    # Prefer higher scores, then more distinctive and structured evidence.
    results.sort(
        key=lambda result: (
            result["score"],
            len(result["evidence"]["distinctive"]),
            len(result["evidence"]["moderate"]),
            len(result["evidence"]["common"]),
        ),
        reverse=True,
    )
    requested_k = max(1, int(top_k))

    # Primary results require the stronger shared-pattern threshold.
    meaningful_results = [
        result for result in results
        if result["score"] >= SHARED_PATTERN_THRESHOLD
    ]
    if meaningful_results:
        return meaningful_results[:requested_k]

    # If none reaches the primary threshold, retain candidates suitable for
    # cautious investigator review instead of returning an empty list.
    # classify_relationship() labels these as "Weak similarity" (0.25-0.45).
    # This is a review fallback, not a claim that cases are connected.
    review_candidates = [
        result for result in results
        if result["score"] >= WEAK_SIMILARITY_THRESHOLD
    ]
    return review_candidates[:requested_k]


def get_case_by_id(case_id):
    """Return one master-dataset case by its actual case_id, or None."""
    if "case_id" not in df.columns:
        return None
    matches = df[df["case_id"].astype(str).str.strip() == str(case_id).strip()]
    if matches.empty:
        return None
    row = matches.iloc[0].to_dict()
    result = {}
    for key, value in row.items():
        if value is None or (isinstance(value, float) and math.isnan(value)):
            result[key] = None
        elif hasattr(value, "item"):
            result[key] = value.item()
        else:
            result[key] = value
    # Keep API payloads focused; internal normalized set columns are not public fields.
    return {key: value for key, value in result.items() if not key.endswith("_set") and key != "complaint_text_normalized" and key != "linkage_text"}