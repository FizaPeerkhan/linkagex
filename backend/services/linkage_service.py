import ast
import os
import math
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

STRUCTURED_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "cybercrime_structured_1083.csv"
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
    "modus_operandi",
    "deception",
    "victim_action",
    "attacker_action",
    "outcome",
    "channels",
    "phones",
    "emails",
    "upi_ids",
    "urls",
    "organizations",
    "locations",
    "payment_method"
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
    "semantic": 0.30,
    "crime": 0.08,
    "modus_operandi": 0.12,
    "deception": 0.05,
    "victim_action": 0.03,
    "attacker_action": 0.03,
    "outcome": 0.03,
    "channel": 0.03,
    "payment": 0.05,
    "organization": 0.05,
    "phone": 0.08,
    "email": 0.06,
    "upi": 0.10,
    "url": 0.08,
    "location": 0.01
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

df = pd.read_csv(STRUCTURED_DATA_PATH)

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

    if isinstance(value, float) and np.isnan(value):
        return set()

    if isinstance(value, np.ndarray):
        return {
            str(x).strip().lower()
            for x in value.tolist()
            if str(x).strip()
        }

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
            return {
                str(x).strip().lower()
                for x in parsed
                if str(x).strip()
            }

        if parsed is None:
            return set()

        return {str(parsed).strip().lower()}

    except Exception:
        return {text.lower()}


# ============================================================
# NORMALIZE HISTORICAL DATA
# ============================================================

for field in LIST_FIELDS:

    if field in df.columns:

        df[field + "_set"] = df[field].apply(
            parse_list
        )


def normalize_text(value):

    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
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

    parts = [
        str(row.get("complaint_text", "")),
        str(row.get("modus_operandi", "")),
        str(row.get("deception", "")),
        str(row.get("victim_action", "")),
        str(row.get("attacker_action", "")),
        str(row.get("outcome", ""))
    ]

    return " ".join(
        p
        for p in parts
        if p and p.lower() != "nan"
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

historical_embeddings = np.asarray(
    historical_embeddings
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

    "modus_operandi":
        "Modus operandi",

    "deception":
        "Deception pattern",

    "victim_action":
        "Victim action",

    "attacker_action":
        "Attacker action",

    "outcome":
        "Outcome",

    "channel":
        "Communication channel",

    "payment":
        "Payment method",

    "organization":
        "Organization/entity",

    "phone":
        "Phone number",

    "email":
        "Email address",

    "upi":
        "UPI ID",

    "url":
        "URL/domain",

    "location":
        "Location"
}


# ============================================================
# LINKAGE SCORE
# ============================================================

def calculate_linkage_score(
    query,
    case,
    semantic_score
):

    component_scores = {}

    evidence_details = {}

    # --------------------------------------------------------
    # Semantic
    # --------------------------------------------------------

    component_scores["semantic"] = (
        max(
            0.0,
            min(
                1.0,
                float(semantic_score)
            )
        )
    )

    # --------------------------------------------------------
    # Crime
    # --------------------------------------------------------

    component_scores["crime"] = (
        crime_similarity(
            query,
            case
        )
    )

    # --------------------------------------------------------
    # Structured fields
    # --------------------------------------------------------

    field_mapping = {

        "modus_operandi":
            "modus_operandi",

        "deception":
            "deception",

        "victim_action":
            "victim_action",

        "attacker_action":
            "attacker_action",

        "outcome":
            "outcome",

        "channel":
            "channels",

        "payment":
            "payment_method",

        "organization":
            "organizations",

        "phone":
            "phones",

        "email":
            "emails",

        "upi":
            "upi_ids",

        "url":
            "urls",

        "location":
            "locations"
    }

    for score_name, field in field_mapping.items():

        query_set = query.get(
            field + "_set",
            set()
        )

        case_set = case.get(
            field + "_set",
            set()
        )

        raw_score = jaccard_similarity(
            query_set,
            case_set
        )

        adjusted_score, shared, evidence = (
            specificity_adjusted_similarity(
                field,
                query_set,
                case_set
            )
        )

        component_scores[
            score_name
        ] = round(
            raw_score,
            4
        )

        evidence_details[
            score_name
        ] = {
            "raw_similarity":
                round(
                    raw_score,
                    4
                ),

            "specificity_adjusted":
                round(
                    adjusted_score,
                    4
                ),

            "shared_values":
                shared,

            "evidence":
                evidence
        }

    # --------------------------------------------------------
    # Calculate final score
    # --------------------------------------------------------

    weighted_score = 0.0
    active_weight = 0.0

    # Semantic always participates.
    weighted_score += (
        BASE_WEIGHTS["semantic"] *
        component_scores["semantic"]
    )

    active_weight += (
        BASE_WEIGHTS["semantic"]
    )

    # Crime participates when information exists.
    if component_scores["crime"] > 0:

        weighted_score += (
            BASE_WEIGHTS["crime"] *
            component_scores["crime"]
        )

        active_weight += (
            BASE_WEIGHTS["crime"]
        )

    # Structured fields use
    # specificity-adjusted similarity.
    for score_name in field_mapping:

        adjusted = evidence_details[
            score_name
        ]["specificity_adjusted"]

        raw = component_scores[
            score_name
        ]

        if raw <= 0:
            continue

        weight = BASE_WEIGHTS[
            score_name
        ]

        weighted_score += (
            weight * adjusted
        )

        active_weight += weight

    # --------------------------------------------------------
    # Normalize over available evidence
    # --------------------------------------------------------
    #
    # This prevents missing optional identifiers from
    # automatically pushing every case toward zero.
    # --------------------------------------------------------

    if active_weight > 0:

        final_score = (
            weighted_score /
            active_weight
        )

    else:

        final_score = 0.0

    return (
        final_score,
        component_scores,
        evidence_details
    )


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

def find_related_cases(
    incident,
    top_k=5
):

    # --------------------------------------------------------
    # Prepare query linkage text
    # --------------------------------------------------------

    query_text = create_linkage_text(
        incident
    )

    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        [query_text],
        normalize_embeddings=True
    )

    # --------------------------------------------------------
    # Semantic similarity
    # --------------------------------------------------------

    semantic_scores = cosine_similarity(
        query_embedding,
        historical_embeddings
    )[0]

    results = []

    # --------------------------------------------------------
    # Prepare query once
    # --------------------------------------------------------

    query = dict(
        incident
    )

    for field in LIST_FIELDS:

        query[field + "_set"] = parse_list(
            query.get(
                field,
                []
            )
        )

    # --------------------------------------------------------
    # Compare against historical repository
    # --------------------------------------------------------

    for index, row in df.iterrows():

        case = row.to_dict()

        score, component_scores, evidence_details = (
            calculate_linkage_score(
                query,
                case,
                float(
                    semantic_scores[index]
                )
            )
        )

        relationship = (
            classify_relationship(
                score
            )
        )

        explanations = (
            generate_explanation(
                component_scores,
                evidence_details
            )
        )

        evidence_summary = (
            build_evidence_summary(
                evidence_details
            )
        )

        # ----------------------------------------------------
        # Weighted contribution breakdown
        # ----------------------------------------------------

        contribution_breakdown = {}

        contribution_breakdown[
            "semantic"
        ] = round(
            BASE_WEIGHTS["semantic"]
            *
            component_scores["semantic"],
            4
        )

        if component_scores[
            "crime"
        ] > 0:

            contribution_breakdown[
                "crime"
            ] = round(
                BASE_WEIGHTS["crime"]
                *
                component_scores["crime"],
                4
            )

        for field in evidence_details:

            raw = component_scores[
                field
            ]

            if raw <= 0:
                continue

            adjusted = evidence_details[
                field
            ]["specificity_adjusted"]

            contribution_breakdown[
                field
            ] = round(
                BASE_WEIGHTS[field]
                *
                adjusted,
                4
            )

        # ----------------------------------------------------
        # Distinctive evidence flag
        # ----------------------------------------------------

        has_distinctive_evidence = (
            len(
                evidence_summary[
                    "distinctive"
                ]
            ) > 0
        )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        results.append({

            "case_id": case.get(
                "complaint_id",
                f"CASE_{index}"
            ),

            "score": round(
                float(score),
                4
            ),

            "relationship":
                relationship,

            "shared_signals":
                explanations,

            "component_scores": {
                key: round(
                    float(value),
                    4
                )
                for key, value
                in component_scores.items()
            },

            "contribution_breakdown":
                contribution_breakdown,

            "evidence": {

                "has_distinctive_evidence":
                    has_distinctive_evidence,

                "distinctive":
                    evidence_summary[
                        "distinctive"
                    ],

                "moderate":
                    evidence_summary[
                        "moderate"
                    ],

                "common":
                    evidence_summary[
                        "common"
                    ]
            }
        })

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # --------------------------------------------------------
    # Only return meaningful relationships
    # --------------------------------------------------------

    meaningful_results = [
        result
        for result in results
        if result["score"]
        >= SHARED_PATTERN_THRESHOLD
    ]

    return meaningful_results[:top_k]