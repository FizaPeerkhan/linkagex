import ast
import os

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
    Safely converts list-like values into normalized sets.

    Handles:
    - None
    - NaN
    - Python lists
    - tuples
    - sets
    - string representations of lists
    - scalar values
    """

    # None
    if value is None:
        return set()

    # Actual list-like objects
    if isinstance(value, (list, tuple, set)):
        return {
            str(x).strip().lower()
            for x in value
            if x is not None and str(x).strip()
        }

    # Safe handling of scalar NaN
    try:
        if pd.isna(value):
            return set()
    except (TypeError, ValueError):
        pass

    # Parse string representation
    try:
        parsed = ast.literal_eval(str(value))

        if isinstance(parsed, (list, tuple, set)):
            return {
                str(x).strip().lower()
                for x in parsed
                if x is not None and str(x).strip()
            }

        return {str(parsed).strip().lower()}

    except (ValueError, SyntaxError):
        return {str(value).strip().lower()}


# ============================================================
# NORMALIZE HISTORICAL DATA
# ============================================================

for field in LIST_FIELDS:

    if field in df.columns:

        df[field + "_set"] = df[field].apply(
            parse_list
        )


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(value):

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

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
# JACCARD SIMILARITY
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
# CRIME SIMILARITY
# ============================================================

def crime_similarity(query, case):

    query_sub = str(
        query.get(
            "crime_subcategory",
            ""
        )
    ).lower()

    case_sub = str(
        case.get(
            "crime_subcategory",
            ""
        )
    ).lower()

    query_cat = str(
        query.get(
            "crime_category",
            ""
        )
    ).lower()

    case_cat = str(
        case.get(
            "crime_category",
            ""
        )
    ).lower()

    # Same subcategory
    if (
        query_sub
        and case_sub
        and query_sub == case_sub
    ):
        return 1.0

    # Same broad category
    if (
        query_cat
        and case_cat
        and query_cat == case_cat
    ):
        return 0.5

    # Different category
    return 0.0


# ============================================================
# LINKAGE WEIGHTS
# ============================================================

WEIGHTS = {

    # Semantic similarity
    "semantic": 0.35,

    # Crime type
    "crime": 0.15,

    # Incident characteristics
    "modus_operandi": 0.15,
    "deception": 0.08,

    # Actions
    "victim_action": 0.08,
    "attacker_action": 0.08,

    # Outcome
    "outcome": 0.05,

    # Communication/payment/entity signals
    "channel": 0.03,
    "payment": 0.01,
    "organization": 0.02
}


# ============================================================
# CALCULATE LINKAGE SCORE
# ============================================================

def calculate_linkage_score(
    query,
    case,
    semantic_score
):

    scores = {}

    # --------------------------------------------------------
    # Semantic
    # --------------------------------------------------------

    scores["semantic"] = semantic_score

    # --------------------------------------------------------
    # Crime
    # --------------------------------------------------------

    scores["crime"] = crime_similarity(
        query,
        case
    )

    # --------------------------------------------------------
    # Modus Operandi
    # --------------------------------------------------------

    scores["modus_operandi"] = (
        jaccard_similarity(
            query.get(
                "modus_operandi_set",
                set()
            ),
            case.get(
                "modus_operandi_set",
                set()
            )
        )
    )

    # --------------------------------------------------------
    # Deception
    # --------------------------------------------------------

    scores["deception"] = (
        jaccard_similarity(
            query.get(
                "deception_set",
                set()
            ),
            case.get(
                "deception_set",
                set()
            )
        )
    )

    # --------------------------------------------------------
    # Victim Action
    # --------------------------------------------------------

    scores["victim_action"] = (
        jaccard_similarity(
            query.get(
                "victim_action_set",
                set()
            ),
            case.get(
                "victim_action_set",
                set()
            )
        )
    )

    # --------------------------------------------------------
    # Attacker Action
    # --------------------------------------------------------

    scores["attacker_action"] = (
        jaccard_similarity(
            query.get(
                "attacker_action_set",
                set()
            ),
            case.get(
                "attacker_action_set",
                set()
            )
        )
    )

    # --------------------------------------------------------
    # Outcome
    # --------------------------------------------------------

    scores["outcome"] = (
        jaccard_similarity(
            query.get(
                "outcome_set",
                set()
            ),
            case.get(
                "outcome_set",
                set()
            )
        )
    )

    # --------------------------------------------------------
    # Channel
    # --------------------------------------------------------

    scores["channel"] = (
        jaccard_similarity(
            query.get(
                "channels_set",
                set()
            ),
            case.get(
                "channels_set",
                set()
            )
        )
    )

    # --------------------------------------------------------
    # Payment Method
    # --------------------------------------------------------

    scores["payment"] = (
        jaccard_similarity(
            query.get(
                "payment_method_set",
                set()
            ),
            case.get(
                "payment_method_set",
                set()
            )
        )
    )

    # --------------------------------------------------------
    # Organizations / Entities
    # --------------------------------------------------------

    scores["organization"] = (
        jaccard_similarity(
            query.get(
                "organizations_set",
                set()
            ),
            case.get(
                "organizations_set",
                set()
            )
        )
    )

    # --------------------------------------------------------
    # Final weighted score
    # --------------------------------------------------------

    final_score = sum(
        WEIGHTS[key] * scores[key]
        for key in WEIGHTS
    )

    return final_score, scores


# ============================================================
# EXPLANATION GENERATOR
# ============================================================

def generate_explanation(scores):

    explanations = []

    # --------------------------------------------------------
    # Semantic
    # --------------------------------------------------------

    if scores["semantic"] >= 0.70:

        explanations.append(
            "Strong semantic similarity"
        )

    elif scores["semantic"] >= 0.50:

        explanations.append(
            "Moderate semantic similarity"
        )

    # --------------------------------------------------------
    # Crime
    # --------------------------------------------------------

    if scores["crime"] == 1.0:

        explanations.append(
            "Same crime subcategory"
        )

    elif scores["crime"] == 0.5:

        explanations.append(
            "Same crime category"
        )

    # --------------------------------------------------------
    # Structured signals
    # --------------------------------------------------------

    if scores["modus_operandi"] > 0:

        explanations.append(
            "Shared modus operandi"
        )

    if scores["deception"] > 0:

        explanations.append(
            "Shared deception pattern"
        )

    if scores["victim_action"] > 0:

        explanations.append(
            "Similar victim action"
        )

    if scores["attacker_action"] > 0:

        explanations.append(
            "Similar attacker action"
        )

    if scores["outcome"] > 0:

        explanations.append(
            "Similar outcome"
        )

    if scores["channel"] > 0:

        explanations.append(
            "Shared communication channel"
        )

    if scores["payment"] > 0:

        explanations.append(
            "Shared payment method"
        )

    if scores["organization"] > 0:

        explanations.append(
            "Shared organization/entity"
        )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if not explanations:

        explanations.append(
            "Similarity primarily based on complaint semantics"
        )

    return explanations


# ============================================================
# MAIN CASE LINKAGE FUNCTION
# ============================================================

def find_related_cases(
    incident,
    top_k=5
):

    # --------------------------------------------------------
    # Prepare query text
    # --------------------------------------------------------

    query_text = create_linkage_text(
        incident
    )

    # --------------------------------------------------------
    # Generate query embedding
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        [query_text],
        normalize_embeddings=True
    )

    # --------------------------------------------------------
    # Calculate semantic similarity
    # --------------------------------------------------------

    semantic_scores = cosine_similarity(
        query_embedding,
        historical_embeddings
    )[0]

    results = []

    # --------------------------------------------------------
    # Compare with every historical case
    # --------------------------------------------------------

    for index, row in df.iterrows():

        case = row.to_dict()

        # Create independent query dictionary
        query = dict(incident)

        # Convert all list fields to normalized sets
        for field in LIST_FIELDS:

            query[field + "_set"] = parse_list(
                query.get(
                    field,
                    []
                )
            )

        # ----------------------------------------------------
        # Calculate combined score
        # ----------------------------------------------------

        score, component_scores = (
            calculate_linkage_score(
                query,
                case,
                float(
                    semantic_scores[index]
                )
            )
        )

        # ----------------------------------------------------
        # Generate explanation
        # ----------------------------------------------------

        explanations = generate_explanation(
            component_scores
        )

        # ----------------------------------------------------
        # Relationship classification
        # ----------------------------------------------------

        if score >= 0.70:

            relationship = (
                "Potential relationship"
            )

        elif score >= 0.45:

            relationship = (
                "Possible shared pattern"
            )

        elif score >= 0.25:

            relationship = (
                "Weak similarity"
            )

        else:

            relationship = (
                "No significant relationship"
            )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append({

            "case_id": case.get(
                "complaint_id",
                f"CASE_{index}"
            ),

            "score": round(
                score,
                4
            ),

            "relationship": relationship,

            "shared_signals": explanations,

            "component_scores": {
                key: round(
                    value,
                    4
                )
                for key, value
                in component_scores.items()
            }
        })

    # ========================================================
    # SORT BY SCORE
    # ========================================================

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # ========================================================
    # ONLY RETURN MEANINGFUL MATCHES
    # ========================================================

    meaningful_results = [
        result
        for result in results
        if result["score"] >= 0.45
    ]

    return meaningful_results[:top_k]