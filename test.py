from backend.services.linkage_service import find_related_cases


# ============================================================
# TEST COMPLAINT
# ============================================================

test_incident = {
    "complaint_text": (
        "I received a WhatsApp message offering me a personal loan. "
        "They asked me to pay Rs. 15000 as a processing fee and after "
        "I paid, they stopped responding."
    ),

    "crime_category": "Financial Fraud",

    "crime_subcategory": "Investment/Loan Scams",

    "modus_operandi": [
        "Fake Processing Fee",
        "Fake Loan Offer"
    ],

    "deception": [
        "Deceptive Offer"
    ],

    "victim_action": [
        "Paid"
    ],

    "attacker_action": [
        "Requested Payment"
    ],

    "outcome": [
        "Scammer Stopped Responding"
    ],

    "channels": [
        "WhatsApp"
    ],

    "amounts": [
        15000.0
    ],

    "payment_method": [],

    "phones": [],

    "emails": [],

    "upi_ids": [],

    "urls": [],

    "organizations": [],

    "locations": []
}


# ============================================================
# RUN LINKAGE
# ============================================================

results = find_related_cases(
    test_incident,
    top_k=5
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 70)
print("LINKAGEX — RELATED CASES")
print("=" * 70)


if not results:

    print()
    print("No meaningful related cases found.")
    print()
    print("Threshold used: 0.45")
    print("=" * 70)

else:

    for i, result in enumerate(
        results,
        start=1
    ):

        print()
        print(f"#{i}")

        print(
            f"Case ID       : "
            f"{result['case_id']}"
        )

        print(
            f"Score         : "
            f"{result['score']}"
        )

        print(
            f"Relationship  : "
            f"{result['relationship']}"
        )

        print(
            "Shared Signals:"
        )

        for signal in result[
            "shared_signals"
        ]:

            print(
                f"  - {signal}"
            )

        print(
            "Component Scores:"
        )

        for key, value in result[
            "component_scores"
        ].items():

            print(
                f"  {key:18}: {value}"
            )

        print("-" * 70)