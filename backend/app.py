from flask import Flask, request, jsonify  # type: ignore[import-not-found]
from flask_cors import CORS  # type: ignore[import-not-found]

from services.nlp_service import analyze_complaint
from services.linkage_service import find_related_cases, get_case_by_id


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return jsonify({
        "status": "success",
        "message": "LinkageX backend is running"
    })


# ============================================================
# NLP ANALYSIS
# ============================================================

@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({
                "status": "error",
                "message": "No JSON data received"
            }), 400

        complaint_text = data.get("complaint_text", "")
        if not isinstance(complaint_text, str):
            return jsonify({
                "status": "error",
                "message": "Complaint text must be a string"
            }), 400
        complaint_text = complaint_text.strip()
        if not complaint_text:
            return jsonify({
                "status": "error",
                "message": "Complaint text is required"
            }), 400

        result = analyze_complaint(complaint_text)
        return jsonify({"status": "success", "data": result})

    except Exception as e:
        app.logger.exception("NLP analysis error")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ============================================================
# CASE LINKAGE
# ============================================================

@app.route("/api/linkage", methods=["POST"])
def linkage():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({
                "status": "error",
                "message": "No JSON data received"
            }), 400

        incident = data.get("incident", data)
        if not isinstance(incident, dict) or not incident:
            return jsonify({
                "status": "error",
                "message": "Incident data must be a non-empty JSON object"
            }), 400

        # Keep the original complaint text for semantic retrieval.
        # Structured incident fields alone can lose useful narrative detail.
        complaint_text = data.get("complaint_text")
        if isinstance(complaint_text, str) and complaint_text.strip():
            incident = dict(incident)
            incident.setdefault("complaint_text", complaint_text.strip())

        results = find_related_cases(incident, top_k=5)

        return jsonify({
            "status": "success",
            "requested_count": 5,
            "match_count": len(results),
            "results": results,
            "note": (
                "Results are ranked similarities, not proof that incidents are connected."
            )
        })

    except Exception as e:
        app.logger.exception("Case linkage error")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ============================================================
# MASTER DATASET CASE DETAILS
# ============================================================

@app.route("/api/cases/<case_id>", methods=["GET"])
def case_details(case_id):
    try:
        case = get_case_by_id(case_id)
        if case is None:
            return jsonify({
                "status": "error",
                "message": f"Case {case_id} was not found in LINKAGEX_MASTER_CASES_V2.csv"
            }), 404
        return jsonify({"status": "success", "case": case})
    except Exception as e:
        app.logger.exception("Case details lookup error")
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
