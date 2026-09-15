from flask import Flask, request, jsonify
from flask_cors import CORS

from services.nlp_service import analyze_complaint
from services.linkage_service import find_related_cases


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

@app.route(
    "/api/analyze",
    methods=["POST"]
)
def analyze():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "status": "error",
                "message": "No JSON data received"
            }), 400

        complaint_text = data.get(
            "complaint_text",
            ""
        ).strip()

        if not complaint_text:

            return jsonify({
                "status": "error",
                "message": "Complaint text is required"
            }), 400

        result = analyze_complaint(
            complaint_text
        )

        return jsonify({
            "status": "success",
            "data": result
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ============================================================
# CASE LINKAGE
# ============================================================

@app.route(
    "/api/linkage",
    methods=["POST"]
)
def linkage():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "status": "error",
                "message": "No JSON data received"
            }), 400

        # ----------------------------------------------------
        # The frontend sends the structured incident
        # produced by the NLP stage.
        # ----------------------------------------------------

        incident = data.get(
            "incident",
            data
        )

        if not incident:

            return jsonify({
                "status": "error",
                "message": "Incident data is required"
            }), 400

        # ----------------------------------------------------
        # Run historical case linkage
        # ----------------------------------------------------

        results = find_related_cases(
            incident,
            top_k=5
        )

        # ----------------------------------------------------
        # Return results
        # ----------------------------------------------------

        return jsonify({

            "status": "success",

            "match_count": len(results),

            "results": results
        })

    except Exception as e:

        print(
            "Linkage error:",
            str(e)
        )

        return jsonify({

            "status": "error",

            "message": str(e)

        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )