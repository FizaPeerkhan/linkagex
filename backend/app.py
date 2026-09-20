from flask import Flask, request, jsonify
from flask_cors import CORS

from werkzeug.security import generate_password_hash, check_password_hash

from database import get_db_connection

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
# REGISTER
# ============================================================

@app.route("/api/register", methods=["POST"])
def register():

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "message": "No JSON data received"
            }), 400

        # ----------------------------------------------------
        # Get registration details
        # ----------------------------------------------------

        full_name = data.get("full_name", "").strip()
        email = data.get("email", "").strip().lower()
        mobile_number = data.get("mobile_number", "").strip()
        age_group = data.get("age_group", "").strip()
        profession = data.get("profession", "").strip()
        city_region = data.get("city_region", "").strip()
        preferred_language = data.get(
            "preferred_language", ""
        ).strip()
        password = data.get("password", "")

        # ----------------------------------------------------
        # Validate required fields
        # ----------------------------------------------------

        if not full_name:
            return jsonify({
                "success": False,
                "message": "Full name is required."
            }), 400

        if not email:
            return jsonify({
                "success": False,
                "message": "Email is required."
            }), 400

        if not mobile_number:
            return jsonify({
                "success": False,
                "message": "Mobile number is required."
            }), 400

        if not age_group:
            return jsonify({
                "success": False,
                "message": "Age group is required."
            }), 400

        if not profession:
            return jsonify({
                "success": False,
                "message": "Profession is required."
            }), 400

        if not city_region:
            return jsonify({
                "success": False,
                "message": "City/Region is required."
            }), 400

        if not password:
            return jsonify({
                "success": False,
                "message": "Password is required."
            }), 400

        # ----------------------------------------------------
        # Connect to database
        # ----------------------------------------------------

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # ----------------------------------------------------
        # Check whether email already exists
        # ----------------------------------------------------

        cursor.execute(
            "SELECT id FROM users WHERE email = %s",
            (email,)
        )

        existing_user = cursor.fetchone()

        if existing_user:

            cursor.close()
            conn.close()

            return jsonify({
                "success": False,
                "message": "An account with this email already exists."
            }), 409

        # ----------------------------------------------------
        # Hash password
        # ----------------------------------------------------

        password_hash = generate_password_hash(password)

        # ----------------------------------------------------
        # Insert new citizen
        # ----------------------------------------------------

        cursor.execute(
            """
            INSERT INTO users
            (
                full_name,
                email,
                mobile_number,
                age_group,
                profession,
                city_region,
                preferred_language,
                password_hash,
                role
            )
            VALUES
            (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s
            )
            """,
            (
                full_name,
                email,
                mobile_number,
                age_group,
                profession,
                city_region,
                preferred_language,
                password_hash,
                "citizen"
            )
        )

        conn.commit()

        cursor.close()
        conn.close()

        return jsonify({
            "success": True,
            "message": "Registration successful."
        }), 201

    except Exception as e:

        print("Registration error:", str(e))

        return jsonify({
            "success": False,
            "message": "Registration failed."
        }), 500


# ============================================================
# LOGIN
# ============================================================

@app.route("/api/login", methods=["POST"])
def login():

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "message": "No JSON data received"
            }), 400

        email = data.get("email", "").strip().lower()
        password = data.get("password", "")

        if not email or not password:

            return jsonify({
                "success": False,
                "message": "Email and password are required."
            }), 400

        # ----------------------------------------------------
        # Connect to database
        # ----------------------------------------------------

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # ----------------------------------------------------
        # Find user by email
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT
                id,
                full_name,
                email,
                mobile_number,
                age_group,
                profession,
                city_region,
                preferred_language,
                password_hash,
                role
            FROM users
            WHERE email = %s
            """,
            (email,)
        )

        user = cursor.fetchone()

        cursor.close()
        conn.close()

        # ----------------------------------------------------
        # Check account
        # ----------------------------------------------------

        if not user:

            return jsonify({
                "success": False,
                "message": "Invalid email or password."
            }), 401

        # ----------------------------------------------------
        # Verify password
        # ----------------------------------------------------

        if not check_password_hash(
            user["password_hash"],
            password
        ):

            return jsonify({
                "success": False,
                "message": "Invalid email or password."
            }), 401

        # ----------------------------------------------------
        # Never send password hash to frontend
        # ----------------------------------------------------

        user.pop("password_hash", None)

        return jsonify({
            "success": True,
            "message": "Login successful.",
            "user": user
        })

    except Exception as e:

        print("Login error:", str(e))

        return jsonify({
            "success": False,
            "message": "Login failed."
        }), 500

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