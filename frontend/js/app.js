/* =========================================================
   LINKAGEX — SHARED APPLICATION JAVASCRIPT
========================================================= */


/* =========================================================
   TOAST
========================================================= */

function toast(message) {

  const root =
    document.getElementById("toast-root") ||
    document.body;

  const el =
    document.createElement("div");

  el.className = "toast";
  el.textContent = message;

  root.appendChild(el);

  setTimeout(
    () => el.remove(),
    2800
  );
}


/* =========================================================
   STATUS CLASS
========================================================= */

function statusClass(s) {

  const x =
    String(s ?? "").toLowerCase();

  if (x.includes("new"))
    return "status-new";

  if (x.includes("reviewed"))
    return "status-reviewed";

  if (x.includes("linked"))
    return "status-linked";

  if (x.includes("unlinked"))
    return "status-unlinked";

  if (x.includes("flagged"))
    return "status-flagged";

  return "status-saved";
}


/* =========================================================
   HTML ESCAPE
========================================================= */

function esc(s) {

  return String(s ?? "").replace(
    /[&<>"']/g,
    m => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;"
    }[m])
  );

}


/* =========================================================
   GET AUTHENTICATED USER
========================================================= */

function getCurrentUser() {

  try {

    const storedUser =
      sessionStorage.getItem("csUser");

    if (!storedUser) {
      return null;
    }

    return JSON.parse(storedUser);

  }

  catch (error) {

    console.error(
      "Unable to read authenticated user:",
      error
    );

    return null;

  }

}


/* =========================================================
   INITIALIZE APPLICATION SHELL
========================================================= */

function initShell(role, active) {

  const user =
    getCurrentUser();


  /*
   * Role comes from the authenticated backend session.
   * It is NOT selected by the user on the frontend.
   */

  if (
    role !== "citizen" &&
    role !== "investigator"
  ) {

    console.error(
      "Invalid or missing account role."
    );

    window.location.href =
      "../login.html";

    return;

  }


  /* =======================================================
     NAVIGATION
  ======================================================= */

  const nav =
    role === "citizen"

      ? [

          [
            "Overview",
            "dashboard.html",
            "⌂"
          ],

          [
            "Report",
            "complaint.html",
            "＋"
          ],

          [
            "My Complaints",
            "complaints.html",
            "▤"
          ],

          [
            "Awareness",
            "awareness.html",
            "◌"
          ],

          [
            "Safety Center",
            "guidance.html",
            "✓"
          ],

          [
            "Account",
            "profile.html",
            "○"
          ]

        ]

      : [

          [
            "Overview",
            "dashboard.html",
            "⌂"
          ],

          [
            "Cases",
            "cases.html",
            "▤"
          ],

          [
            "New Cases",
            "cases.html?filter=new",
            "＋"
          ],

          [
            "Unlinked Cases",
            "cases.html?filter=unlinked",
            "○"
          ],

          [
            "Flagged Cases",
            "cases.html?filter=flagged",
            "⚑"
          ],

          [
            "Potential Linkages",
            "linkage.html",
            "⌁"
          ],

          [
            "Upload Report",
            "upload-report.html",
            "↑"
          ],

          [
            "Profile",
            "profile.html",
            "○"
          ],

          [
            "Settings",
            "#",
            "⚙"
          ]

        ];


  /* =======================================================
     SIDEBAR GROUPS
  ======================================================= */

  const groupedItems = [

    "Report",
    "Cases",
    "Potential Linkages",
    "Upload Report",
    "Profile"

  ];


  /* =======================================================
     BUILD SIDEBAR
  ======================================================= */

  const sidebarNavigation =

    nav.map(
      (n, i) => {

        const startsGroup =
          i === 0 ||
          groupedItems.includes(
            n[0]
          );


        const groupLabel =

          i === 0

            ? "Overview"

            : n[0] === "Profile"

              ? "Account"

              : n[0] === "Cases"

                ? "Cases"

                : n[0] === "Potential Linkages"

                  ? "Intelligence"

                  : "Intake";


        return `

          ${
            startsGroup

              ? `

                <div class="nav-group">

                  <div class="nav-label">
                    ${groupLabel}
                  </div>

              `

              : ""
          }


          <a
            class="nav-item ${
              active === n[0]
                ? "active"
                : ""
            }"
            href="${n[1]}"
          >

            <span class="nav-icon">
              ${n[2]}
            </span>

            ${esc(n[0])}

          </a>


          ${
            i === nav.length - 1
              ? "</div>"
              : ""
          }

        `;

      }
    ).join("");


  /* =======================================================
     SIDEBAR
  ======================================================= */

  document.body.insertAdjacentHTML(

    "afterbegin",

    `

      <aside
        class="sidebar"
        id="sidebar"
      >

        <div class="brand">

          <span
            class="brand-mark"
            aria-hidden="true"
          >

            <span></span>

          </span>


          <div>

            <strong>
              LINKAGEX
            </strong>

            <small>
              INTELLIGENCE
            </small>

          </div>

        </div>


        ${sidebarNavigation}

      </aside>

    `
  );


  /* =======================================================
     USER INFORMATION
  ======================================================= */

  const userName =

    user?.full_name ||
    user?.name ||
    (
      role === "citizen"
        ? "Citizen"
        : "Investigator"
    );


  const initials =

    userName

      .split(/\s+/)

      .filter(Boolean)

      .slice(0, 2)

      .map(
        word =>
          word
            .charAt(0)
            .toUpperCase()
      )

      .join("")

    ||

    (
      role === "citizen"
        ? "CZ"
        : "IN"
    );


  const accountLabel =

    role === "citizen"
      ? "Citizen Account"
      : "Investigator Account";


  /* =======================================================
     MAIN APPLICATION
  ======================================================= */

  document.body.insertAdjacentHTML(

    "afterbegin",

    `

      <div class="main">

        <header class="topbar">

          <div class="topbar-left">

            <button
              class="mobile-menu"
              id="mobileMenu"
              type="button"
              aria-label="Open navigation"
            >
              ☰
            </button>


            <div>

              <div class="topbar-title">

                ${
                  role === "citizen"
                    ? "Citizen Portal"
                    : "Investigator Intelligence"
                }

              </div>


              <div class="topbar-sub">

                ${
                  role === "citizen"
                    ? "Secure reporting & awareness"
                    : "Structured incident analysis & linkage support"
                }

              </div>

            </div>

          </div>


          <div class="user-chip">

            <span class="user-name">
              ${esc(userName)}
            </span>


            <span
              class="avatar"
              title="${esc(accountLabel)}"
            >
              ${esc(initials)}
            </span>

          </div>

        </header>


        <div
          class="content"
          id="content"
        ></div>

      </div>

    `
  );


  /* =======================================================
     MOBILE MENU
  ======================================================= */

  const mobileMenu =
    document.getElementById(
      "mobileMenu"
    );


  if (mobileMenu) {

    mobileMenu.onclick = () => {

      const sidebar =
        document.getElementById(
          "sidebar"
        );

      if (sidebar) {

        sidebar.classList.toggle(
          "open"
        );

      }

    };

  }

}


/* =========================================================
   PAGE HEADER
========================================================= */

function pageHead(
  title,
  sub,
  action = ""
) {

  return `

    <div class="page-head">

      <div>

        <div class="eyebrow-dark">

          ${
            title.includes("Case")
              ? "CASE INTELLIGENCE"
              : "LINKAGEX INTELLIGENCE"
          }

        </div>


        <h1>
          ${esc(title)}
        </h1>


        <p>
          ${esc(sub)}
        </p>

      </div>


      ${action}

    </div>

  `;

}


/* =========================================================
   APPLICATION SHELL
========================================================= */

function shell(
  role,
  active
) {

  document.body.innerHTML = "";

  initShell(
    role,
    active
  );

  return document.getElementById(
    "content"
  );

}


/* =========================================================
   STATUS BADGE
========================================================= */

function badge(s) {

  return `

    <span
      class="status ${statusClass(s)}"
    >

      ${esc(s)}

    </span>

  `;

}


/* =========================================================
   INCIDENT VALUE FORMATTER
========================================================= */

function formatIncidentValue(
  value,
  key = ""
) {

  if (
    value === null ||
    value === undefined ||
    value === ""
  ) {

    return "Not provided";

  }


  /* -------------------------------------------------------
     AMOUNTS
  ------------------------------------------------------- */

  if (key === "amounts") {

    const amounts =
      Array.isArray(value)
        ? value
        : [value];


    const validAmounts =
      amounts.filter(
        amount =>
          amount !== null &&
          amount !== undefined &&
          amount !== ""
      );


    if (
      validAmounts.length === 0
    ) {

      return "Not provided";

    }


    return validAmounts
      .map(
        amount => {

          const numeric =
            Number(amount);


          if (
            Number.isNaN(numeric)
          ) {

            return String(amount);

          }


          return (
            "₹" +
            numeric.toLocaleString(
              "en-IN"
            )
          );

        }
      )
      .join(", ");

  }


  /* -------------------------------------------------------
     ARRAYS
  ------------------------------------------------------- */

  if (Array.isArray(value)) {

    if (value.length === 0) {
      return "Not provided";
    }


    return value
      .map(
        item =>
          String(item).trim()
      )
      .filter(Boolean)
      .join(", ");

  }


  /* -------------------------------------------------------
     EMPTY STRINGS
  ------------------------------------------------------- */

  if (
    typeof value === "string" &&
    value.trim() === ""
  ) {

    return "Not provided";

  }


  return String(value);

}


/* =========================================================
   INCIDENT FIELD
========================================================= */

function field(
  label,
  value,
  key,
  editable = false
) {

  const displayValue =
    formatIncidentValue(
      value,
      key
    );


  return `

    <div class="field">

      <div class="field-label">

        ${esc(label)}

      </div>


      <div
        class="field-value"
        data-field="${esc(key)}"
      >

        ${esc(displayValue)}


        ${
          editable

            ? `

              <button
                type="button"
                class="edit-link"
                onclick="editField('${esc(key)}')"
              >
                Edit
              </button>

            `

            : ""

        }

      </div>

    </div>

  `;

}


/* =========================================================
   INCIDENT FIELD DEFINITIONS

   These fields match the LinkageX NLP schema.
========================================================= */

const INCIDENT_FIELD_DEFINITIONS = [

  {
    label: "Crime Category",
    key: "crime_category"
  },

  {
    label: "Crime Type",
    key: "crime_subcategory"
  },

  {
    label: "Incident Date",
    key: "incident_date"
  },

  {
    label: "Modus Operandi",
    key: "modus_operandi"
  },

  {
    label: "Deception Method",
    key: "deception"
  },

  {
    label: "Victim Action",
    key: "victim_action"
  },

  {
    label: "Attacker Action",
    key: "attacker_action"
  },

  {
    label: "Outcome",
    key: "outcome"
  },

  {
    label: "Channel",
    key: "channels"
  },

  {
    label: "Payment Method",
    key: "payment_method"
  },

  {
    label: "Amount",
    key: "amounts"
  },

  {
    label: "Organizations",
    key: "organizations"
  },

  {
    label: "Location",
    key: "locations"
  }

];


/* =========================================================
   RENDER STRUCTURED INCIDENT
========================================================= */

function renderIncident(
  incident = {},
  editable = false
) {

  return `

    <div class="card">

      <div class="section-head">

        <h3>
          Structured Incident
        </h3>

        <span>
          Standardized incident representation
        </span>

      </div>


      <div class="incident-grid">

        ${
          INCIDENT_FIELD_DEFINITIONS
            .map(
              definition =>

                field(
                  definition.label,
                  incident[
                    definition.key
                  ],
                  definition.key,
                  editable
                )

            )
            .join("")
        }

      </div>

    </div>

  `;

}


/* =========================================================
   LOAD CURRENT ANALYSIS
========================================================= */

function getStoredAnalysis() {

  const stored =
    sessionStorage.getItem(
      "linkageX_analysis"
    );


  if (!stored) {
    return null;
  }


  try {

    const result =
      JSON.parse(stored);


    if (
      !result ||
      !result.data ||
      !result.data.incident
    ) {

      return null;

    }


    return result;

  }

  catch (error) {

    console.error(
      "Unable to parse LinkageX analysis:",
      error
    );

    return null;

  }

}


/* =========================================================
   SAVE CURRENT ANALYSIS
========================================================= */

function saveStoredAnalysis(
  result
) {

  sessionStorage.setItem(
    "linkageX_analysis",
    JSON.stringify(result)
  );

}


/* =========================================================
   EDIT INCIDENT FIELD
========================================================= */

function editField(key) {

  const el =
    document.querySelector(
      `[data-field="${key}"]`
    );


  if (!el) return;


  if (
    el.querySelector("input")
  ) {

    return;

  }


  const result =
    getStoredAnalysis();


  if (!result) {

    toast(
      "Complaint information could not be found."
    );

    return;

  }


  const incident =
    result.data.incident;


  const originalValue =
    incident[key];


  let inputValue = "";


  if (
    Array.isArray(
      originalValue
    )
  ) {

    inputValue =
      originalValue.join(", ");

  }

  else if (
    originalValue !== null &&
    originalValue !== undefined
  ) {

    inputValue =
      String(originalValue);

  }


  /* -------------------------------------------------------
     DATE
  ------------------------------------------------------- */

  if (
    key === "incident_date"
  ) {

    el.innerHTML = `

      <input
        type="date"
        class="incident-edit-input"
        value="${esc(inputValue)}"
        aria-label="Edit incident date"
      >


      <button
        type="button"
        class="edit-link"
        onclick="saveEdited('${esc(key)}')"
      >
        Save
      </button>


      <button
        type="button"
        class="edit-link"
        onclick="cancelEdit('${esc(key)}')"
      >
        Cancel
      </button>

    `;

  }


  /* -------------------------------------------------------
     NORMAL TEXT
  ------------------------------------------------------- */

  else {

    if (
      key === "amounts"
    ) {

      const amount =
        Array.isArray(
          originalValue
        )
          ? originalValue[0]
          : originalValue;


      inputValue =
        amount !== null &&
        amount !== undefined
          ? String(amount)
          : "";

    }


    el.innerHTML = `

      <input
        type="text"
        class="incident-edit-input"
        value="${esc(inputValue)}"
        aria-label="Edit ${esc(key)}"
      >


      <button
        type="button"
        class="edit-link"
        onclick="saveEdited('${esc(key)}')"
      >
        Save
      </button>


      <button
        type="button"
        class="edit-link"
        onclick="cancelEdit('${esc(key)}')"
      >
        Cancel
      </button>

    `;

  }


  const input =
    el.querySelector("input");


  if (input) {

    input.focus();


    /*
     * Do not select date inputs because
     * browsers handle date selection themselves.
     */

    if (
      key !== "incident_date"
    ) {

      input.select();

    }

  }

}


/* =========================================================
   CANCEL EDIT
========================================================= */

function cancelEdit(key) {

  const result =
    getStoredAnalysis();


  if (!result) {
    return;
  }


  const incident =
    result.data.incident;


  const el =
    document.querySelector(
      `[data-field="${key}"]`
    );


  if (!el) {
    return;
  }


  el.innerHTML = `

    ${esc(
      formatIncidentValue(
        incident[key],
        key
      )
    )}


    <button
      type="button"
      class="edit-link"
      onclick="editField('${esc(key)}')"
    >
      Edit
    </button>

  `;

}


/* =========================================================
   NORMALIZE LIST VALUE
========================================================= */

function normalizeListValue(
  value
) {

  if (
    value === null ||
    value === undefined
  ) {

    return [];

  }


  return String(value)
    .split(",")
    .map(
      item =>
        item.trim()
    )
    .filter(Boolean);

}


/* =========================================================
   SAVE EDITED INCIDENT FIELD
========================================================= */

function saveEdited(key) {

  const el =
    document.querySelector(
      `[data-field="${key}"]`
    );


  if (!el) return;


  const input =
    el.querySelector("input");


  if (!input) return;


  const result =
    getStoredAnalysis();


  if (!result) {

    toast(
      "Complaint information could not be found."
    );

    return;

  }


  const incident =
    result.data.incident;


  const rawValue =
    input.value.trim();


  /* =======================================================
     INCIDENT DATE
  ======================================================= */

  if (
    key === "incident_date"
  ) {

    incident.incident_date =
      rawValue;

  }


  /* =======================================================
     AMOUNT
  ======================================================= */

  else if (
    key === "amounts"
  ) {

    const cleaned =
      rawValue
        .replace(/₹/g, "")
        .replace(/,/g, "")
        .trim();


    if (
      cleaned === ""
    ) {

      incident.amounts = [];

    }

    else {

      const numericValue =
        Number(cleaned);


      if (
        Number.isNaN(
          numericValue
        ) ||
        numericValue < 0
      ) {

        toast(
          "Please enter a valid amount."
        );

        return;

      }


      incident.amounts = [
        numericValue
      ];

    }

  }


  /* =======================================================
     LIST FIELDS
  ======================================================= */

  else if (

    [
      "modus_operandi",
      "deception",
      "victim_action",
      "attacker_action",
      "outcome",
      "channels",
      "payment_method",
      "organizations",
      "locations",
      "phones",
      "emails",
      "upi_ids",
      "urls"
    ].includes(key)

  ) {

    incident[key] =
      normalizeListValue(
        rawValue
      );

  }


  /* =======================================================
     NORMAL TEXT FIELDS
  ======================================================= */

  else {

    incident[key] =
      rawValue;

  }


  /* =======================================================
     UPDATE MISSING FIELDS
  ======================================================= */

  let missingFields =

    Array.isArray(
      result.data.missing_fields
    )

      ? [
          ...result.data.missing_fields
        ]

      : [];


  const currentValue =
    incident[key];


  const hasValue =

    Array.isArray(
      currentValue
    )

      ? currentValue.length > 0

      : String(
          currentValue ?? ""
        ).trim() !== "";


  if (hasValue) {

    missingFields =
      missingFields.filter(
        fieldName =>
          fieldName !== key
      );

  }

  else {

    if (
      !missingFields.includes(key)
    ) {

      missingFields.push(key);

    }

  }


  result.data.incident =
    incident;

  result.data.missing_fields =
    missingFields;


  /* =======================================================
     PERSIST UPDATED ANALYSIS
  ======================================================= */

  saveStoredAnalysis(
    result
  );


  /* =======================================================
     UPDATE DISPLAY
  ======================================================= */

  const displayValue =
    formatIncidentValue(
      incident[key],
      key
    );


  el.innerHTML = `

    ${esc(displayValue)}


    <button
      type="button"
      class="edit-link"
      onclick="editField('${esc(key)}')"
    >
      Edit
    </button>

  `;


  toast(
    "Field updated successfully."
  );

}


/* =========================================================
   BUILD FINALIZED CASE
========================================================= */

function buildFinalizedCase(
  victimType = "me"
) {

  const result =
    getStoredAnalysis();


  if (!result) {
    return null;
  }


  const data =
    result.data || {};


  const incident =
    {
      ...(data.incident || {})
    };


  return {

    complaint_text:
      data.complaint_text || "",

    incident:
      incident,

    missing_fields: [],

    victim_type:
      victimType,

    finalized_at:
      new Date().toISOString()

  };

}


/* =========================================================
   SAVE FINALIZED CASE
========================================================= */

function saveFinalizedCase(
  victimType = "me"
) {

  const finalizedCase =
    buildFinalizedCase(
      victimType
    );


  if (!finalizedCase) {

    toast(
      "Unable to finalize complaint."
    );

    return false;

  }


  sessionStorage.setItem(
    "linkageX_final_case",
    JSON.stringify(
      finalizedCase
    )
  );


  /*
   * Keep the analysis synchronized
   * with the finalized incident.
   */

  const result =
    getStoredAnalysis();


  if (result) {

    result.data.missing_fields =
      [];

    result.data.incident =
      finalizedCase.incident;

    saveStoredAnalysis(
      result
    );

  }


  return true;

}


/* =========================================================
   LOGOUT
========================================================= */

function logout() {

  sessionStorage.clear();

  window.location.href =
    "../login.html";

}