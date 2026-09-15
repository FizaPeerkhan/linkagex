function toast(message) {
  const root = document.getElementById("toast-root") || document.body;

  const el = document.createElement("div");
  el.className = "toast";
  el.textContent = message;

  root.appendChild(el);

  setTimeout(() => el.remove(), 2800);
}


function statusClass(s) {
  const x = String(s ?? "").toLowerCase();

  if (x.includes("new")) return "status-new";
  if (x.includes("reviewed")) return "status-reviewed";
  if (x.includes("linked")) return "status-linked";
  if (x.includes("unlinked")) return "status-unlinked";
  if (x.includes("flagged")) return "status-flagged";

  return "status-saved";
}


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
    const storedUser = sessionStorage.getItem("csUser");

    if (!storedUser) {
      return null;
    }

    return JSON.parse(storedUser);

  } catch (error) {
    console.error("Unable to read authenticated user:", error);
    return null;
  }
}


/* =========================================================
   INITIALIZE APPLICATION SHELL
========================================================= */

function initShell(role, active) {

  const user = getCurrentUser();

  /*
   * Role comes from the authenticated backend session.
   * It is NOT selected by the user on the frontend.
   */

  if (role !== "citizen" && role !== "investigator") {
    console.error("Invalid or missing account role.");
    window.location.href = "../login.html";
    return;
  }


  /* =======================================================
     NAVIGATION
  ======================================================= */

  const nav = role === "citizen"
    ? [
        ["Overview", "dashboard.html", "⌂"],
        ["Report", "complaint.html", "＋"],
        ["My Complaints", "complaints.html", "▤"],
        ["Awareness", "awareness.html", "◌"],
        ["Safety Center", "guidance.html", "✓"],
        ["Account", "profile.html", "○"]
      ]

    : [
        ["Overview", "dashboard.html", "⌂"],
        ["Cases", "cases.html", "▤"],
        ["New Cases", "cases.html?filter=new", "＋"],
        ["Unlinked Cases", "cases.html?filter=unlinked", "○"],
        ["Flagged Cases", "cases.html?filter=flagged", "⚑"],
        ["Potential Linkages", "linkage.html", "⌁"],
        ["Incident Clusters", "clusters.html", "◉"],
        ["Upload Report", "upload-report.html", "↑"],
        ["Profile", "profile.html", "○"],
        ["Settings", "#", "⚙"]
      ];


  /* =======================================================
     SIDEBAR
  ======================================================= */

  const groupedItems = [
    "Report",
    "Cases",
    "Potential Linkages",
    "Upload Report",
    "Profile"
  ];

  const sidebarNavigation = nav.map((n, i) => {

    const startsGroup =
      i === 0 || groupedItems.includes(n[0]);

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
          ? `<div class="nav-group">
               <div class="nav-label">${groupLabel}</div>`
          : ""
      }

      <a
        class="nav-item ${active === n[0] ? "active" : ""}"
        href="${n[1]}"
      >
        <span class="nav-icon">${n[2]}</span>
        ${esc(n[0])}
      </a>

      ${
        i === nav.length - 1
          ? "</div>"
          : ""
      }
    `;
  }).join("");


  document.body.insertAdjacentHTML(
    "afterbegin",
    `
      <aside class="sidebar" id="sidebar">

        <div class="brand">

          <span class="brand-mark">
            <span></span>
          </span>

          <div>
            <strong>CYBERSHIELD</strong>
            <small>INTELLIGENCE</small>
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
    (role === "citizen"
      ? "Citizen"
      : "Investigator");

  const initials =
    userName
      .split(/\s+/)
      .filter(Boolean)
      .slice(0, 2)
      .map(word => word.charAt(0).toUpperCase())
      .join("") ||
    (role === "citizen" ? "CZ" : "IN");


  const accountLabel =
    role === "citizen"
      ? "Citizen Account"
      : "Investigator Account";


  /* =======================================================
     MAIN CONTENT SHELL
  ======================================================= */

  document.body.insertAdjacentHTML(
    "afterbegin",
    `
      <div class="main">

        <header class="topbar">

          <div>

            <button
              class="mobile-menu"
              id="mobileMenu"
              type="button"
              aria-label="Open navigation"
            >
              ☰
            </button>

            <span class="topbar-title">
              ${
                role === "citizen"
                  ? "Citizen Portal"
                  : "Investigator Intelligence"
              }
            </span>

            <div class="topbar-sub">
              ${
                role === "citizen"
                  ? "Secure reporting & awareness"
                  : "Structured incident analysis & linkage support"
              }
            </div>

          </div>


          <div class="user-chip">

            <span>
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
    document.getElementById("mobileMenu");

  if (mobileMenu) {

    mobileMenu.onclick = () => {

      const sidebar =
        document.getElementById("sidebar");

      if (sidebar) {
        sidebar.classList.toggle("open");
      }

    };

  }
}


/* =========================================================
   PAGE HEADER
========================================================= */

function pageHead(title, sub, action = "") {

  return `
    <div class="page-head">

      <div>

        <div class="eyebrow-dark">
          ${
            title.includes("Case")
              ? "CASE INTELLIGENCE"
              : "CYBERSHIELD INTELLIGENCE"
          }
        </div>

        <h1>${esc(title)}</h1>

        <p>${esc(sub)}</p>

      </div>

      ${action}

    </div>
  `;
}


/* =========================================================
   APPLICATION SHELL
========================================================= */

function shell(role, active) {

  document.body.innerHTML = "";

  initShell(role, active);

  return document.getElementById("content");
}


/* =========================================================
   STATUS BADGE
========================================================= */

function badge(s) {

  return `
    <span class="status ${statusClass(s)}">
      ${esc(s)}
    </span>
  `;
}


/* =========================================================
   STRUCTURED INCIDENT FIELD
========================================================= */

function field(
  label,
  value,
  key,
  editable = false
) {

  return `
    <div class="field">

      <div class="field-label">
        ${esc(label)}
      </div>

      <div
        class="field-value"
        data-field="${esc(key)}"
      >

        ${esc(value)}

        ${
          editable
            ? `
              <button
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
   RENDER STRUCTURED INCIDENT
========================================================= */

function renderIncident(
  incident = CS.incident,
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

        ${field(
          "Crime Category",
          incident.crimeCategory,
          "crimeCategory",
          editable
        )}

        ${field(
          "Crime Type",
          incident.crimeType,
          "crimeType",
          editable
        )}

        ${field(
          "Channel",
          incident.channel,
          "channel",
          editable
        )}

        ${field(
          "Attacker / Entity",
          incident.attackerEntity,
          "attackerEntity",
          editable
        )}

        ${field(
          "Deception Method",
          incident.deceptionMethod,
          "deceptionMethod",
          editable
        )}

        ${field(
          "Victim Action",
          incident.victimAction,
          "victimAction",
          editable
        )}

        ${field(
          "Amount Lost",
          incident.amount
            ? "₹" + incident.amount.toLocaleString("en-IN")
            : "Not provided",
          "amount",
          editable
        )}

        ${field(
          "Outcome",
          incident.outcome,
          "outcome",
          editable
        )}

        ${field(
          "Incident Date",
          incident.incidentDate || "Not provided",
          "incidentDate",
          editable
        )}

        ${field(
          "Location",
          incident.location || "Not provided",
          "location",
          editable
        )}

      </div>

    </div>
  `;
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

  const current =
    el.textContent
      .replace(" Edit", "")
      .trim();

  el.innerHTML = `
    <input
      value="${esc(
        current
          .replace(/^₹/, "")
          .replaceAll(",", "")
      )}"
    >

    <button
      class="edit-link"
      onclick="saveEdited('${esc(key)}')"
    >
      Save
    </button>
  `;
}


/* =========================================================
   SAVE EDITED FIELD
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

  let val = input.value;

  if (key === "amount") {

    const numericValue =
      Number(
        val
          .replace(/,/g, "")
          .replace(/[₹]/g, "")
      );

    val =
      numericValue
        ? "₹" + numericValue.toLocaleString("en-IN")
        : "Not provided";
  }


  el.textContent = val + " ";

  el.insertAdjacentHTML(
    "beforeend",
    `
      <button
        class="edit-link"
        onclick="editField('${esc(key)}')"
      >
        Edit
      </button>
    `
  );

  toast("Field updated for review.");
}


/* =========================================================
   LOGOUT
========================================================= */

function logout() {

  sessionStorage.clear();

  /*
   * login.html is located directly inside /frontend.
   * Citizen/investigator pages are one directory deeper,
   * so ../login.html is the correct path.
   */

  window.location.href = "../login.html";
}