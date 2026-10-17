/* =========================================================
   LINKAGEX — SHARED APPLICATION JAVASCRIPT
========================================================= */

/* =========================================================
   TOAST
========================================================= */

function toast(message) {
  const root = document.getElementById("toast-root") || document.body;
  const el = document.createElement("div");

  el.className = "toast";
  el.textContent = message;
  root.appendChild(el);

  setTimeout(() => el.remove(), 2800);
}

/* =========================================================
   STATUS CLASS
========================================================= */

function statusClass(s) {
  const x = String(s ?? "").toLowerCase();

  if (x.includes("new")) return "status-new";
  if (x.includes("reviewed")) return "status-reviewed";
  if (x.includes("linked")) return "status-linked";
  if (x.includes("unlinked")) return "status-unlinked";
  if (x.includes("flagged")) return "status-flagged";

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
    const storedUser = sessionStorage.getItem("csUser");

    if (!storedUser) return null;

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

  if (role !== "citizen" && role !== "investigator") {
    console.error("Invalid or missing account role.");
    window.location.href = "../login.html";
    return;
  }

  const userName = String(
    user?.full_name ||
    user?.name ||
    (role === "citizen" ? "Citizen" : "Investigator")
  );

  const initials = userName
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map(word => word.charAt(0).toUpperCase())
    .join("") || (role === "citizen" ? "CZ" : "IN");

  const accountLabel =
    role === "citizen" ? "Citizen Account" : "Investigator Account";

  const roleLabel =
    role === "citizen" ? "Citizen Portal" : "Investigator Portal";

  const icons = {
    dashboard: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><rect x="4" y="4" width="6" height="6"/><rect x="14" y="4" width="6" height="6"/><rect x="4" y="14" width="6" height="6"/><rect x="14" y="14" width="6" height="6"/></svg>',

    report: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M12 5v14M5 12h14"/></svg>',

    cases: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/></svg>',

    awareness: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M12 3l7 3v6c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6z"/><path d="M12 8v5M12 16.2h.01"/></svg>',

    guidance: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M9.5 9.3a2.5 2.5 0 014.9.8c0 1.7-2.4 2-2.4 3.6M12 17.2h.01"/></svg>',

    link: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M10 13.5l4-4"/><path d="M7.8 15.8l-1.5 1.5a3.5 3.5 0 01-5-5l4-4a3.5 3.5 0 015 0" transform="translate(2 0)"/><path d="M16.2 8.2l1.5-1.5a3.5 3.5 0 015 5l-4 4a3.5 3.5 0 01-5 0" transform="translate(-2 0)"/></svg>',

    upload: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M12 16V4m-5 5l5-5 5 5"/><path d="M4 16v4h16v-4"/></svg>',

    cluster: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><circle cx="5" cy="6" r="2"/><circle cx="18.5" cy="5" r="2"/><circle cx="12" cy="18" r="2"/><path d="M7 7l3.7 8.8M16.7 6.8l-3.6 9"/></svg>',

    profile: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><circle cx="12" cy="8" r="3.5"/><path d="M5 20c.7-3.4 3.2-5.2 7-5.2s6.3 1.8 7 5.2"/></svg>',

    signout: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4"/><path d="M16 17l5-5-5-5M21 12H9"/></svg>'
  };

  const nav = role === "citizen"
    ? [
        {
          label: "Dashboard",
          href: "dashboard.html",
          icon: "dashboard",
          active: ["Dashboard", "Overview"]
        },
        {
          label: "Submit Complaint",
          href: "complaint.html",
          icon: "report",
          active: ["Submit Complaint", "Report"]
        },
        {
          label: "My Complaints",
          href: "complaints.html",
          icon: "cases",
          active: ["My Complaints"]
        },
        {
          label: "Regional Awareness",
          href: "awareness.html",
          icon: "awareness",
          active: ["Regional Awareness", "Awareness"]
        },
        {
          label: "Safety Center",
          href: "guidance.html",
          icon: "guidance",
          active: ["Safety Center"]
        }
      ]
    : [
        {
          label: "Dashboard",
          href: "dashboard.html",
          icon: "dashboard",
          active: ["Dashboard", "Overview"]
        },
        {
          label: "Cases",
          href: "cases.html",
          icon: "cases",
          active: ["Cases"]
        },
        {
          label: "New Cases",
          href: "cases.html?filter=new",
          icon: "report",
          active: []
        },
        {
          label: "Unlinked Cases",
          href: "cases.html?filter=unlinked",
          icon: "cases",
          active: []
        },
        {
          label: "Flagged Cases",
          href: "cases.html?filter=flagged",
          icon: "awareness",
          active: []
        },
        {
          label: "Potential Linkages",
          href: "linkage.html",
          icon: "link",
          active: ["Potential Linkages"]
        },
        {
          label: "Incident Clusters",
          href: "clusters.html",
          icon: "cluster",
          active: []
        },
        {
          label: "Upload Report",
          href: "upload-report.html",
          icon: "upload",
          active: ["Upload Report"]
        },
        {
          label: "Profile",
          href: "profile.html",
          icon: "profile",
          active: ["Profile"]
        }
      ];

  const brandSvg = `
    <svg viewBox="0 0 30 30" aria-hidden="true">
      <circle cx="8" cy="9" r="2.6" fill="#54d9ff"/>
      <circle cx="22" cy="8" r="2.2" fill="#54d9ff"/>
      <circle cx="15" cy="21" r="2.6" fill="#ffb454"/>
      <path d="M8 9L22 8M8 9L15 21M22 8L15 21"
            stroke="#54d9ff" stroke-width="1" opacity=".55"/>
    </svg>`;

  /*
   * Shared layout styles.
   * They are applied to all pages that use shell().
   */
  if (!document.getElementById("linkage-shell-overrides")) {
    const style = document.createElement("style");
    style.id = "linkage-shell-overrides";

    style.textContent = `
      :root {
        --sidebar-w: 248px;
        --topbar-h: 64px;
      }

      body {
        margin: 0;
        min-height: 100vh;
        overflow-x: hidden;
      }

      .app-shell {
        position: relative;
        display: flex;
        align-items: stretch;
        width: 100%;
        min-height: 100vh;
      }

      .app-shell::before {
        content: "";
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        background-image:
          linear-gradient(rgba(148,197,224,.14) 1px, transparent 1px),
          linear-gradient(90deg, rgba(148,197,224,.14) 1px, transparent 1px);
        background-size: 64px 64px;
        opacity: .22;
        mask-image: radial-gradient(
          ellipse 90% 70% at 20% 0%,
          black,
          transparent 75%
        );
      }

      /* Sidebar */
      .app-shell .sidebar {
        position: relative !important;
        inset: auto !important;
        z-index: 2;
        flex: 0 0 248px !important;
        width: 248px !important;
        min-width: 248px !important;
        max-width: 248px !important;
        height: auto !important;
        min-height: 100vh;
        max-height: none !important;
        overflow-y: visible;
        display: flex;
        flex-direction: column;
        padding: 22px 0 18px !important;
        margin: 0 !important;
        transform: none;
        background: #0a0f18 !important;
        border: 0;
        border-right: 1px solid rgba(148,197,224,.14) !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        color: #e9f1f8;
      }

      .app-shell .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0 22px 22px;
        margin: 0 0 6px;
        border: 0;
        border-bottom: 1px solid rgba(148,197,224,.14);
        border-radius: 0;
        color: #e9f1f8;
      }

      .app-shell .sidebar-brand svg {
        display: block;
        width: 26px;
        height: 26px;
        flex: 0 0 26px;
      }

      .app-shell .sidebar-brand span {
        color: #e9f1f8;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 16px;
        font-weight: 600;
        letter-spacing: 0;
      }

      .app-shell .sidebar-role {
        padding: 16px 22px 10px;
        color: #54d9ff;
        font: 500 .68rem 'IBM Plex Mono', monospace;
        letter-spacing: .04em;
        text-transform: uppercase;
      }

      .app-shell .sidebar-nav {
        display: flex;
        flex: 1;
        flex-direction: column;
        gap: 1px;
        margin: 0;
        padding: 0;
        list-style: none;
      }

      .app-shell .sidebar-nav li {
        margin: 0;
        padding: 0;
      }

      .app-shell .sidebar-nav a {
        display: flex;
        align-items: center;
        gap: 12px;
        min-height: 0;
        margin: 0;
        padding: 10px 22px;
        border: 0;
        border-left: 2px solid transparent;
        border-radius: 0;
        background: transparent;
        color: #8fa0b6;
        font: 400 .9rem 'IBM Plex Sans', sans-serif;
        text-decoration: none;
        box-shadow: none;
        transition: color .18s, background .18s, border-color .18s;
      }

      .app-shell .sidebar-nav a svg {
        display: block;
        width: 17px;
        height: 17px;
        flex: 0 0 17px;
        opacity: .85;
      }

      .app-shell .sidebar-nav a:hover {
        color: #e9f1f8;
        background: rgba(148,197,224,.05);
      }

      .app-shell .sidebar-nav a.active {
        color: #54d9ff;
        background: rgba(84,217,255,.16);
        border-left-color: #54d9ff;
      }

      .app-shell .sidebar-foot {
        margin-top: 10px;
        padding: 14px 22px 0;
        border: 0;
        border-top: 1px solid rgba(148,197,224,.14);
      }

      .app-shell .sidebar-foot a {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 14px 0 0;
        border: 0;
        background: transparent;
        color: #5b6a80;
        font: 400 .85rem 'IBM Plex Sans', sans-serif;
        text-decoration: none;
      }

      .app-shell .sidebar-foot a:hover {
        color: #ff7a7a;
      }

      .app-shell .sidebar-foot svg {
        display: block;
        width: 16px;
        height: 16px;
        flex: 0 0 16px;
      }

      /* Main content column */
      .main-col {
        position: relative;
        z-index: 1;
        display: flex;
        flex: 1 1 0;
        flex-direction: column;
        width: auto;
        min-width: 0;
        min-height: 100vh;
        margin: 0 !important;
        color: #e9f1f8;
      }

      /* Transparent glass header */
      .main-col > .topbar {
        position: sticky;
        top: 0;
        z-index: 5;
        display: flex;
        flex: 0 0 64px;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
        width: 100%;
        height: 64px;
        min-height: 64px;
        margin: 0;
        padding: 0 30px;
        background: rgba(10,15,24,.45);
        border: 0;
        border-bottom: 1px solid rgba(148,197,224,.14);
        border-radius: 0;
        box-shadow: none;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
      }

      .topbar-crumb {
        min-width: 0;
        color: #5b6a80;
        font: 400 .76rem 'IBM Plex Mono', monospace;
        letter-spacing: .02em;
      }

      .topbar-crumb b {
        color: #8fa0b6;
        font-weight: 500;
      }

      .topbar-user {
        display: flex;
        align-items: center;
        gap: 10px;
        flex: 0 0 auto;
        padding: 0;
        border: 0;
        background: transparent;
        text-decoration: none;
      }

      .topbar-user:hover {
        background: transparent;
      }

      .topbar-user .user-meta {
        display: flex;
        flex-direction: column;
        line-height: 1.25;
        text-align: right;
      }

      .topbar-user .user-meta b {
        display: block;
        max-width: 180px;
        overflow: hidden;
        color: #e9f1f8;
        font: 500 .85rem 'IBM Plex Sans', sans-serif;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .topbar-user .user-meta span {
        color: #5b6a80;
        font: 400 .68rem 'IBM Plex Mono', monospace;
        text-transform: uppercase;
        letter-spacing: .03em;
      }

      .topbar-user .user-chip {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 30px;
        height: 30px;
        flex: 0 0 30px;
        border: 1px solid rgba(148,197,224,.28);
        border-radius: 50%;
        background: rgba(84,217,255,.16);
        color: #54d9ff;
        font: 600 .85rem 'Space Grotesk', sans-serif;
      }

      /* Page content */
      .main-col > .content {
        position: relative;
        z-index: 1;
        width: 100%;
        max-width: 1180px;
        min-width: 0;
        margin: 0 auto;
        padding: 34px 34px 70px;
        color: #e9f1f8;
      }

      .page-head {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 24px;
        flex-wrap: wrap;
        margin-bottom: 30px;
      }

      .page-head h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(1.4rem, 2.4vw, 1.9rem);
        letter-spacing: -.01em;
        margin: 0 0 8px;
      }

      .page-head p {
        max-width: 56ch;
        color: #8fa0b6;
        font-size: .95rem;
      }

      /* Mobile navigation button */
      .mobile-menu {
        display: none;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        padding: 0;
        border: 1px solid rgba(148,197,224,.28);
        border-radius: 0;
        background: transparent;
        color: #e9f1f8;
        cursor: pointer;
      }

      .mobile-menu:hover {
        background: rgba(84,217,255,.12);
        border-color: #54d9ff;
      }

      @media (max-width: 900px) {
        .app-shell .sidebar {
          position: fixed !important;
          inset: 0 auto 0 0 !important;
          z-index: 1000;
          width: 248px !important;
          min-width: 248px !important;
          max-width: 248px !important;
          height: 100vh !important;
          min-height: 100vh !important;
          overflow-y: auto;
          transform: translateX(-100%);
          transition: transform .25s;
          box-shadow: 15px 0 45px rgba(0,0,0,.3) !important;
        }

        .app-shell .sidebar.open {
          transform: translateX(0);
        }

        .main-col {
          width: 100%;
          margin-left: 0 !important;
        }

        .main-col > .topbar {
          padding: 0 18px;
        }

        .main-col > .content {
          padding: 24px 18px 60px;
        }

        .mobile-menu {
          display: inline-flex;
          flex: 0 0 36px;
        }
      }

      @media (max-width: 520px) {
        .main-col > .content {
          padding: 20px 14px 44px;
        }

        .topbar-user .user-meta b {
          max-width: 100px;
        }
      }
    `;

    document.head.appendChild(style);
  }

  const navHtml = nav.map(item => {
    const isActive = item.active.includes(active);

    return `
      <li>
        <a
          href="${item.href}"
          class="${isActive ? "active" : ""}"
          ${isActive ? 'aria-current="page"' : ""}
        >
          ${icons[item.icon]}
          <span>${esc(item.label)}</span>
        </a>
      </li>`;
  }).join("");

  const currentPage = esc(
    active || (role === "citizen" ? "Dashboard" : "Dashboard")
  );

  document.body.innerHTML = `
    <div class="app-shell">
      <aside class="sidebar" id="sidebar">
        <div class="sidebar-brand">
          ${brandSvg}
          <span>LinkageX</span>
        </div>

        <div class="sidebar-role">${roleLabel}</div>

        <ul class="sidebar-nav">
          ${navHtml}
        </ul>

        <div class="sidebar-foot">
          <a href="../login.html" id="signOutLink">
            ${icons.signout}
            <span>Sign out</span>
          </a>
        </div>
      </aside>

      <div class="main-col">
        <header class="topbar">
          <div style="display:flex;align-items:center;gap:12px;min-width:0">
            <button
              class="mobile-menu"
              id="mobileMenu"
              type="button"
              aria-label="Open navigation"
            >☰</button>

            <div class="topbar-crumb">
              LinkageX / <b>${currentPage}</b>
            </div>
          </div>

          <a class="topbar-user" href="profile.html" aria-label="Open profile">
            <div class="user-meta">
              <b>${esc(userName)}</b>
              <span>${esc(role)} account</span>
            </div>

            <div class="user-chip" title="${esc(accountLabel)}">
              ${esc(initials)}
            </div>
          </a>
        </header>

        <main class="content" id="content"></main>
      </div>
    </div>

    <div id="toast-root" class="toast-root"></div>
  `;

  const mobileMenu = document.getElementById("mobileMenu");
  const sidebar = document.getElementById("sidebar");
  const signOutLink = document.getElementById("signOutLink");

  if (mobileMenu && sidebar) {
    mobileMenu.addEventListener("click", () => {
      sidebar.classList.toggle("open");
    });
  }

  if (signOutLink) {
    signOutLink.addEventListener("click", event => {
      event.preventDefault();
      logout();
    });
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
          ${title.includes("Case")
            ? "CASE INTELLIGENCE"
            : "LINKAGEX INTELLIGENCE"}
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
   INCIDENT VALUE FORMATTER
========================================================= */

function formatIncidentValue(value, key = "") {
  if (
    value === null ||
    value === undefined ||
    value === ""
  ) {
    return "Not provided";
  }

  if (key === "amounts") {
    const amounts = Array.isArray(value) ? value : [value];

    const validAmounts = amounts.filter(
      amount =>
        amount !== null &&
        amount !== undefined &&
        amount !== ""
    );

    if (validAmounts.length === 0) {
      return "Not provided";
    }

    return validAmounts.map(amount => {
      const numeric = Number(amount);

      if (Number.isNaN(numeric)) {
        return String(amount);
      }

      return "₹" + numeric.toLocaleString("en-IN");
    }).join(", ");
  }

  if (Array.isArray(value)) {
    if (value.length === 0) {
      return "Not provided";
    }

    return value
      .map(item => String(item).trim())
      .filter(Boolean)
      .join(", ");
  }

  if (typeof value === "string" && value.trim() === "") {
    return "Not provided";
  }

  return String(value);
}

/* =========================================================
   INCIDENT FIELD
========================================================= */

function field(label, value, key, editable = false) {
  const displayValue = formatIncidentValue(value, key);

  return `
    <div class="field">
      <div class="field-label">${esc(label)}</div>

      <div class="field-value" data-field="${esc(key)}">
        ${esc(displayValue)}

        ${editable ? `
          <button
            type="button"
            class="edit-link"
            onclick="editField('${esc(key)}')"
          >Edit</button>
        ` : ""}
      </div>
    </div>
  `;
}

/* =========================================================
   INCIDENT FIELD DEFINITIONS
========================================================= */

const INCIDENT_FIELD_DEFINITIONS = [
  { label: "Crime Category", key: "crime_category" },
  { label: "Crime Type", key: "crime_subcategory" },
  { label: "Incident Date", key: "incident_date" },
  { label: "Modus Operandi", key: "modus_operandi" },
  { label: "Deception Method", key: "deception" },
  { label: "Victim Action", key: "victim_action" },
  { label: "Attacker Action", key: "attacker_action" },
  { label: "Outcome", key: "outcome" },
  { label: "Channel", key: "channels" },
  { label: "Payment Method", key: "payment_method" },
  { label: "Amount", key: "amounts" },
  { label: "Organizations", key: "organizations" },
  { label: "Location", key: "locations" }
];

/* =========================================================
   RENDER STRUCTURED INCIDENT
========================================================= */

function renderIncident(incident = {}, editable = false) {
  return `
    <div class="card">
      <div class="section-head">
        <h3>Structured Incident</h3>
        <span>Standardized incident representation</span>
      </div>

      <div class="incident-grid">
        ${INCIDENT_FIELD_DEFINITIONS.map(definition =>
          field(
            definition.label,
            incident[definition.key],
            definition.key,
            editable
          )
        ).join("")}
      </div>
    </div>
  `;
}

/* =========================================================
   LOAD CURRENT ANALYSIS
========================================================= */

function getStoredAnalysis() {
  const stored = sessionStorage.getItem("linkageX_analysis");

  if (!stored) return null;

  try {
    const result = JSON.parse(stored);

    if (!result || !result.data || !result.data.incident) {
      return null;
    }

    return result;
  } catch (error) {
    console.error("Unable to parse LinkageX analysis:", error);
    return null;
  }
}

/* =========================================================
   SAVE CURRENT ANALYSIS
========================================================= */

function saveStoredAnalysis(result) {
  sessionStorage.setItem(
    "linkageX_analysis",
    JSON.stringify(result)
  );
}

/* =========================================================
   EDIT INCIDENT FIELD
========================================================= */

function editField(key) {
  const el = document.querySelector(`[data-field="${key}"]`);

  if (!el) return;
  if (el.querySelector("input")) return;

  const result = getStoredAnalysis();

  if (!result) {
    toast("Complaint information could not be found.");
    return;
  }

  const incident = result.data.incident;
  const originalValue = incident[key];

  let inputValue = "";

  if (Array.isArray(originalValue)) {
    inputValue = originalValue.join(", ");
  } else if (
    originalValue !== null &&
    originalValue !== undefined
  ) {
    inputValue = String(originalValue);
  }

  if (key === "incident_date") {
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
      >Save</button>

      <button
        type="button"
        class="edit-link"
        onclick="cancelEdit('${esc(key)}')"
      >Cancel</button>
    `;
  } else {
    if (key === "amounts") {
      const amount = Array.isArray(originalValue)
        ? originalValue[0]
        : originalValue;

      inputValue =
        amount !== null && amount !== undefined
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
      >Save</button>

      <button
        type="button"
        class="edit-link"
        onclick="cancelEdit('${esc(key)}')"
      >Cancel</button>
    `;
  }

  const input = el.querySelector("input");

  if (input) {
    input.focus();

    if (key !== "incident_date") {
      input.select();
    }
  }
}

/* =========================================================
   CANCEL EDIT
========================================================= */

function cancelEdit(key) {
  const result = getStoredAnalysis();

  if (!result) return;

  const incident = result.data.incident;
  const el = document.querySelector(`[data-field="${key}"]`);

  if (!el) return;

  el.innerHTML = `
    ${esc(formatIncidentValue(incident[key], key))}

    <button
      type="button"
      class="edit-link"
      onclick="editField('${esc(key)}')"
    >Edit</button>
  `;
}

/* =========================================================
   NORMALIZE LIST VALUE
========================================================= */

function normalizeListValue(value) {
  if (value === null || value === undefined) {
    return [];
  }

  return String(value)
    .split(",")
    .map(item => item.trim())
    .filter(Boolean);
}

/* =========================================================
   SAVE EDITED INCIDENT FIELD
========================================================= */

function saveEdited(key) {
  const el = document.querySelector(`[data-field="${key}"]`);

  if (!el) return;

  const input = el.querySelector("input");

  if (!input) return;

  const result = getStoredAnalysis();

  if (!result) {
    toast("Complaint information could not be found.");
    return;
  }

  const incident = result.data.incident;
  const rawValue = input.value.trim();

  if (key === "incident_date") {
    incident.incident_date = rawValue;
  } else if (key === "amounts") {
    const cleaned = rawValue
      .replace(/₹/g, "")
      .replace(/,/g, "")
      .trim();

    if (cleaned === "") {
      incident.amounts = [];
    } else {
      const numericValue = Number(cleaned);

      if (Number.isNaN(numericValue) || numericValue < 0) {
        toast("Please enter a valid amount.");
        return;
      }

      incident.amounts = [numericValue];
    }
  } else if (
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
    incident[key] = normalizeListValue(rawValue);
  } else {
    incident[key] = rawValue;
  }

  let missingFields = Array.isArray(result.data.missing_fields)
    ? [...result.data.missing_fields]
    : [];

  const currentValue = incident[key];

  const hasValue = Array.isArray(currentValue)
    ? currentValue.length > 0
    : String(currentValue ?? "").trim() !== "";

  if (hasValue) {
    missingFields = missingFields.filter(
      fieldName => fieldName !== key
    );
  } else if (!missingFields.includes(key)) {
    missingFields.push(key);
  }

  result.data.incident = incident;
  result.data.missing_fields = missingFields;

  saveStoredAnalysis(result);

  const displayValue = formatIncidentValue(incident[key], key);

  el.innerHTML = `
    ${esc(displayValue)}

    <button
      type="button"
      class="edit-link"
      onclick="editField('${esc(key)}')"
    >Edit</button>
  `;

  toast("Field updated successfully.");
}

/* =========================================================
   BUILD FINALIZED CASE
========================================================= */

function buildFinalizedCase(victimType = "me") {
  const result = getStoredAnalysis();

  if (!result) return null;

  const data = result.data || {};
  const incident = { ...(data.incident || {}) };

  return {
    complaint_text: data.complaint_text || "",
    incident: incident,
    missing_fields: [],
    victim_type: victimType,
    finalized_at: new Date().toISOString()
  };
}

/* =========================================================
   SAVE FINALIZED CASE
========================================================= */

function saveFinalizedCase(victimType = "me") {
  const finalizedCase = buildFinalizedCase(victimType);

  if (!finalizedCase) {
    toast("Unable to finalize complaint.");
    return false;
  }

  sessionStorage.setItem(
    "linkageX_final_case",
    JSON.stringify(finalizedCase)
  );

  /* Keep the analysis synchronized with the finalized incident. */
  const result = getStoredAnalysis();

  if (result) {
    result.data.missing_fields = [];
    result.data.incident = finalizedCase.incident;
    saveStoredAnalysis(result);
  }

  return true;
}

/* =========================================================
   LOGOUT
========================================================= */

function logout() {
  sessionStorage.clear();
  window.location.href = "../login.html";
}