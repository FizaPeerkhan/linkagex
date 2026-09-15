function toast(message){
  const root=document.getElementById("toast-root")||document.body;
  const el=document.createElement("div");el.className="toast";el.textContent=message;
  root.appendChild(el);setTimeout(()=>el.remove(),2800);
}

function statusClass(s){
  const x=s.toLowerCase();if(x.includes("new"))return"status-new";if(x.includes("reviewed"))return"status-reviewed";if(x.includes("linked"))return"status-linked";if(x.includes("unlinked"))return"status-unlinked";if(x.includes("flagged"))return"status-flagged";return"status-saved"
}

function esc(s){
  return String(s??"").replace(/[&<>"']/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[m]))
}

function initShell(role,active){
  const nav=role==="citizen"?[
    ["Overview","dashboard.html","⌂"],["Report","complaint.html","＋"],["My Complaints","complaints.html","▤"],["Awareness","awareness.html","◌"],["Safety Center","guidance.html","✓"],
    ["Account","profile.html","○"]
  ]:[
    ["Overview","dashboard.html","⌂"],["Cases","cases.html","▤"],["New Cases","cases.html?filter=new","＋"],["Unlinked Cases","cases.html?filter=unlinked","○"],["Flagged Cases","cases.html?filter=flagged","⚑"],
    ["Potential Linkages","linkage.html","⌁"],["Incident Clusters","clusters.html","◉"],["Upload Report","upload-report.html","↑"],["Profile","profile.html","○"],["Settings","#","⚙"]
  ];

  document.body.insertAdjacentHTML("afterbegin",`<aside class="sidebar" id="sidebar"><div class="brand"><span class="brand-mark"><span></span></span><div><strong>CYBERSHIELD</strong><small>INTELLIGENCE</small></div></div>${nav.map((n,i)=>`${i===0||(["Report","Cases","Potential Linkages","Upload Report","Profile"].includes(n[0]))?`<div class="nav-group">`:""}${i===0||(["Report","Cases","Potential Linkages","Upload Report","Profile"].includes(n[0]))?`<div class="nav-label">${i===0?"Overview":n[0]==="Profile"?"Account":n[0]==="Cases"?"Cases":n[0]==="Potential Linkages"?"Intelligence":"Intake"}</div>`:""}<a class="nav-item ${active===n[0]?"active":""}" href="${n[1]}"><span class="nav-icon">${n[2]}</span>${n[0]}</a>${i===nav.length-1?"</div>":""}`).join("")}<div class="sidebar-footer">Prototype interface<br><span>Frontend-only • Mock intelligence data</span></div></aside>`);

  document.body.insertAdjacentHTML("afterbegin",`<div class="main"><header class="topbar"><div><button class="mobile-menu" id="mobileMenu">☰</button> <span class="topbar-title">${role==="citizen"?"Citizen Portal":"Investigator Intelligence"}</span><div class="topbar-sub">${role==="citizen"?"Secure reporting & awareness":"Structured incident analysis & linkage support"}</div></div><div class="user-chip"><span>${role==="citizen"?"Citizen Account":"Investigator • Demo"}</span><span class="avatar">${role==="citizen"?"CZ":"IN"}</span></div></header><div class="content" id="content"></div></div>`);

  document.getElementById("mobileMenu").onclick=()=>document.getElementById("sidebar").classList.toggle("open");
}

function pageHead(title,sub,action=""){
  return `<div class="page-head"><div><div class="eyebrow-dark">${title.includes("Case")?"CASE INTELLIGENCE":"CYBERSHIELD INTELLIGENCE"}</div><h1>${title}</h1><p>${sub}</p></div>${action}</div>`
}

function shell(role,active){
  document.body.innerHTML="";
  initShell(role,active);
  return document.getElementById("content")
}

function badge(s){
  return `<span class="status ${statusClass(s)}">${esc(s)}</span>`
}

function field(label,value,key,editable=false){
  return `<div class="field"><div class="field-label">${label}</div><div class="field-value" data-field="${key}">${esc(value)} ${editable?`<button class="edit-link" onclick="editField('${key}')">Edit</button>`:""}</div></div>`
}

function renderIncident(incident=CS.incident,editable=false){
  return `<div class="card"><div class="section-head"><h3>Structured Incident</h3><span>Standardized incident representation</span></div><div class="incident-grid">
  ${field("Crime Category",incident.crimeCategory,"crimeCategory",editable)}${field("Crime Type",incident.crimeType,"crimeType",editable)}
  ${field("Channel",incident.channel,"channel",editable)}${field("Attacker / Entity",incident.attackerEntity,"attackerEntity",editable)}
  ${field("Deception Method",incident.deceptionMethod,"deceptionMethod",editable)}${field("Victim Action",incident.victimAction,"victimAction",editable)}
  ${field("Amount Lost",incident.amount?"₹"+incident.amount.toLocaleString("en-IN"):"Not provided","amount",editable)}${field("Outcome",incident.outcome,"outcome",editable)}
  ${field("Incident Date",incident.incidentDate||"Not provided","incidentDate",editable)}${field("Location",incident.location||"Not provided","location",editable)}
  </div></div>`
}

function editField(key){
  const el=document.querySelector(`[data-field="${key}"]`);if(!el)return;
  const current=el.textContent.replace(" Edit","").trim();
  el.innerHTML=`<input value="${esc(current.replace(/^₹/,"").replaceAll(",",""))}"><button class="edit-link" onclick="saveEdited('${key}')">Save</button>`;
}

function saveEdited(key){
  const el=document.querySelector(`[data-field="${key}"]`),input=el.querySelector("input");let val=input.value;if(key==="amount")val="₹"+Number(val||0).toLocaleString("en-IN");el.textContent=val+" ";el.insertAdjacentHTML("beforeend",`<button class="edit-link" onclick="editField('${key}')">Edit</button>`);toast("Field updated for review.");
}

function logout(){
  sessionStorage.clear();
  location.href="../index.html"
}