const CS = {

  incident: {

    caseId:"CC001", crimeCategory:"Financial Fraud", crimeType:"Loan Scam", channel:"WhatsApp",

    attackerEntity:"Unknown", deceptionMethod:"Fake loan offer", victimAction:"Paid processing fee",

    amount:15000, outcome:"Money lost", incidentDate:"2026-08-28", location:"Mumbai",

    indicators:["WhatsApp","Loan offer","Processing fee","UPI","Payment requested"],

    confidence:{crimeCategory:.94,crimeType:.91,channel:.97,deceptionMethod:.89,victimAction:.94,amount:.99}

  },

  missingFields:["incidentDate","contactMethod"],

  related:[

    {caseId:"CC034",similarity:.91,signals:{channel:1,crimeType:1,mo:.90,victimAction:.90,temporalProximity:.80,entity:.50},commonSignals:["WhatsApp","Loan offer","Processing fee","Payment requested","Similar time period"],differences:["Amount differs","Entity not identified in both cases","Location differs"]},

    {caseId:"CC078",similarity:.86,signals:{channel:.98,crimeType:1,mo:.84,victimAction:.80,temporalProximity:.76,entity:.45},commonSignals:["WhatsApp","Loan Scam","Payment"],differences:["Different amount","Different region"]},

    {caseId:"CC102",similarity:.81,signals:{channel:.72,crimeType:1,mo:.90,victimAction:.75,temporalProximity:.65,entity:.35},commonSignals:["Loan Scam","Similar MO"],differences:["Different channel","Entity unknown"]},

    {caseId:"CC143",similarity:.76,signals:{channel:.95,crimeType:.70,mo:.78,victimAction:.60,temporalProximity:.58,entity:.20},commonSignals:["WhatsApp","Financial Fraud"],differences:["Different crime subtype","Different amount"]},

    {caseId:"CC211",similarity:.72,signals:{channel:.80,crimeType:.65,mo:.76,victimAction:.55,temporalProximity:.50,entity:.30},commonSignals:["Similar deception pattern"],differences:["Different channel","Different date"]}

  ],

  cases:[

    ["CC001","Loan Scam","28 Aug 2026","New","5","Mumbai","WhatsApp"],

    ["CC002","UPI Fraud","27 Aug 2026","Potentially Linked","3","Pune","UPI"],

    ["CC003","Loan Scam","26 Aug 2026","New","4","Mumbai","WhatsApp"],

    ["CC004","Phishing","25 Aug 2026","Under Review","2","Delhi","Email"],

    ["CC005","Fake Job Offer","24 Aug 2026","Unlinked","0","Navi Mumbai","WhatsApp"],

    ["CC006","Investment Scam","23 Aug 2026","Flagged","6","Bengaluru","Telegram"],

    ["CC007","Online Shopping Fraud","22 Aug 2026","New","1","Mumbai","Website"],

    ["CC008","Account Takeover","21 Aug 2026","Potentially Linked","4","Pune","Instagram"],

    ["CC009","UPI Fraud","20 Aug 2026","Unlinked","0","Thane","UPI"],

    ["CC010","Impersonation Scam","19 Aug 2026","Under Review","2","Mumbai","Phone call"],

    ["CC011","Loan Scam","18 Aug 2026","New","5","Pune","WhatsApp"],

    ["CC012","Phishing","17 Aug 2026","Saved for Future Correlation","0","Delhi","SMS"],

    ["CC013","Fake Job Offer","16 Aug 2026","New","2","Mumbai","WhatsApp"],

    ["CC014","Investment Scam","15 Aug 2026","Potentially Linked","3","Hyderabad","Telegram"],

    ["CC015","Online Shopping Fraud","14 Aug 2026","Unlinked","0","Pune","Website"],

    ["CC016","UPI Fraud","13 Aug 2026","New","3","Mumbai","UPI"],

    ["CC017","Loan Scam","12 Aug 2026","Flagged","4","Thane","WhatsApp"],

    ["CC018","Phishing","11 Aug 2026","New","1","Mumbai","Email"],

    ["CC019","Account Takeover","10 Aug 2026","Under Review","2","Pune","Instagram"],

    ["CC020","Impersonation Scam","09 Aug 2026","Unlinked","0","Navi Mumbai","Phone call"]

  ],

  complaints:[

    ["CC001","Loan Scam","28 Aug 2026","Submitted"],["CC002","UPI Fraud","24 Aug 2026","Submitted"],["CC003","Online Shopping Fraud","18 Aug 2026","Reviewed"]

  ],

  trends:{region:["UPI / Payment Fraud","Online Shopping Fraud","Loan Scams","Account Takeover"],age:["Online payment fraud","Fake job offers","Social media scams"]},

  clusters:[

    {name:"Cluster A — Loan Scam",count:12,ids:["C01","C34","C78","C102","C143"],channels:["WhatsApp","Phone"],regions:["Mumbai","Pune"],range:"12–28 Aug 2026"},

    {name:"Cluster B — UPI Fraud",count:9,ids:["C08","C45","C91"],channels:["UPI","SMS"],regions:["Mumbai","Thane"],range:"13–27 Aug 2026"},

    {name:"Cluster C — Social Media Account Takeover",count:7,ids:["C11","C28","C54"],channels:["Instagram","WhatsApp"],regions:["Pune","Mumbai"],range:"10–25 Aug 2026"}

  ]

};

function fetchCases(){return CS.cases}

function fetchIncident(){return CS.incident}

function getRelatedCases(){return CS.related}

function getSimilarityBreakdown(id){return CS.related.find(x=>x.caseId===id)||CS.related[0]}

function getMissingFields(){return CS.missingFields}

function submitComplaint(text){return {ok:true,incident:CS.incident,received:text}}

function saveCase(id){localStorage.setItem("saved:"+id,"1");return true}