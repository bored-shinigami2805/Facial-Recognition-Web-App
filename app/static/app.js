// FaceMatch front-end. Plain vanilla JS + fetch(); no build step.
// Talks to the FastAPI backend: /api/enroll, /api/recognize, /api/people, /api/config.

const $ = (id) => document.getElementById(id);

// escape server-provided strings (names) before putting them in innerHTML
const escapeHtml = (s) =>
  String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// small session state (resets on reload) used for the stat cards + history
const session = { scans: 0, matched: 0, unknown: 0 };
let peopleById = {};   // id -> {name, thumbnail}
let webcamStream = null;

// ---------------------------------------------------------------- helpers
function toast(msg, kind = "info") {
  const el = document.createElement("div");
  el.className = "toast " + kind;
  el.textContent = msg;
  $("toast-host").appendChild(el);
  setTimeout(() => {
    el.style.opacity = "0";
    el.style.transition = "opacity .3s";
    setTimeout(() => el.remove(), 300);
  }, 3200);
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch (_) { /* non-JSON (e.g. 204) */ }
  if (!res.ok) throw new Error((data && data.detail) || `Request failed (${res.status})`);
  return data;
}

const nowHM = () =>
  new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

// ---------------------------------------------------------------- clock
function tickClock() {
  $("clock").textContent = new Date().toLocaleTimeString([], { hour12: false });
}
setInterval(tickClock, 1000);
tickClock();

// ---------------------------------------------------------------- nav
document.querySelectorAll(".nav-link").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-link").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    btn.classList.add("active");
    $(btn.dataset.view).classList.add("active");
    if (btn.dataset.view === "gallery") loadGallery();
  });
});

// ---------------------------------------------------------------- config / stats
async function loadConfig() {
  const cfg = await api("/api/config");
  $("thr-slider").value = cfg.threshold;
  $("thr-val").textContent = cfg.threshold.toFixed(2);
}
$("thr-slider").addEventListener("input", (e) => {
  $("thr-val").textContent = parseFloat(e.target.value).toFixed(2);
});

function refreshStats() {
  $("stat-scans").textContent = session.scans;
  $("stat-matched").textContent = session.matched;
  $("stat-unknown").textContent = session.unknown;
}

async function loadPeople() {
  const people = await api("/api/people");
  peopleById = {};
  let images = 0;
  people.forEach((p) => {
    peopleById[p.id] = { name: p.name, thumbnail: p.thumbnail };
    images += p.image_count;
  });
  $("stat-people").textContent = people.length;
  return people;
}

// ================================================================ RECOGNIZE
const ring = $("scan-ring");

function setRingMode(mode) {
  // mode: "" (placeholder) | "camera" | "result"
  ring.classList.remove("camera", "result");
  if (mode) ring.classList.add(mode);
}

// -- webcam --
$("primary-btn").addEventListener("click", async () => {
  if (!webcamStream) {
    // start the camera
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
      $("webcam").srcObject = webcamStream;
      setRingMode("camera");
      $("scan-hint").textContent = "Camera on — line up a face and capture.";
      $("primary-btn").textContent = "Capture & recognize";
      $("reset-btn").classList.remove("hidden");
    } catch (e) {
      toast("Could not access camera: " + e.message, "err");
    }
  } else {
    // capture current frame -> recognize
    const video = $("webcam");
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob((blob) => recognizeBlob(blob), "image/png");
  }
});

function stopCamera() {
  if (webcamStream) {
    webcamStream.getTracks().forEach((t) => t.stop());
    webcamStream = null;
  }
}

// -- file upload --
$("upload-btn").addEventListener("click", () => $("rec-file").click());
$("rec-file").addEventListener("change", (e) => {
  if (e.target.files[0]) recognizeBlob(e.target.files[0]);
});

// -- demo chips --
document.querySelectorAll(".chip[data-demo]").forEach((chip) => {
  chip.addEventListener("click", async () => {
    try {
      const blob = await (await fetch(chip.dataset.demo)).blob();
      recognizeBlob(blob);
    } catch (e) { toast("Could not load sample image.", "err"); }
  });
});

// -- reset --
$("reset-btn").addEventListener("click", () => {
  stopCamera();
  setRingMode("");
  $("primary-btn").textContent = "Start camera";
  $("reset-btn").classList.add("hidden");
  $("scan-hint").textContent = "No image loaded";
  $("result-empty").classList.remove("hidden");
  $("result-body").classList.add("hidden");
});

// -- the actual recognize call --
async function recognizeBlob(blob) {
  stopCamera();
  setRingMode("");
  ring.classList.add("busy");
  $("scan-hint").textContent = "Scanning…";
  $("primary-btn").disabled = true;

  const fd = new FormData();
  fd.append("file", blob, "scan.png");
  fd.append("threshold", $("thr-slider").value);

  try {
    const data = await api("/api/recognize", { method: "POST", body: fd });
    await drawScan(blob, data.matches);
    setRingMode("result");
    $("scan-hint").textContent =
      `${data.faces_found} face(s) found · threshold ${data.threshold.toFixed(2)}`;
    renderResult(data);
    updateHistory(data);
    $("reset-btn").classList.remove("hidden");
    $("primary-btn").textContent = "Start camera";
  } catch (e) {
    toast(e.message, "err");
    $("scan-hint").textContent = "Scan failed";
  } finally {
    ring.classList.remove("busy");
    $("primary-btn").disabled = false;
  }
}

// draw the scanned image plus detection boxes onto the result canvas
function drawScan(blob, matches) {
  return new Promise((resolve) => {
    const canvas = $("scan-canvas");
    const img = new Image();
    img.onload = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);

      const scale = img.naturalWidth / 400;
      const th = Math.max(14, Math.round(16 * scale));
      const pad = Math.max(3, 4 * scale);
      ctx.lineWidth = Math.max(2, 3 * scale);
      ctx.font = `${th}px sans-serif`;
      ctx.textBaseline = "top";

      matches.forEach((m) => {
        const color = m.name === "Unknown" ? "#ef4444" : "#22c55e";
        const [x1, y1, x2, y2] = m.box;
        ctx.strokeStyle = color;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        const label = m.name + (m.distance != null ? `  (${m.distance.toFixed(2)})` : "");
        const tw = ctx.measureText(label).width;
        const ly = Math.max(0, y1 - th - pad * 2);
        ctx.fillStyle = color;
        ctx.fillRect(x1, ly, tw + pad * 2, th + pad * 2);
        ctx.fillStyle = "#fff";
        ctx.fillText(label, x1 + pad, ly + pad);
      });

      URL.revokeObjectURL(img.src);
      resolve();
    };
    img.onerror = () => resolve();
    img.src = URL.createObjectURL(blob);
  });
}

function avatarHtml(match, cls) {
  const p = match.person_id != null ? peopleById[match.person_id] : null;
  if (p && p.thumbnail) return `<img class="identity-avatar ${cls}" src="${escapeHtml(p.thumbnail)}" alt="">`;
  return `<div class="identity-avatar ${cls} placeholder">
      <svg viewBox="0 0 24 24" width="30" height="30" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="8.5" r="4"/><path d="M4 20a8 8 0 0 1 16 0"/></svg></div>`;
}

function renderResult(data) {
  const empty = $("result-empty");
  const body = $("result-body");
  empty.classList.add("hidden");
  body.classList.remove("hidden");

  if (data.faces_found === 0) {
    body.innerHTML = `<div class="result-empty" style="margin:auto">
        <div class="result-empty-title">No face detected</div>
        <div class="muted">Try a clearer, front-facing photo.</div></div>`;
    return;
  }

  // sort so a matched face is shown as the headline
  const matches = [...data.matches];
  const headline = matches.find((m) => m.name !== "Unknown") || matches[0];
  const known = headline.name !== "Unknown";
  const conf = headline.confidence != null ? Math.round(headline.confidence * 100) : null;

  let html = `<div class="identity">
      ${avatarHtml(headline, known ? "" : "placeholder")}
      <div>
        <div class="identity-name">${escapeHtml(headline.name)}</div>
        <div class="identity-sub">${data.faces_found} face(s) in this photo</div>
        <span class="pill ${known ? "pill-ok" : "pill-warn"}">
          ${known ? "● Recognized" : "● Not recognized"}</span>
      </div>
    </div>`;

  html += `<div class="metrics">
      <div class="metric"><div class="metric-label">Distance</div>
        <div class="metric-val">${headline.distance == null ? "–" : headline.distance.toFixed(3)}</div></div>
      <div class="metric"><div class="metric-label">Confidence</div>
        <div class="metric-val">${conf == null ? "–" : conf + "%"}</div>
        <div class="confbar"><span style="width:${conf || 0}%"></span></div></div>
    </div>`;

  // extra faces (if any) listed compactly below
  const rest = matches.filter((m) => m !== headline);
  if (rest.length) {
    html += `<div class="result-multi" style="margin-top:14px">
      <div class="multi-head">Other faces</div>` +
      rest.map((m) => `<div class="multi-row">
          <span class="dot ${m.name === "Unknown" ? "" : "dot-green"}" style="${m.name === "Unknown" ? "background:#f59e0b" : ""}"></span>
          <span class="name">${escapeHtml(m.name)}</span>
          <span class="dist">${m.distance == null ? "–" : m.distance.toFixed(3)}</span>
        </div>`).join("") + `</div>`;
  }

  $("result-body").innerHTML = html;
}

function updateHistory(data) {
  session.scans += 1;
  const list = $("history-list");
  const emptyEl = list.querySelector(".history-empty");
  if (emptyEl) emptyEl.remove();

  data.matches.forEach((m) => {
    const known = m.name !== "Unknown";
    if (known) session.matched += 1; else session.unknown += 1;

    const item = document.createElement("div");
    item.className = "history-item";
    item.innerHTML = `
      <span class="dot" style="background:${known ? "var(--green-dot)" : "var(--amber)"}"></span>
      <span class="name">${escapeHtml(m.name)}</span>
      <span class="time">${nowHM()}</span>`;
    list.prepend(item);
  });
  // keep the list short
  [...list.querySelectorAll(".history-item")].slice(8).forEach((n) => n.remove());
  refreshStats();
}

// ================================================================ ENROLL
const dz = $("dropzone");
const enrFiles = $("enr-files");

dz.addEventListener("click", () => enrFiles.click());
["dragover", "dragenter"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => { enrFiles.files = e.dataTransfer.files; renderPreviews(); });
enrFiles.addEventListener("change", renderPreviews);

function renderPreviews() {
  const grid = $("preview-grid");
  grid.innerHTML = "";
  [...enrFiles.files].forEach((f) => {
    const img = document.createElement("img");
    img.src = URL.createObjectURL(f);
    grid.appendChild(img);
  });
}

$("enr-submit").addEventListener("click", async () => {
  const name = $("enr-name").value.trim();
  if (!name) return toast("Please enter a name.", "err");
  if (!enrFiles.files.length) return toast("Please add at least one photo.", "err");

  const fd = new FormData();
  fd.append("name", name);
  [...enrFiles.files].forEach((f) => fd.append("files", f));

  const btn = $("enr-submit");
  btn.disabled = true; btn.textContent = "Enrolling…";
  try {
    const data = await api("/api/enroll", { method: "POST", body: fd });
    toast(data.message, "ok");
    $("enr-name").value = "";
    enrFiles.value = "";
    $("preview-grid").innerHTML = "";
    await loadPeople();      // refresh the "people enrolled" stat
    refreshStats();
  } catch (e) {
    toast(e.message, "err");
  } finally {
    btn.disabled = false; btn.textContent = "Enroll person";
  }
});

// ================================================================ GALLERY
async function loadGallery() {
  const people = await loadPeople();
  const list = $("gallery-list");
  if (!people.length) {
    list.innerHTML = `<div class="gallery-empty">Nobody enrolled yet.<br>Head to the Enroll tab to add someone.</div>`;
    return;
  }
  list.innerHTML = people.map((p) => `
    <div class="person-card">
      <img class="person-avatar" src="${escapeHtml(p.thumbnail || "")}" alt="${escapeHtml(p.name)}">
      <div class="person-name">${escapeHtml(p.name)}</div>
      <div class="person-count">${p.image_count} photo(s)</div>
      <button class="btn-del" data-id="${p.id}">Delete</button>
    </div>`).join("");

  list.querySelectorAll(".btn-del").forEach((b) =>
    b.addEventListener("click", () => deletePerson(b.dataset.id)));
}

async function deletePerson(id) {
  if (!confirm("Delete this person and their photos?")) return;
  try {
    await api(`/api/people/${id}`, { method: "DELETE" });
    toast("Person deleted.", "ok");
    await loadGallery();
  } catch (e) { toast(e.message, "err"); }
}

// ================================================================ init
(async function init() {
  try {
    await loadConfig();
    await loadPeople();
    refreshStats();
  } catch (e) {
    toast("Could not reach the server. Is uvicorn running?", "err");
  }
})();
