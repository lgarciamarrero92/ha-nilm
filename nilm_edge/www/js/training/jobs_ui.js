// js/training/jobs_ui.js
// Mobile-friendly paginated jobs UI + polling for /training endpoint.
//
// Columns (4): Appliance | Range | Status | Remove
//
// Fixes:
//  - No "DONE" on status:"success" (only on explicit done/completed flags)
//  - If server provides no progress, UI derives phase from state + shows elapsed time
//  - GET never sends body
//  - Renders a simplified 4-column table with pagination
//  - Poll failures do NOT flip to ERROR
//  - Handles backend "Missing/invalid action..." payloads cleanly

const STORAGE_KEY = "nilm_training_jobs_v5";
const MAX_JOBS = 200;
const PAGE_SIZE = 5;

const STATUS = {
  PREPARED: "prepared",
  SENT: "sent",
  QUEUED: "queued",
  RUNNING: "running",
  DONE: "done",
  ERROR: "error",
  STALE: "stale",
};

const POLLABLE = new Set([STATUS.SENT, STATUS.QUEUED, STATUS.RUNNING]);
const ACTIVE_STATUSES = new Set([STATUS.PREPARED, STATUS.SENT, STATUS.QUEUED, STATUS.RUNNING]);
const TERMINAL_STATUSES = new Set([STATUS.DONE, STATUS.ERROR, STATUS.STALE]);

function nowIso() {
  return new Date().toISOString();
}

function escapeHtml(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function safeStr(v, fallback = "-") {
  return typeof v === "string" && v.trim() ? v.trim() : fallback;
}

function fmtRange(startMs, endMs) {
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) return "-";
  try {
    const fmt = new Intl.DateTimeFormat("en-GB", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
    const a = fmt.format(new Date(startMs));
    const b = fmt.format(new Date(endMs));
    return `${a}\n${b}`;
  } catch {
    return "-";
  }
}

function computeSelectedRange(selectedWindows) {
  if (!Array.isArray(selectedWindows) || selectedWindows.length === 0) {
    return { startMs: NaN, endMs: NaN, n: 0 };
  }
  let min = Infinity;
  let max = -Infinity;
  for (const w of selectedWindows) {
    const a = Number(w?.start);
    const b = Number(w?.end);
    if (Number.isFinite(a)) min = Math.min(min, a);
    if (Number.isFinite(b)) max = Math.max(max, b);
  }
  return {
    startMs: Number.isFinite(min) ? min : NaN,
    endMs: Number.isFinite(max) ? max : NaN,
    n: selectedWindows.length,
  };
}

function shortId(id, left = 8, right = 4) {
  const s = String(id || "");
  if (!s) return "-";
  if (s.length <= left + right + 3) return s;
  return `${s.slice(0, left)}…${s.slice(-right)}`;
}

// IMPORTANT: do NOT treat generic "success" as DONE.
function normalizeStatus(raw) {
  const s = String(raw || "").toLowerCase().trim();
  if (!s) return null;

  if (s.includes("error") || s.includes("fail") || s.includes("oom")) return STATUS.ERROR;

  // recognize training-server completion states as DONE
  if (s === "done" || s === "training_server_done" || s.endsWith("_done") || s.includes("finished") || s.includes("completed") || s.includes("complete")) {
    return STATUS.DONE;
  }

  if (s.includes("accepted")) return STATUS.QUEUED;
  if (s.includes("queue") || s.includes("pending") || s.includes("enqueued")) return STATUS.QUEUED;
  if (s.includes("run") || s.includes("train") || s.includes("process")) return STATUS.RUNNING;
  if (s.includes("sent") || s.includes("upload")) return STATUS.SENT;
  if (s.includes("prep") || s.includes("ready")) return STATUS.PREPARED;

  return s;
}


function badgeHtml(status) {
  const s = String(status || "").toLowerCase();
  const cls =
    s === STATUS.DONE ? "badge-green" :
    s === STATUS.RUNNING ? "badge-blue" :
    s === STATUS.QUEUED || s === STATUS.SENT ? "badge-yellow" :
    s === STATUS.ERROR ? "badge-red" :
    s === STATUS.PREPARED ? "badge-gray" :
    s === STATUS.STALE ? "badge-red" :
    "badge-gray";
  return `<span class="badge ${cls}">${escapeHtml(s || "unknown")}</span>`;
}

function methodAllowsBody(method) {
  const m = String(method || "GET").toUpperCase();
  return !(m === "GET" || m === "HEAD");
}

async function safeFetchJson(url, { method = "POST", body = null, timeoutMs = 12000 } = {}) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);

  const m = String(method || "GET").toUpperCase();
  const init = { method: m, signal: ctrl.signal };

  if (methodAllowsBody(m)) {
    init.headers = { "Content-Type": "application/json" };
    init.body = body == null ? "{}" : body;
  }

  try {
    const resp = await fetch(url, init);
    const text = await resp.text().catch(() => "");
    let json = null;
    try { json = text ? JSON.parse(text) : null; } catch { json = null; }
    return { ok: resp.ok, status: resp.status, text, json };
  } catch (e) {
    return { ok: false, status: 0, text: String(e?.message || e), json: null };
  } finally {
    clearTimeout(t);
  }
}

// Show "phase + elapsed" even if server sends no epoch/loss progress.
function formatElapsedMs(ms) {
  if (!Number.isFinite(ms) || ms < 0) return "";
  const sec = Math.floor(ms / 1000);
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  if (m <= 0) return `${s}s`;
  return `${m}m ${s}s`;
}

function isDoneStatus(s) {
  return String(s || "").toLowerCase() === STATUS.DONE;
}

function computeDisplayJobs(allJobs) {
  const sorted = stableSortByCreatedDesc(allJobs);
  return sorted.filter((job) => {
    const normalized = normalizeStatus(job?.status);
    if (ACTIVE_STATUSES.has(normalized)) return true;
    return job?.has_been_visible === true && TERMINAL_STATUSES.has(normalized);
  });
}

function sanitizeForPersistence(allJobs) {
  return stableSortByCreatedDesc(allJobs)
    .filter((job) => ACTIVE_STATUSES.has(normalizeStatus(job?.status)))
    .map((job) => {
      const { has_been_visible, ...persisted } = job || {};
      return persisted;
    })
    .slice(0, MAX_JOBS);
}

function progressLine(job) {
    if (job.status === STATUS.PREPARED) return "";
    const p = job.progress && typeof job.progress === "object" ? job.progress : null;

    if (isDoneStatus(job.status)) {
        return "";
    }

    const parts = [];
    const fineTuneTarget = String(p?.fine_tune_target || job?.training_metrics?.fine_tune_target || "").toLowerCase();
    const stageRaw = String(p?.stage || "").toLowerCase();
    if (fineTuneTarget === "weak_onoff" && stageRaw) {
      const stageLabel = stageRaw.includes("stage1")
        ? "stage 1/2"
        : stageRaw.includes("stage2")
          ? "stage 2/2"
          : stageRaw;
      parts.push(stageLabel);
    }

    // While running/queued: show epoch if available
    const ep = Number(p?.epoch);
    const tot = Number(p?.total_epochs ?? p?.totalEpochs);
    if (Number.isFinite(ep) && Number.isFinite(tot) && tot > 0) {
        parts.push(`epoch ${ep}/${tot}`);
    }

    // local elapsed (only for running/queued)
    if (job.status === STATUS.RUNNING || job.status === STATUS.QUEUED) {
        const startedAt = Date.parse(job.started_at || "") || 0;
        if (startedAt) parts.push(formatElapsedMs(Date.now() - startedAt));
    }

    // Use a nicer separator
    return parts.length ? parts.join(" • ") : "";
}

function formatMetricValue(v, digits = 3) {
  if (v == null || v === "") return "";
  const n = Number(v);
  if (!Number.isFinite(n)) return "";
  return n.toFixed(digits);
}

function trainingMetricsLine(job) {
  return "";

  const m = job.training_metrics && typeof job.training_metrics === "object" ? job.training_metrics : null;
  if (!m || !isDoneStatus(job.status)) return "";

  const preferredOnOffF1Text = formatMetricValue(m.onoff_f1);
  const preferredOnOffThrText = formatMetricValue(m.onoff_threshold, 2);
  if (preferredOnOffF1Text && preferredOnOffThrText) {
    return `Best F1 ${preferredOnOffF1Text} - Thr ${preferredOnOffThrText}`;
  }

  const preferredOnoffF1 = formatMetricValue(m.onoff_f1);
  const preferredOnoffThr = formatMetricValue(m.onoff_threshold, 2);
  if (preferredOnoffF1 && preferredOnoffThr) {
    return `Best F1 ${preferredOnoffF1} â€¢ Thr ${preferredOnoffThr}`;
  }

  const powerF1 = formatMetricValue(m.power_f1);
  const powerThr = formatMetricValue(m.power_threshold, 1);
  if (powerF1 && powerThr) {
    return `Best F1 ${powerF1} • Thr ${powerThr} W`;
  }

  const onoffF1 = formatMetricValue(m.onoff_f1);
  const onoffThr = formatMetricValue(m.onoff_threshold, 2);
  if (onoffF1 && onoffThr) {
    return `Best F1 ${onoffF1} • Thr ${onoffThr}`;
  }

  return "";
}

function parseStatusPayload(payload) {
  if (!payload || typeof payload !== "object") return { status: null };

  const topStatus = String(payload.status || "").toLowerCase().trim();

  if (topStatus === "error") {
    return {
      status: STATUS.ERROR,
      error: payload.message || payload.detail || payload.error || "Unknown error",
      message: payload.message || payload.detail || "",
      training_server_job_id: payload.training_server_job_id || payload.trainingServerJobId || payload.training_server_id || null,
      saved_path: null,
      has_result: false,
      progress: null,
    };
  }

  const msgMaybe = String(payload.message || payload.detail || "");
  if (msgMaybe.includes("Missing/invalid action")) {
    return {
      status: STATUS.ERROR,
      error: msgMaybe,
      message: msgMaybe,
      training_server_job_id: payload.training_server_job_id || payload.trainingServerJobId || payload.training_server_id || null,
      saved_path: null,
      has_result: false,
      progress: null,
    };
  }

  const job = (payload.job && typeof payload.job === "object") ? payload.job : payload;

  const stateRaw =
    job.training_server_status ?? job.trainingServerStatus ?? job.state ?? job.job_state ??
    payload.training_server_status ?? payload.trainingServerStatus ?? payload.state ?? payload.job_state ??
    job.status ?? null;

  const norm = normalizeStatus(stateRaw);

  const trainingServerJobId =
    job.training_server_job_id || job.trainingServerJobId || job.training_server_id ||
    payload.training_server_job_id || payload.trainingServerJobId || payload.training_server_id || null;

  const savedPath =
    (typeof job.saved_path === "string" && job.saved_path.trim()) ||
    (typeof payload.saved_path === "string" && payload.saved_path.trim()) ||
    null;

  const embeddingDim =
    (Number.isFinite(Number(job.embedding_dim)) ? Number(job.embedding_dim) : null) ??
    (Number.isFinite(Number(payload.embedding_dim)) ? Number(payload.embedding_dim) : null);

  const hasResult = Boolean(savedPath || (embeddingDim && embeddingDim > 0));

  // if server returns "error": null but state is error-ish, handle that:
  if (norm === STATUS.ERROR || job.error || payload.error) {
    return {
      status: STATUS.ERROR,
      training_server_job_id: trainingServerJobId,
      error: job.error || payload.error || job.message || payload.message || "Error",
      message: job.message || payload.message || "",
      saved_path: savedPath,
      has_result: false,
      progress: job.progress || payload.progress || null,
    };
  }

  // DONE only on explicit done/completed OR server-side ready flag (if you add it later)
  const doneFlag = job.done === true || payload.done === true || job.ready === true || payload.ready === true;
  if (norm === STATUS.DONE || doneFlag === true) {
    return {
      status: STATUS.DONE,
      training_server_job_id: trainingServerJobId,
      saved_path: savedPath,
      has_result: true,
      message: job.message || payload.message || "",
      progress: job.progress || payload.progress || { phase: "done" },
      training_metrics: job.training_metrics || payload.training_metrics || null,
    };
  }

  return {
    status: norm || null,
    training_server_job_id: trainingServerJobId,
    saved_path: savedPath,
    has_result: hasResult,
    message: job.message || payload.message || "",
    progress: job.progress || payload.progress || null,
    training_metrics: job.training_metrics || payload.training_metrics || null,
    raw_state: stateRaw,
  };
}

function stableSortByCreatedDesc(items) {
  return items.slice().sort((a, b) => {
    const ta = Date.parse(a.created_at || "") || 0;
    const tb = Date.parse(b.created_at || "") || 0;
    return tb - ta;
  });
}

function looksLikeSendSuccess(sendJson) {
  if (!sendJson || typeof sendJson !== "object") return { ok: false, reason: "no-json" };
  if (String(sendJson.status || "").toLowerCase() === "error") {
    return { ok: false, reason: sendJson.message || sendJson.error || "status=error" };
  }
  const trainingServerJobId = sendJson.training_server_job_id || sendJson.trainingServerJobId || sendJson.training_server_id || null;
  if (!trainingServerJobId || typeof trainingServerJobId !== "string" || !trainingServerJobId.trim()) {
    return { ok: false, reason: "missing training job id (the training server likely failed even if HTTP 200)" };
  }
  return { ok: true, training_server_job_id: trainingServerJobId };
}

async function mapLimit(items, limit, fn) {
  const out = new Array(items.length);
  let i = 0;
  const workers = Array.from({ length: Math.max(1, limit) }, async () => {
    while (i < items.length) {
      const idx = i++;
      try { out[idx] = await fn(items[idx], idx); }
      catch (e) { out[idx] = { __error: String(e?.message || e) }; }
    }
  });
  await Promise.all(workers);
  return out;
}

export function createJobsUI({
  tableBodyEl,
  noJobsEl,
  refreshBtnEl,
  paginationEl,
  paginationInfoEl,
  prevPageBtnEl,
  nextPageBtnEl,
  deleteModalEl,
  deleteModalMessageEl,
  confirmDeleteBtnEl,
  cancelDeleteBtnEl,
  showModal = null,
  hideModal = null,
  pollEveryMs = 5000,
  endpointBase = "training",
  pollConcurrency = 4,
} = {}) {
  let jobs = [];
  let pollTimer = null;
  let refreshHandler = null;
  let prevPageHandler = null;
  let nextPageHandler = null;
  let confirmDeleteHandler = null;
  let cancelDeleteHandler = null;
  let pendingDeleteJobId = null;
  let inFlight = false;
  let currentPage = 1;

  function closeDeleteModal() {
    pendingDeleteJobId = null;
    if (deleteModalEl) {
      if (typeof hideModal === "function") {
        hideModal(deleteModalEl);
      } else {
        deleteModalEl.style.display = "none";
      }
    }
  }

  function openDeleteModal(jobId) {
    pendingDeleteJobId = jobId;
    if (deleteModalMessageEl) {
      const job = jobs.find(j => j.job_id === jobId);
      const appliance = safeStr(job?.appliance_name, "this job");
      deleteModalMessageEl.textContent = `Remove "${appliance}" from the local jobs list?`;
    }
    if (deleteModalEl) {
      if (typeof showModal === "function") {
        showModal(deleteModalEl);
      } else {
        deleteModalEl.style.display = "flex";
        requestAnimationFrame(() => {
          const modalCard = deleteModalEl.querySelector(".modal-content");
          if (modalCard && typeof modalCard.scrollIntoView === "function") {
            modalCard.scrollIntoView({ block: "center", inline: "center" });
          }
        });
      }
    }
  }

  function load() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      const parsed = raw ? JSON.parse(raw) : null;
      jobs = Array.isArray(parsed) ? parsed : [];
    } catch {
      jobs = [];
    }
    jobs = sanitizeForPersistence(jobs);
  }

  function save() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(sanitizeForPersistence(jobs))); }
    catch {}
  }

  function clearSessionTerminalJobs() {
    jobs = jobs.filter((job) => !TERMINAL_STATUSES.has(normalizeStatus(job?.status)));
    for (const job of jobs) {
      if (job && Object.prototype.hasOwnProperty.call(job, "has_been_visible")) {
        delete job.has_been_visible;
      }
    }
    jobs = stableSortByCreatedDesc(jobs).slice(0, MAX_JOBS);
    save();
  }

  function beginNewDraft() {
    clearSessionTerminalJobs();
    render();
  }

  function upsert(job) {
    const id = job.job_id;
    const idx = jobs.findIndex(j => j.job_id === id);
    if (idx >= 0) jobs[idx] = { ...jobs[idx], ...job, updated_at: nowIso() };
    else jobs.unshift(job);
    jobs = stableSortByCreatedDesc(jobs).slice(0, MAX_JOBS);
    save();
  }

  function patch(jobId, partial) {
    const idx = jobs.findIndex(j => j.job_id === jobId);
    if (idx < 0) return;
    jobs[idx] = { ...jobs[idx], ...partial, updated_at: nowIso() };
    jobs = stableSortByCreatedDesc(jobs).slice(0, MAX_JOBS);
    save();
  }

  function render() {
    if (!tableBodyEl) return;

    tableBodyEl.innerHTML = "";
    const jobsPanelEl = tableBodyEl.closest(".panel-surface");
    const hasPrepared = jobs.some((job) =>
      job?.has_seen_prepared === true ||
      job?.status === STATUS.PREPARED ||
      ACTIVE_STATUSES.has(normalizeStatus(job?.status))
    );
    const displayJobs = computeDisplayJobs(jobs);
    const hasDisplayJobs = displayJobs.length > 0;

    if (jobsPanelEl) jobsPanelEl.classList.toggle("hidden", !hasPrepared);
    if (!hasPrepared) {
      if (noJobsEl) noJobsEl.style.display = "none";
      if (paginationEl) paginationEl.classList.add("hidden");
      if (paginationInfoEl) paginationInfoEl.textContent = "";
      if (prevPageBtnEl) prevPageBtnEl.disabled = true;
      if (nextPageBtnEl) nextPageBtnEl.disabled = true;
      return;
    }

    if (noJobsEl) {
      noJobsEl.style.display = hasDisplayJobs ? "none" : "block";
      noJobsEl.textContent = "No jobs in progress.";
    }
    if (paginationEl) paginationEl.classList.toggle("hidden", !(hasDisplayJobs && displayJobs.length > PAGE_SIZE));

    const totalPages = Math.max(1, Math.ceil(displayJobs.length / PAGE_SIZE));
    currentPage = Math.min(Math.max(currentPage, 1), totalPages);
    const startIndex = (currentPage - 1) * PAGE_SIZE;
    const visibleJobs = displayJobs.slice(startIndex, startIndex + PAGE_SIZE);

    if (paginationInfoEl) {
      const from = displayJobs.length ? startIndex + 1 : 0;
      const to = Math.min(startIndex + PAGE_SIZE, displayJobs.length);
      paginationInfoEl.textContent = displayJobs.length
        ? `Showing ${from}-${to} of ${displayJobs.length} jobs`
        : "";
    }

    if (prevPageBtnEl) prevPageBtnEl.disabled = currentPage <= 1;
    if (nextPageBtnEl) nextPageBtnEl.disabled = currentPage >= totalPages;

    let visibilityChanged = false;
    for (const job of visibleJobs) {
      if (job?.has_been_visible !== true) {
        job.has_been_visible = true;
        visibilityChanged = true;
      }
      const appliance = safeStr(job.appliance_name, "-");
      const timeRange = fmtRange(job.time_start_ms, job.time_end_ms);
      const supervisionMode = String(job.supervision_mode || "").toLowerCase();
      const fineTuneTarget = String(job?.training_metrics?.fine_tune_target || job?.fine_tune_target || "").toLowerCase();
      const windowsLabel = supervisionMode === "sensor"
        ? "Ground-truth power supervision"
        : (fineTuneTarget === "weak_onoff"
            ? (Number.isFinite(Number(job.n_windows))
                ? `${Number(job.n_windows)} ON windows (weak supervision)`
                : "Weak interval supervision")
            : (Number.isFinite(Number(job.n_windows))
            ? `${Number(job.n_windows)} ON window${Number(job.n_windows) === 1 ? "" : "s"}`
            : "Interval supervision"));

      const statusBadge = badgeHtml(job.status || "unknown");
      const extra = job.error_message
        ? `<div class="text-xs text-red-700 mt-1">${escapeHtml(job.error_message)}</div>`
        : (() => {
            const line = progressLine(job);
            const metrics = trainingMetricsLine(job);
            const parts = [];
            if (line) parts.push(`<div class="text-xs text-gray-600 mt-1">${escapeHtml(line)}</div>`);
            if (metrics) parts.push(`<div class="text-xs text-gray-600 mt-1">${escapeHtml(metrics)}</div>`);
            return parts.join("");
          })();

      const tr = document.createElement("tr");
      tr.className = "border-b last:border-b-0";
      tr.innerHTML = `
        <td class="py-2 pr-3">
          <div class="job-appliance">${escapeHtml(appliance)}</div>
          <div class="job-meta">${escapeHtml(windowsLabel)}</div>
        </td>
        <td class="py-2 pr-3 text-sm text-gray-500">${escapeHtml(timeRange)}</td>
        <td class="py-2 pr-3">${statusBadge}${extra}</td>
      `;

      tableBodyEl.appendChild(tr);
    }
    if (visibilityChanged) save();
  }

  function recordPrepared({ job_id, appliance_name, appliance_type, selected_windows, supervision_mode, time_start_ms, time_end_ms, message } = {}) {
    if (!job_id) return;
    clearSessionTerminalJobs();
    const range = computeSelectedRange(selected_windows);
    const preparedStartMs = Number.isFinite(Number(time_start_ms)) ? Number(time_start_ms) : range.startMs;
    const preparedEndMs = Number.isFinite(Number(time_end_ms)) ? Number(time_end_ms) : range.endMs;

    upsert({
      job_id,
      created_at: nowIso(),
      updated_at: nowIso(),
      appliance_name: safeStr(appliance_name, "-"),
      appliance_type: safeStr(appliance_type, "-"),
      supervision_mode: safeStr(supervision_mode, "intervals"),
      time_start_ms: preparedStartMs,
      time_end_ms: preparedEndMs,
      n_windows: range.n,
      status: STATUS.PREPARED,
      training_server_job_id: null,
      saved_path: null,
      has_result: false,
      has_embedding: false,
      error_message: null,
      has_seen_prepared: true,
      message: safeStr(message, ""),
      progress: { phase: "prepared" },
      started_at: null,
    });

    render();
  }

  function recordSent({ job_id, training_server_job_id, message } = {}) {
    if (!job_id) return;

    if (!training_server_job_id) {
      patch(job_id, {
        status: STATUS.ERROR,
        error_message: "Send returned HTTP 200 but no training job id (the training server likely failed).",
        message: safeStr(message, ""),
      });
      render();
      return;
    }

    patch(job_id, {
      status: STATUS.QUEUED,
      training_server_job_id,
      error_message: null,
      message: safeStr(message, ""),
      progress: { phase: "queued" },
      // start timer from "sent" time (so user sees elapsed even without server progress)
      started_at: nowIso(),
    });

    render();
  }

  async function pollJobStatus(jobId) {
    const enc = encodeURIComponent(jobId);
    const url = `${endpointBase}?action=status&job_id=${enc}`;

    let r = await safeFetchJson(url, { method: "POST", body: "{}", timeoutMs: 12000 });
    if (!r.ok && (r.status === 405 || r.status === 404)) {
      r = await safeFetchJson(url, { method: "GET", body: null, timeoutMs: 12000 });
    }

    if (r.ok && r.json && typeof r.json === "object") return { ok: true, payload: r.json };
    return { ok: false, httpStatus: r.status, text: r.text };
  }

  async function pollOnce() {
    if (inFlight) return;
    inFlight = true;

    try {
      // Optional staling
      const now = Date.now();
      for (const job of jobs) {
        if (job.status === STATUS.PREPARED) {
          const t = Date.parse(job.created_at || "") || 0;
          if (t && (now - t) > 24 * 3600 * 1000) job.status = STATUS.STALE;
        }
      }

      const candidates = jobs.filter(j => j.training_server_job_id && POLLABLE.has(normalizeStatus(j.status)));
      const results = await mapLimit(candidates, pollConcurrency, async (job) => {
        const res = await pollJobStatus(job.job_id);
        return { job, res };
      });

      for (const item of results) {
        const job = item?.job;
        const res = item?.res;

        if (!job || !res || !res.ok) continue; // don't flip to ERROR on polling failures

        const parsed = parseStatusPayload(res.payload);

        if (parsed.status === STATUS.ERROR) {
          patch(job.job_id, {
            status: STATUS.ERROR,
            error_message: parsed.error || "Training server/backend error.",
            message: parsed.message || "",
            progress: parsed.progress || { phase: "error" },
            has_result: false,
            has_embedding: false,
          });
          continue;
        }

        if (parsed.status === STATUS.DONE) {
          patch(job.job_id, {
            status: STATUS.DONE,
            error_message: null,
            message: parsed.message || "",
            saved_path: parsed.saved_path || job.saved_path || null,
            has_result: true,
            has_embedding: true,
            progress: parsed.progress || { phase: "done" },
            training_metrics: parsed.training_metrics || job.training_metrics || null,
          });
          continue;
        }

        if (parsed.status) {
          // IMPORTANT: if server doesn't send progress, derive it from state so we don't stay "queued" forever
          const derivedProgress =
            parsed.progress ||
            { phase: parsed.status }; // queued/running/etc.

          // start timer when it first becomes running/queued
          const startedAt = job.started_at || (parsed.status === STATUS.RUNNING || parsed.status === STATUS.QUEUED ? nowIso() : null);

          patch(job.job_id, {
            status: parsed.status,
            error_message: null,
            message: parsed.message || "",
            saved_path: parsed.saved_path || job.saved_path || null,
            has_result: Boolean(parsed.has_result),
            progress: derivedProgress,
            training_metrics: parsed.training_metrics || job.training_metrics || null,
            started_at: startedAt,
          });
        }
      }

      save();
      render();
    } finally {
      inFlight = false;
    }
  }

  function init() {
    load();
    save();
    render();

    if (refreshBtnEl) {
      refreshHandler = () => pollOnce().catch(() => {});
      refreshBtnEl.addEventListener("click", refreshHandler);
    }
    if (prevPageBtnEl) {
      prevPageHandler = () => {
        currentPage = Math.max(1, currentPage - 1);
        render();
      };
      prevPageBtnEl.addEventListener("click", prevPageHandler);
    }
    if (nextPageBtnEl) {
      nextPageHandler = () => {
        const displayJobs = computeDisplayJobs(jobs);
        const totalPages = Math.max(1, Math.ceil(displayJobs.length / PAGE_SIZE));
        currentPage = Math.min(totalPages, currentPage + 1);
        render();
      };
      nextPageBtnEl.addEventListener("click", nextPageHandler);
    }
    if (confirmDeleteBtnEl) {
      confirmDeleteHandler = () => {
        if (!pendingDeleteJobId) return;
        jobs = jobs.filter(j => j.job_id !== pendingDeleteJobId);
        save();
        closeDeleteModal();
        const totalPages = Math.max(1, Math.ceil(computeDisplayJobs(jobs).length / PAGE_SIZE));
        currentPage = Math.min(currentPage, totalPages);
        render();
      };
      confirmDeleteBtnEl.addEventListener("click", confirmDeleteHandler);
    }
    if (cancelDeleteBtnEl) {
      cancelDeleteHandler = () => closeDeleteModal();
      cancelDeleteBtnEl.addEventListener("click", cancelDeleteHandler);
    }
    if (deleteModalEl) {
      deleteModalEl.addEventListener("click", (event) => {
        if (event.target === deleteModalEl) closeDeleteModal();
      });
    }

    pollOnce().catch(() => {});
    pollTimer = setInterval(() => pollOnce().catch(() => {}), pollEveryMs);
  }

  function cleanup() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = null;
    if (refreshBtnEl && refreshHandler) refreshBtnEl.removeEventListener("click", refreshHandler);
    if (prevPageBtnEl && prevPageHandler) prevPageBtnEl.removeEventListener("click", prevPageHandler);
    if (nextPageBtnEl && nextPageHandler) nextPageBtnEl.removeEventListener("click", nextPageHandler);
    if (confirmDeleteBtnEl && confirmDeleteHandler) confirmDeleteBtnEl.removeEventListener("click", confirmDeleteHandler);
    if (cancelDeleteBtnEl && cancelDeleteHandler) cancelDeleteBtnEl.removeEventListener("click", cancelDeleteHandler);
    closeDeleteModal();
    refreshHandler = null;
    prevPageHandler = null;
    nextPageHandler = null;
    confirmDeleteHandler = null;
    cancelDeleteHandler = null;
  }

  return {
    init,
    cleanup,
    recordPrepared,
    recordSent,

    async retrySend(jobId) {
      patch(jobId, { message: "Retrying send…", error_message: null });
      render();

      const enc = encodeURIComponent(jobId);
      const r = await safeFetchJson(`${endpointBase}?action=send&job_id=${enc}`, {
        method: "POST",
        body: "{}",
        timeoutMs: 20000,
      });

      if (!r.ok || !r.json) {
        patch(jobId, {
          status: STATUS.ERROR,
          error_message: `Retry send failed: HTTP ${r.status} ${String(r.text || "").slice(0, 200)}`,
        });
        render();
        return;
      }

      const verdict = looksLikeSendSuccess(r.json);
      if (!verdict.ok) {
        patch(jobId, {
          status: STATUS.ERROR,
          error_message: `Retry send suspicious: ${verdict.reason}`,
          message: r.json?.message || "",
        });
        render();
        return;
      }

      patch(jobId, {
        status: STATUS.QUEUED,
        training_server_job_id: verdict.training_server_job_id,
        error_message: null,
        message: r.json?.message || "Sent to training server.",
        has_result: false,
        has_embedding: false,
        saved_path: null,
        progress: { phase: "queued" },
        started_at: nowIso(),
      });

      render();
    },
    requestDeleteJob(jobId) {
      openDeleteModal(jobId);
    },
    beginNewDraft,
    deleteJob(jobId) {
      jobs = jobs.filter(j => j.job_id !== jobId);
      save();
      const totalPages = Math.max(1, Math.ceil(computeDisplayJobs(jobs).length / PAGE_SIZE));
      currentPage = Math.min(currentPage, totalPages);
      render();
    },
    _debug: {
      pollOnce,
      getJobs: () => jobs.slice(),
      clearStorage: () => localStorage.removeItem(STORAGE_KEY),
    },
  };
}

