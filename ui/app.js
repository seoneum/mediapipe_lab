const state = {
  dossiers: [],
  current: null,
  tab: "overview",
  recommendationPayload: null,
  recommendationDraft: null,
  integrations: null,
  clips: [],
  selectedClipId: null,
  clipAnalysis: null,
  gptReview: null,
  clipBusy: null,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

function escapeHtml(value = "") {
  return String(value).replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[char]);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    method: options.method || "GET",
    headers: options.body ? { "Content-Type": "application/json" } : undefined,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.message || `요청 실패 (${response.status})`);
  return payload;
}

function toast(message, type = "success") {
  const item = document.createElement("div");
  item.className = `toast ${type === "error" ? "error" : ""}`;
  item.textContent = message;
  $("#toast-region").append(item);
  window.setTimeout(() => item.remove(), 3600);
}

function formatDate(value, withTime = false) {
  if (!value) return "–";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("ko-KR", {
    year: "numeric", month: "short", day: "numeric",
    ...(withTime ? { hour: "2-digit", minute: "2-digit" } : {}),
  }).format(date);
}

function initials(name = "") {
  const text = name.trim();
  if (!text) return "–";
  return [...text].slice(0, 2).join("");
}

function splitList(value) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function chipMarkup(items, emptyText = "아직 확인된 정보 없음") {
  if (!items?.length) return `<span class="chip empty">${escapeHtml(emptyText)}</span>`;
  return items.map((item) => `<span class="chip">${escapeHtml(item)}</span>`).join("");
}

function noteMarkup(items) {
  if (!items?.length) return `<div class="empty-inline">아직 인수인계 메모가 없습니다.</div>`;
  return items.map((item) => `<div>${escapeHtml(item)}</div>`).join("");
}

async function loadDossiers(preferredId = null) {
  state.dossiers = await api("/api/dossiers");
  renderChildList();
  if (!state.dossiers.length) {
    state.current = null;
    renderEmpty();
    return;
  }
  const remembered = preferredId || localStorage.getItem("ondamm-selected-child");
  const target = state.dossiers.find((item) => item.child_id === remembered) || state.dossiers[0];
  await selectChild(target.child_id);
}

function renderChildList() {
  const list = $("#child-list");
  list.innerHTML = state.dossiers.map((item) => `
    <button class="child-option ${state.current?.child_id === item.child_id ? "selected" : ""}"
      type="button" role="option" aria-selected="${state.current?.child_id === item.child_id}"
      data-child-id="${escapeHtml(item.child_id)}">
      <span class="child-avatar">${escapeHtml(initials(item.display_name))}</span>
      <div><strong>${escapeHtml(item.display_name)}</strong><small>${escapeHtml(item.age_band)} · ${item.canonical_status === "active" ? "활성" : "잠금"}</small></div>
    </button>`).join("");
}

async function loadLocalClips({ preserveSelection = true, silent = false } = {}) {
  if (!state.current) return;
  const previous = preserveSelection ? state.selectedClipId : null;
  try {
    state.clips = await api(`/api/dossiers/${encodeURIComponent(state.current.child_id)}/clips`);
    state.selectedClipId = state.clips.some((clip) => clip.clip_id === previous)
      ? previous
      : (state.clips[0]?.clip_id || null);
  } catch (error) {
    state.clips = [];
    state.selectedClipId = null;
    if (!silent) toast(error.message, "error");
  }
}

async function selectChild(childId) {
  state.current = await api(`/api/dossiers/${encodeURIComponent(childId)}`);
  localStorage.setItem("ondamm-selected-child", childId);
  state.recommendationPayload = null;
  state.recommendationDraft = null;
  state.clipAnalysis = null;
  state.gptReview = null;
  state.clipBusy = null;
  if (!state.integrations) {
    state.integrations = await api("/api/integrations").catch(() => ({
      mediapipe: { configured: false }, openai: { configured: false, model: null },
    }));
  }
  await loadLocalClips({ preserveSelection: false, silent: true });
  renderChildList();
  renderCurrent();
  document.body.classList.remove("sidebar-open");
}

function renderEmpty() {
  $("#empty-screen").hidden = false;
  $$(".view").forEach((view) => { view.hidden = true; });
  $("#subject-name").textContent = "지원 기록철을 선택하세요";
  $("#subject-meta").textContent = "아동별 지원 정보와 승인된 기록을 안전하게 관리합니다.";
  $("#subject-avatar").textContent = "–";
  $("#status-pill").hidden = true;
  $("#manage-status-button").disabled = true;
  $$('[data-action="new-session"]').forEach((button) => { button.disabled = true; });
}

function renderCurrent() {
  const dossier = state.current;
  if (!dossier) return renderEmpty();
  $("#empty-screen").hidden = true;
  $("#subject-name").textContent = dossier.display_name;
  $("#subject-meta").textContent = `${dossier.age_band} · ${dossier.communication_modality}`;
  $("#subject-avatar").textContent = initials(dossier.display_name);
  $("#welcome-name").textContent = dossier.display_name;
  $("#updated-at").textContent = formatDate(dossier.updated_at, true);
  $("#manage-status-button").disabled = false;

  const active = dossier.canonical_status === "active";
  const statusPill = $("#status-pill");
  statusPill.hidden = false;
  statusPill.textContent = active ? "활성" : "동의 철회 · 잠금";
  statusPill.classList.toggle("locked", !active);
  document.body.classList.toggle("locked", !active);
  $$('[data-action="new-session"]').forEach((button) => { button.disabled = !active; });

  $("#session-count").textContent = dossier.approved_session_summaries.length;
  $("#plan-count").textContent = dossier.approved_plan_history.length;
  const ready = [
    dossier.confirmed_preferences.length || dossier.effective_strategies.length,
    dossier.approved_plan_history.length,
    dossier.approved_session_summaries.length,
  ].filter(Boolean).length;
  $("#readiness-count").textContent = `${ready} / 3`;

  $("#overview-preferences").innerHTML = chipMarkup(dossier.confirmed_preferences);
  $("#overview-strategies").innerHTML = chipMarkup(dossier.effective_strategies);
  $("#overview-supports").innerHTML = chipMarkup(dossier.triggers_and_calming_supports);

  $("#detail-name").textContent = dossier.display_name;
  $("#detail-age").textContent = dossier.age_band;
  $("#detail-communication").textContent = dossier.communication_modality;
  $("#detail-id").textContent = dossier.local_canonical_id;
  $("#detail-preferences").innerHTML = chipMarkup(dossier.confirmed_preferences);
  $("#detail-avoidances").innerHTML = chipMarkup(dossier.confirmed_avoidances);
  $("#detail-strategies").innerHTML = chipMarkup(dossier.effective_strategies);
  $("#detail-supports").innerHTML = chipMarkup(dossier.triggers_and_calming_supports);
  $("#detail-handoff").innerHTML = noteMarkup(dossier.handoff_notes);
  const facialProfiles = dossier.approved_facial_movement_profiles || [];
  $("#detail-facial-profiles").innerHTML = facialProfiles.length
    ? facialProfiles.map((profile) => `<div><strong>${escapeHtml(profile.display_name)}</strong> · ${escapeHtml(profile.blendshape_names.join(", "))} · threshold ${Number(profile.activation_threshold).toFixed(2)} · 승인 ${escapeHtml(profile.approved_by)}</div>`).join("")
    : `<div class="empty-inline">기본 얼굴 움직임 규칙을 사용 중입니다. 승인된 개인화 프로필은 아직 없습니다.</div>`;

  renderSessions();
  renderPlans();
  renderRecommendationPlaceholder();
  $("#learning-plan").innerHTML = "";
  $("#sensing-result").innerHTML = `<div class="preview-placeholder"><span class="draft-stamp">DRAFT</span><h3>관찰 전입니다</h3><p>결과는 임시 초안으로만 표시되고 공식 기록에는 들어가지 않습니다.</p></div>`;
  renderClipReview();
  $("#export-result").innerHTML = "";
  setTab(state.tab);
}

function renderSessions() {
  const sessions = [...state.current.approved_session_summaries].reverse();
  const recent = sessions.slice(0, 3);
  $("#recent-sessions").innerHTML = recent.length ? recent.map((item) => `
    <div class="record-row"><div><strong>${escapeHtml(item.title)}</strong><p>${escapeHtml(item.activity_name)} · ${escapeHtml(item.observed_response)}</p></div><time>${formatDate(item.created_at)}</time></div>`).join("") : `<div class="empty-inline">아직 승인된 수업 기록이 없습니다.</div>`;

  $("#session-list").innerHTML = sessions.length ? sessions.map((item) => `
    <article class="timeline-item">
      <div class="timeline-meta"><time>${formatDate(item.created_at, true)}</time><span>승인 · ${escapeHtml(item.approved_by)}</span><code>${escapeHtml(item.session_id)}</code></div>
      <div class="timeline-content"><h3>${escapeHtml(item.title)}</h3><p>${escapeHtml(item.activity_name)} ${item.tags?.length ? `· ${item.tags.map(escapeHtml).join(" · ")}` : ""}</p>
        <div class="observation-pair"><div><strong>관찰한 반응</strong><span>${escapeHtml(item.observed_response)}</span></div><div><strong>교육자 해석</strong><span>${escapeHtml(item.educator_interpretation)}</span></div></div>
      </div>
    </article>`).join("") : `<div class="card empty-inline">아직 승인된 수업 기록이 없습니다. 첫 기록을 남겨 보세요.</div>`;
}

function renderPlans() {
  const plans = [...state.current.approved_plan_history].reverse();
  $("#approved-plans").innerHTML = plans.length ? plans.map((plan) => `
    <article class="plan-item"><header><div><p class="eyebrow">${formatDate(plan.created_at)}</p><h4>${escapeHtml(plan.goal)}</h4></div><span>승인 · ${escapeHtml(plan.approved_by || "미승인")}</span></header><p>${escapeHtml(plan.summary)}</p></article>`).join("") : `<div class="card empty-inline">아직 승인된 활동 계획이 없습니다.</div>`;
}

function eventTypeName(value) {
  return ({ gaze_diverted: "시선 구역 변화", face_missing: "얼굴 프레임 이탈", posture_shifted: "자세 쏠림" })[value] || value || "이벤트";
}

function expressionHintName(value) {
  return ({
    neutral: "두드러진 움직임 없음",
    eyes_closed: "양눈 닫힘 움직임",
    left_eye_closed: "왼눈 닫힘 움직임",
    right_eye_closed: "오른눈 닫힘 움직임",
    mouth_smile: "입꼬리 상승 움직임",
    jaw_open: "턱 열림 움직임",
    brow_raise: "눈썹 올림 움직임",
    brow_lower: "눈썹 내림 움직임",
    eye_squint: "눈 가늘게 뜸 움직임",
    eyes_wide: "눈 크게 뜸 움직임",
    mouth_frown: "입꼬리 하강 움직임",
    lip_pucker: "입술 오므림 움직임",
    lip_press: "입술 누름 움직임",
    mouth_stretch: "입 늘림 움직임",
    mouth_dimple: "입꼬리 당김 움직임",
    cheek_puff: "볼 부풀림 움직임",
    nose_sneer: "코 주변 올림 움직임",
    mouth_left: "입 왼쪽 이동",
    mouth_right: "입 오른쪽 이동",
    tongue_out: "혀 내밈 움직임",
    smile: "입꼬리 상승 계열",
    surprise: "턱 열림·눈 크게 뜸 계열",
    blink: "눈 감김 계열",
    frown: "입꼬리 하강·눈썹 내림 계열",
    squint: "눈 가늘게 뜸 계열",
  })[value] || value || "확인 불가";
}

function renderClipReview() {
  const workspace = $("#clip-review-workspace");
  if (!workspace || !state.current) return;
  if (state.current.canonical_status !== "active") {
    workspace.innerHTML = `<article class="card clip-lock"><strong>동의 철회로 잠김</strong><p>로컬 영상 재생, MediaPipe 분석, GPT 프레임 검토가 모두 차단되었습니다.</p></article>`;
    return;
  }
  if (!state.clips.length) {
    workspace.innerHTML = `<article class="card clip-empty"><span class="clip-empty-icon">▣</span><h4>불러올 로컬 이벤트 영상이 없습니다</h4><p><code>scripts/ondamm_learning.sh --record-events</code>로 생성한 메타데이터 기반 MP4만 표시합니다.</p></article>`;
    return;
  }

  const selected = state.clips.find((clip) => clip.clip_id === state.selectedClipId) || state.clips[0];
  const mediaPipe = state.integrations?.mediapipe || { configured: false };
  const openAI = state.integrations?.openai || { configured: false };
  const triggerMarkup = Object.entries(selected.trigger_values || {}).map(([key, value]) => `
    <span><b>${escapeHtml(key)}</b>${escapeHtml(value)}</span>`).join("") || `<span><b>trigger</b>메타데이터 없음</span>`;
  const expressionEntries = Object.entries(state.clipAnalysis?.expression_label_counts || {});
  const analysisMarkup = state.clipAnalysis ? `
    <div class="analysis-result">
      <div class="analysis-summary"><span>대표 움직임 힌트</span><strong>${escapeHtml(expressionHintName(state.clipAnalysis.dominant_expression_hint))}</strong></div>
      <div class="analysis-counts">${expressionEntries.length ? expressionEntries.map(([label, count]) => `<span>${escapeHtml(expressionHintName(label))}<b>${count}</b></span>`).join("") : `<span>분석한 프레임에서 얼굴 blendshape를 확인하지 못했습니다.</span>`}</div>
      <p>${escapeHtml(state.clipAnalysis.non_diagnostic_notice || "표정 움직임 힌트이며 감정 상태로 확정하지 않습니다.")}</p>
    </div>` : `<div class="analysis-placeholder">선택 영상을 이 Mac 안에서만 샘플링해 얼굴 blendshape 움직임을 확인합니다.</div>`;
  const gptMarkup = state.gptReview ? `
    <div class="gpt-result"><div class="gpt-result-head"><strong>GPT 관찰 보조 초안</strong><span>${escapeHtml(state.gptReview.model || "GPT")}</span></div><div class="gpt-text">${escapeHtml(state.gptReview.review_text).replace(/\n/g, "<br>")}</div><p>전체 영상 전송 안 함 · 추출 프레임 ${state.gptReview.remote_frame_count}장 · 기록철 자동 반영 안 함</p></div>` : "";

  workspace.innerHTML = `
    <div class="integration-strip card">
      <div><span class="integration-dot ready"></span><strong>MediaPipe</strong><small>로컬 얼굴 blendshape</small></div>
      <div><span class="integration-dot ${openAI.configured ? "ready" : "off"}"></span><strong>GPT</strong><small>${openAI.configured ? escapeHtml(openAI.model || "연결됨") : "OPENAI_API_KEY 필요"}</small></div>
      <p>GPT는 버튼을 누르고 동의한 경우에만 영상에서 추출한 최대 3장 프레임을 전송합니다.</p>
    </div>
    <div class="clip-review-grid">
      <aside class="card clip-list-panel">
        <div class="clip-panel-head"><span>이벤트 영상</span><b>${state.clips.length}</b></div>
        <div class="clip-list">${state.clips.map((clip) => `
          <button type="button" data-clip-id="${escapeHtml(clip.clip_id)}" class="clip-item ${clip.clip_id === selected.clip_id ? "selected" : ""}">
            <span class="clip-type">${escapeHtml(eventTypeName(clip.event_type))}</span>
            <strong>${escapeHtml(clip.event_id)}</strong>
            <small>${clip.duration_seconds.toFixed(1)}초 · ${formatDate(clip.created_at, true)}</small>
          </button>`).join("")}</div>
      </aside>
      <article class="card clip-player-panel">
        <div class="clip-panel-head"><div><span>로컬 재생</span><strong>${escapeHtml(eventTypeName(selected.event_type))}</strong></div><span class="local-only-pill">LOCAL ONLY</span></div>
        <video controls playsinline preload="metadata" src="${escapeHtml(selected.media_url)}" aria-label="선택한 로컬 특이 이벤트 영상"></video>
        <div class="clip-trigger-grid">${triggerMarkup}</div>
        <p class="clip-path">${escapeHtml(selected.relative_path)}</p>
      </article>
      <article class="card clip-analysis-panel">
        <div class="analysis-block">
          <div class="clip-panel-head"><div><span>표정 움직임 힌트</span><strong>Google MediaPipe</strong></div><span class="local-only-pill">로컬</span></div>
          ${analysisMarkup}
          <button class="button button-secondary full-button" id="run-mediapipe-analysis" type="button" ${state.clipBusy || !mediaPipe.configured ? "disabled" : ""}>${state.clipBusy === "mediapipe" ? "로컬 분석 중…" : "MediaPipe로 분석"}</button>
        </div>
        <div class="analysis-block gpt-block">
          <div class="clip-panel-head"><div><span>선택적 원격 검토</span><strong>GPT 프레임 검토</strong></div><span class="remote-pill">최대 3장</span></div>
          ${openAI.configured ? `
            <label class="consent-check"><input type="checkbox" id="gpt-frame-consent"><span>선택 영상에서 추출한 프레임을 OpenAI API로 전송하는 데 동의합니다. 전체 MP4는 전송하지 않습니다.</span></label>
            <button class="button button-quiet full-button" id="run-gpt-review" type="button" ${state.clipBusy ? "disabled" : ""}>${state.clipBusy === "gpt" ? "GPT 검토 중…" : "동의 후 GPT 검토"}</button>` : `
            <div class="key-missing"><strong>GPT 연결 대기</strong><p>서버 실행 전에 환경 변수 <code>OPENAI_API_KEY</code>를 설정하세요. 키는 브라우저로 전달되지 않습니다.</p></div>`}
          ${gptMarkup}
        </div>
      </article>
    </div>`;
}

function setTab(tab) {
  state.tab = tab;
  $$(".tab").forEach((button) => button.classList.toggle("active", button.dataset.tab === tab));
  $$(".view").forEach((view) => { view.hidden = view.dataset.view !== tab || !state.current; });
  if (state.current) $("#empty-screen").hidden = true;
}

function openCreateDialog() {
  $("#create-dialog").showModal();
  window.setTimeout(() => $("#create-form input[name='child_id']").focus(), 30);
}

function openSessionDialog() {
  if (!state.current || state.current.canonical_status !== "active") return toast("잠긴 기록철에는 수업 기록을 추가할 수 없습니다.", "error");
  $("#session-dialog").showModal();
}

function renderRecommendationPlaceholder() {
  $("#recommendation-preview").innerHTML = `<div class="preview-placeholder"><div class="placeholder-lines"><i></i><i></i><i></i></div><h3>초안이 여기에 표시됩니다</h3><p>왼쪽에서 목표를 입력하면 근거와 제안 활동을 검토할 수 있습니다.</p></div>`;
}

function renderRecommendation(draft) {
  $("#recommendation-preview").innerHTML = `
    <div class="recommendation-result">
      <div class="result-head"><div><p class="eyebrow">REVIEW DRAFT · NOT SAVED</p><h3>${escapeHtml(draft.goal)}</h3></div><span class="draft-banner">승인 전 초안</span></div>
      <p class="summary">${escapeHtml(draft.summary)}</p>
      <h4>제안 활동</h4><ol>${draft.suggested_activities.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ol>
      <h4>근거</h4><ul>${draft.rationale_lines.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
      <div class="approval-box"><label>최종 승인자<input id="recommendation-approver" value="${escapeHtml(state.recommendationPayload.drafted_by)}"></label><button class="button button-primary" id="approve-recommendation" type="button">검토 후 승인 저장</button></div>
    </div>`;
}

async function refreshCurrent() {
  if (!state.current) return;
  await loadDossiers(state.current.child_id);
}

$("#child-list").addEventListener("click", (event) => {
  const option = event.target.closest("[data-child-id]");
  if (option) selectChild(option.dataset.childId).catch((error) => toast(error.message, "error"));
});

$$('[data-tab]').forEach((button) => button.addEventListener("click", () => setTab(button.dataset.tab)));
document.addEventListener("click", (event) => {
  const go = event.target.closest("[data-go]");
  if (go) setTab(go.dataset.go);
  const action = event.target.closest("[data-action='new-session']");
  if (action) openSessionDialog();
  const close = event.target.closest("[data-close-dialog]");
  if (close) close.closest("dialog").close();
});

$("#new-child-button").addEventListener("click", openCreateDialog);
$("#empty-create-button").addEventListener("click", openCreateDialog);
$("#mobile-menu").addEventListener("click", () => document.body.classList.toggle("sidebar-open"));

$("#create-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = new FormData(event.currentTarget);
  const body = {
    child_id: data.get("child_id").trim(), display_name: data.get("display_name").trim(),
    age_band: data.get("age_band").trim(), communication_modality: data.get("communication_modality").trim(),
    confirmed_preferences: splitList(data.get("confirmed_preferences")),
    effective_strategies: splitList(data.get("effective_strategies")),
    triggers_and_calming_supports: splitList(data.get("triggers_and_calming_supports")),
  };
  try {
    const created = await api("/api/dossiers", { method: "POST", body });
    $("#create-dialog").close(); event.currentTarget.reset();
    toast(`${created.display_name}님의 로컬 지원 기록철을 만들었습니다.`);
    await loadDossiers(created.child_id);
  } catch (error) { toast(error.message, "error"); }
});

$("#session-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = new FormData(event.currentTarget);
  const body = {
    title: data.get("title").trim(), activity_name: data.get("activity_name").trim(),
    observed_response: data.get("observed_response").trim(), educator_interpretation: data.get("educator_interpretation").trim(),
    approved_by: data.get("approved_by").trim(), tags: splitList(data.get("tags")),
  };
  try {
    await api(`/api/dossiers/${encodeURIComponent(state.current.child_id)}/sessions`, { method: "POST", body });
    $("#session-dialog").close(); event.currentTarget.reset();
    event.currentTarget.elements.approved_by.value = "local-operator";
    toast("승인된 수업 기록을 저장했습니다.");
    await refreshCurrent(); setTab("sessions");
  } catch (error) { toast(error.message, "error"); }
});

$("#new-facial-profile-button").addEventListener("click", () => {
  if (!state.current || state.current.canonical_status !== "active") return;
  $("#facial-profile-dialog").showModal();
});

$("#facial-profile-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = new FormData(event.currentTarget);
  const body = {
    label: data.get("label").trim(),
    display_name: data.get("display_name").trim(),
    blendshape_names: splitList(data.get("blendshape_names")),
    aggregation: data.get("aggregation"),
    activation_threshold: Number(data.get("activation_threshold")),
    approved_by: data.get("approved_by").trim(),
    source_session_ids: splitList(data.get("source_session_ids")),
  };
  try {
    await api(`/api/dossiers/${encodeURIComponent(state.current.child_id)}/facial-movement-profiles/approve`, { method: "POST", body });
    $("#facial-profile-dialog").close(); event.currentTarget.reset();
    event.currentTarget.elements.activation_threshold.value = "0.35";
    event.currentTarget.elements.approved_by.value = "local-operator";
    toast("승인된 얼굴 움직임 프로필을 기록철에 저장했습니다.");
    await refreshCurrent(); setTab("dossier");
  } catch (error) { toast(error.message, "error"); }
});

$("#recommendation-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = new FormData(event.currentTarget);
  state.recommendationPayload = {
    goal: data.get("goal").trim(), caregiver_input: data.get("caregiver_input").trim(), drafted_by: data.get("drafted_by").trim(),
  };
  try {
    state.recommendationDraft = await api(`/api/dossiers/${encodeURIComponent(state.current.child_id)}/recommendations/preview`, { method: "POST", body: state.recommendationPayload });
    renderRecommendation(state.recommendationDraft);
  } catch (error) { toast(error.message, "error"); }
});

$("#recommendation-preview").addEventListener("click", async (event) => {
  if (!event.target.closest("#approve-recommendation")) return;
  const approvedBy = $("#recommendation-approver").value.trim();
  if (!approvedBy) return toast("승인자를 입력해 주세요.", "error");
  try {
    await api(`/api/dossiers/${encodeURIComponent(state.current.child_id)}/recommendations/approve`, {
      method: "POST", body: { ...state.recommendationPayload, approved_by: approvedBy },
    });
    toast("검토한 활동 계획을 승인 기록으로 저장했습니다.");
    await refreshCurrent(); setTab("plan");
  } catch (error) { toast(error.message, "error"); }
});

$("#learning-plan-button").addEventListener("click", async () => {
  const form = $("#recommendation-form");
  const goal = form.elements.goal.value.trim();
  if (!goal) return toast("먼저 이번 목표를 입력해 주세요.", "error");
  try {
    const plan = await api(`/api/dossiers/${encodeURIComponent(state.current.child_id)}/learning-plan/preview`, {
      method: "POST", body: { goal, caregiver_input: form.elements.caregiver_input.value.trim() },
    });
    $("#learning-plan").innerHTML = plan.steps.map((step, index) => `
      <article class="learning-step"><span class="number">${index + 1}</span><h4>${escapeHtml(step.title)}</h4><p>${escapeHtml(step.activity_focus)}</p><time>${Math.round(step.duration_seconds / 60)}분</time></article>`).join("");
    toast("5단계 학습 프로그램 초안을 만들었습니다. 기록철에는 저장되지 않았습니다.");
  } catch (error) { toast(error.message, "error"); }
});

$("#clip-refresh-button").addEventListener("click", async () => {
  await loadLocalClips({ preserveSelection: true });
  state.clipAnalysis = null;
  state.gptReview = null;
  renderClipReview();
  toast(`${state.clips.length}개의 로컬 이벤트 영상을 확인했습니다.`);
});

$("#clip-review-workspace").addEventListener("click", async (event) => {
  const clipButton = event.target.closest("[data-clip-id]");
  if (clipButton) {
    state.selectedClipId = clipButton.dataset.clipId;
    state.clipAnalysis = null;
    state.gptReview = null;
    renderClipReview();
    return;
  }
  if (event.target.closest("#run-mediapipe-analysis")) {
    state.clipBusy = "mediapipe";
    renderClipReview();
    try {
      state.clipAnalysis = await api(`/api/dossiers/${encodeURIComponent(state.current.child_id)}/clips/${encodeURIComponent(state.selectedClipId)}/mediapipe`, { method: "POST", body: {} });
      toast("MediaPipe 로컬 분석을 완료했습니다. 감정 판정이 아닌 움직임 힌트입니다.");
    } catch (error) {
      toast(error.message, "error");
    } finally {
      state.clipBusy = null;
      renderClipReview();
    }
    return;
  }
  if (event.target.closest("#run-gpt-review")) {
    const consent = $("#gpt-frame-consent")?.checked === true;
    if (!consent) return toast("추출 프레임 원격 전송 동의를 먼저 확인해 주세요.", "error");
    state.clipBusy = "gpt";
    renderClipReview();
    try {
      state.gptReview = await api(`/api/dossiers/${encodeURIComponent(state.current.child_id)}/clips/${encodeURIComponent(state.selectedClipId)}/gpt-review`, {
        method: "POST", body: { confirm_remote_frame_upload: true },
      });
      toast("GPT 관찰 보조 초안을 받았습니다. 공식 기록에는 반영되지 않았습니다.");
    } catch (error) {
      toast(error.message, "error");
    } finally {
      state.clipBusy = null;
      renderClipReview();
    }
  }
});

$("#sensing-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = new FormData(event.currentTarget);
  try {
    const result = await api(`/api/dossiers/${encodeURIComponent(state.current.child_id)}/sensing/demo`, {
      method: "POST", body: { duration_seconds: Number(data.get("duration_seconds")), audio_presence_note: data.get("audio_presence_note").trim() },
    });
    const expressionHint = Object.entries(result.facial_movement_counts || result.expression_label_counts || {}).sort((a, b) => b[1] - a[1])[0]?.[0];
    $("#sensing-result").innerHTML = `
      <div class="result-head"><div><p class="eyebrow">REVIEWED NOTE DRAFT</p><h3>관찰 메모 초안</h3></div><span class="draft-stamp">DRAFT</span></div>
      <div class="sensing-metrics"><div><span>프레임</span><strong>${result.frame_count}</strong></div><div><span>얼굴 존재</span><strong>${Math.round(result.face_present_ratio * 100)}%</strong></div><div><span>자세 추정</span><strong>${Math.round(result.pose_present_ratio * 100)}%</strong></div><div><span>표정 움직임</span><strong>${escapeHtml(expressionHintName(expressionHint))}</strong></div></div>
      <div class="draft-lines">${result.reviewed_note_draft.map((line) => `<p>${escapeHtml(line)}</p>`).join("")}</div>
      <div class="non-authoritative"><strong>비권위 초안</strong><br>${escapeHtml(result.non_authoritative_notice)}</div>`;
    toast("데모 관찰 초안을 만들었습니다. 공식 기록에는 반영되지 않았습니다.");
  } catch (error) { toast(error.message, "error"); }
});

$("#export-button").addEventListener("click", async () => {
  const actorId = $("#export-actor").value.trim();
  if (!actorId) return toast("담당자를 입력해 주세요.", "error");
  try {
    const result = await api(`/api/dossiers/${encodeURIComponent(state.current.child_id)}/handoff/export`, { method: "POST", body: { actor_id: actorId } });
    $("#export-result").innerHTML = `<div class="export-success"><strong>생성 완료 · ${escapeHtml(result.manifest.artifact_id)}</strong>문서: ${escapeHtml(result.export_path)}<br>서명: ${escapeHtml(result.manifest_path)}</div>`;
    toast("서명된 인수인계 자료를 만들었습니다.");
    await refreshCurrent(); setTab("handoff");
  } catch (error) { toast(error.message, "error"); }
});

$("#manage-status-button").addEventListener("click", () => {
  if (!state.current) return;
  const active = state.current.canonical_status === "active";
  const form = $("#status-form");
  form.elements.status.value = active ? "withdrawn_locked" : "active";
  form.elements.reason_code.value = active ? "consent_withdrawn" : "consent_restored";
  form.elements.reason.value = "";
  $("#status-dialog-copy").innerHTML = active
    ? `<div class="status-warning"><strong>동의 철회 잠금</strong><br>조회·수정·추천·인수인계·내보내기를 막습니다. 민감한 상태 변경이므로 사유를 기록해 주세요.</div>`
    : `<div class="status-restore"><strong>기록철 활성 복구</strong><br>승인된 근거가 있을 때만 다시 활성화합니다. 복구 사유는 감사 기록에 남습니다.</div>`;
  $("#status-submit").textContent = active ? "동의 철회로 잠금" : "승인 근거로 활성 복구";
  $("#status-submit").className = `button ${active ? "button-danger" : "button-primary"}`;
  $("#status-dialog").showModal();
});

$("#status-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = new FormData(event.currentTarget);
  try {
    await api(`/api/dossiers/${encodeURIComponent(state.current.child_id)}/status`, {
      method: "POST", body: Object.fromEntries(data.entries()),
    });
    $("#status-dialog").close();
    toast(data.get("status") === "active" ? "지원 기록철을 활성 상태로 복구했습니다." : "동의 철회 상태로 잠갔습니다.");
    await refreshCurrent();
  } catch (error) { toast(error.message, "error"); }
});

loadDossiers().catch((error) => {
  console.error(error);
  toast(`앱을 불러오지 못했습니다: ${error.message}`, "error");
});
