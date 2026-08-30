const state = {
  dossiers: [],
  current: null,
  tab: "overview",
  recommendationPayload: null,
  recommendationDraft: null,
  ragResult: null,
  ragHistory: [],
  ragBusy: false,
  integrations: null,
  clips: [],
  selectedClipId: null,
  clipAnalysis: null,
  gptReview: null,
  eventReviews: null,
  clipBusy: null,
  patternMemory: null,
  patternBusy: null,
  rights: null,
  acclimationCompleted: false,
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

async function loadEventReviews({ silent = false } = {}) {
  state.eventReviews = null;
  if (!state.current || !state.selectedClipId) return;
  try {
    state.eventReviews = await api(`/api/dossiers/${encodeURIComponent(state.current.child_id)}/clips/${encodeURIComponent(state.selectedClipId)}/reviews`);
  } catch (error) {
    if (!silent) toast(error.message, "error");
  }
}

async function loadPatternMemory({ silent = false } = {}) {
  state.patternMemory = null;
  if (!state.current) return;
  try {
    state.patternMemory = await api(`/api/dossiers/${encodeURIComponent(state.current.child_id)}/patterns`);
  } catch (error) {
    if (!silent) toast(error.message, "error");
  }
}

async function selectChild(childId) {
  state.current = await api(`/api/dossiers/${encodeURIComponent(childId)}`);
  state.rights = await api(`/api/dossiers/${encodeURIComponent(childId)}/rights`);
  state.acclimationCompleted = false;
  localStorage.setItem("ondamm-selected-child", childId);
  state.recommendationPayload = null;
  state.recommendationDraft = null;
  state.ragResult = null;
  state.ragHistory = [];
  state.ragBusy = false;
  state.clipAnalysis = null;
  state.gptReview = null;
  state.eventReviews = null;
  state.clipBusy = null;
  state.patternBusy = null;
  if (!state.integrations) {
    state.integrations = await api("/api/integrations").catch(() => ({
      mediapipe: { configured: false }, openai: { configured: false, model: null }, llm: { configured: false },
    }));
  }
  await loadLocalClips({ preserveSelection: false, silent: true });
  await loadEventReviews({ silent: true });
  await loadPatternMemory({ silent: true });
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
  $("#child-stop-button").disabled = true;
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
  $("#child-stop-button").disabled = false;

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
    ? facialProfiles.map((profile) => `<div><strong>${escapeHtml(profile.display_name)}</strong> · ${escapeHtml(profile.blendshape_names.join(", "))} · 활성 기준 ${Number(profile.activation_threshold).toFixed(2)} · 승인 ${escapeHtml(profile.approved_by)}</div>`).join("")
    : `<div class="empty-inline">기본 얼굴 움직임 규칙을 사용 중입니다. 승인된 개인화 프로필은 아직 없습니다.</div>`;

  renderSessions();
  renderPlans();
  renderRecommendationPlaceholder();
  renderLocalRag();
  $("#learning-plan").innerHTML = "";
  $("#sensing-result").innerHTML = `<div class="preview-placeholder"><span class="draft-stamp">초안</span><h3>관찰 전입니다</h3><p>결과는 임시 초안으로만 표시되고 공식 기록에는 들어가지 않습니다.</p></div>`;
  renderClipReview();
  renderPatternMemory();
  renderRights();
  $("#export-result").innerHTML = "";
  setTab(state.tab);
}

function renderRights() {
  const rights = state.rights;
  if (!rights || !state.current) return;
  const camera = rights.purposes?.camera_capture;
  const research = rights.purposes?.research_metrics;
  const ready = rights.pre_session_ready && camera?.active;
  const recordActive = state.current.canonical_status === "active";
  const badge = $("#rights-ready-badge");
  badge.textContent = ready ? "촬영 준비 확인됨" : "촬영 준비 안 됨";
  badge.classList.toggle("locked", !ready);
  $("#rights-summary").innerHTML = `
    ${recordActive ? "" : `<div class="rights-state stop"><strong>기록철 잠김</strong><span>동의 철회 상태이므로 새 동의와 교육 전 확인을 등록할 수 없습니다.</span></div>`}
    <div class="rights-state ${camera?.active ? "ok" : "warn"}"><strong>촬영 동의</strong><span>${camera?.active ? `확인됨 · ${escapeHtml(camera.signer_name || "")}` : "서명 동의가 아직 없습니다."}</span></div>
    <div class="rights-state ${research?.active ? "ok" : "warn"}"><strong>연구 전용 지표 동의</strong><span>${research?.active ? `확인됨 · ${escapeHtml(research.signer_name || "")}` : "동의가 없으면 표정·흥미·주의 분석은 실행되지 않습니다."}</span></div>
    <div class="rights-state ${rights.pre_session_ready ? "ok" : "warn"}"><strong>교육 전 확인</strong><span>${rights.pre_session_ready ? `완료 · ${formatDate(rights.pre_session_check?.expires_at, true)}까지` : "적응·설명·중단 연습·현재 의사를 확인해 주세요."}</span></div>
    <div class="rights-state ${rights.child_stop_active ? "stop" : "ok"}"><strong>아동의 현재 선택</strong><span>${escapeHtml(rights.child_stop_message)}</span></div>`;
  $("#child-stop-button").classList.toggle("active-stop", rights.child_stop_active);
  $$("form input, form select, form button, #acclimation-button", $("#rights-panel")).forEach((control) => {
    control.disabled = !recordActive;
  });
  if (recordActive) {
    $("#acclimation-confirmed").disabled = !state.acclimationCompleted;
    $("#acclimation-button").disabled = Boolean(acclimationInterval) || state.acclimationCompleted;
  }
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
  return ({ gaze_diverted: "시선 구역 변화", face_missing: "얼굴 프레임 이탈", posture_shifted: "자세 쏠림", facial_movement_detected: "지정 미세 움직임", temporal_movement_candidate: "새 반복 패턴 후보", repeating_micro_motion: "새 반복 미세 움직임 후보" })[value] || value || "이벤트";
}

function candidateClip(candidateId) {
  return state.clips.find((clip) => clip.trigger_values?.candidate_id === candidateId) || null;
}

function renderPatternMemory() {
  const workspace = $("#pattern-memory-workspace");
  if (!workspace || !state.current) return;
  if (state.current.canonical_status !== "active") {
    workspace.innerHTML = `<article class="card pattern-memory-empty"><strong>동의 철회로 잠김</strong><p>패턴 조회·승격·suppression이 모두 차단되었습니다.</p></article>`;
    return;
  }
  const memory = state.patternMemory;
  if (!memory?.configured) {
    workspace.innerHTML = `<article class="card pattern-memory-empty"><strong>Temporal runtime 대기</strong><p>아직 이 아동의 로컬 pattern memory가 없습니다. checkpoint를 연결한 runtime이 첫 독립 episode를 관찰하면 자동 생성됩니다.</p><code>outputs/ondamm/pattern-memory/${escapeHtml(state.current.child_id)}/</code></article>`;
    return;
  }
  const known = memory.known_patterns || [];
  const candidates = memory.candidates || [];
  const knownMarkup = known.length ? known.map((pattern) => `
    <article class="pattern-card known-pattern">
      <div class="pattern-card-head"><span>승인된 패턴</span><strong>${escapeHtml(pattern.display_name)}</strong></div>
      <p>${escapeHtml(pattern.pattern_id)} · 승인 근거 ${pattern.support_count}회</p>
      <div class="pattern-stats"><span>후속 검출 <b>${pattern.occurrence_count}</b></span><span>거리 임계값 <b>${Number(pattern.distance_threshold).toFixed(3)}</b></span></div>
      <small>prototype ${escapeHtml(pattern.prototype_digest.slice(0, 12))}… · ${escapeHtml(pattern.approved_by)}</small>
    </article>`).join("") : `<div class="empty-inline">승인된 temporal pattern이 아직 없습니다.</div>`;
  const candidateMarkup = candidates.length ? candidates.map((candidate) => {
    const clip = candidateClip(candidate.candidate_id);
    const selectedForReview = clip && state.selectedClipId === clip.clip_id;
    const reviewStatus = selectedForReview ? state.eventReviews?.summary?.status : null;
    let actionMarkup = `<button class="button button-quiet" data-pattern-watch="${escapeHtml(candidate.candidate_id)}" type="button">더 관찰</button>`;
    if (clip) {
      actionMarkup += `<button class="button button-secondary" data-pattern-clip="${escapeHtml(clip.clip_id)}" type="button">${selectedForReview ? "아래 검토 중" : "영상·검토 열기"}</button>`;
    }
    if (selectedForReview && reviewStatus === "consensus_accepted") {
      actionMarkup += `<form class="pattern-promotion-form" data-candidate-id="${escapeHtml(candidate.candidate_id)}" data-clip-id="${escapeHtml(clip.clip_id)}"><input name="display_name" required maxlength="120" placeholder="새 패턴 표시 이름"><input name="approved_by" required maxlength="120" placeholder="별도 승인자"><button class="button button-primary" type="submit">Known pattern으로 등록</button></form>`;
    } else if (selectedForReview && reviewStatus === "consensus_rejected") {
      actionMarkup += `<form class="pattern-suppression-form" data-candidate-id="${escapeHtml(candidate.candidate_id)}" data-clip-id="${escapeHtml(clip.clip_id)}"><input name="approved_by" required maxlength="120" placeholder="승인자"><input name="reason" required maxlength="500" placeholder="suppression 사유"><button class="button button-danger" type="submit">Suppression memory에 등록</button></form>`;
    } else if (clip) {
      actionMarkup += `<p class="pattern-review-hint">3개 역할의 합의 후 별도 승격 또는 suppression을 실행할 수 있습니다.</p>`;
    }
    return `<article class="pattern-card candidate-pattern">
      <div class="pattern-card-head"><span>${candidate.occurrence_count >= memory.policy.strong_candidate_occurrences ? "반복이 뚜렷한 검토 후보" : "아직 이름 붙이지 않은 후보"}</span><strong>${escapeHtml(candidate.candidate_id)}</strong></div>
      <p>반복 ${candidate.occurrence_count} / ${memory.policy.min_occurrences_for_clip} · ${escapeHtml(candidate.review_state)}</p>
      <div class="pattern-stats"><span>평균 길이 <b>${Number(candidate.mean_duration_seconds).toFixed(2)}초</b></span><span>품질 <b>${Math.round(Number(candidate.mean_quality_score) * 100)}%</b></span><span>개인화 threshold <b>${Number(candidate.recommended_distance_threshold ?? memory.policy.known_distance_threshold).toFixed(3)}</b></span></div>
      <small>nearest known ${escapeHtml(candidate.nearest_known_pattern || "없음")} · distance ${Number(candidate.nearest_known_distance).toFixed(3)} · 영상 ${clip ? "1개" : "없음"}</small>
      <div class="pattern-actions">${actionMarkup}</div>
    </article>`;
  }).join("") : `<div class="empty-inline">관찰 중인 unknown candidate가 없습니다.</div>`;
  workspace.innerHTML = `
    <div class="pattern-policy-strip card"><span>시간 흐름 인코더 <b>고정</b></span><span>기준 패턴 갱신 <b>사람만 승인</b></span><span>짧은 영상 기준 <b>${memory.policy.min_occurrences_for_clip}회 반복</b></span><span>1~${memory.policy.min_occurrences_for_clip - 1}회 <b>영상 저장 안 함</b></span></div>
    <div class="pattern-columns"><section><h4>새로운 반복 후보</h4><div class="pattern-grid">${candidateMarkup}</div></section><section><h4>승인된 Known pattern</h4><div class="pattern-grid">${knownMarkup}</div></section></div>`;
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

function reviewRoleName(value) {
  return ({ guardian: "보호자", teacher: "교사", institutional_social_worker: "기관 사회복지사" })[value] || value;
}

function reviewDecisionName(value) {
  return ({ accepted: "의미 있는 움직임", rejected: "이벤트 아님", uncertain: "추가 맥락 필요" })[value] || value;
}

function reviewStatusName(value) {
  return ({
    pending: "검토 대기",
    partially_reviewed: "일부 검토 완료",
    needs_context: "추가 맥락 필요",
    disagreement: "의견 불일치",
    consensus_accepted: "3자 의견 일치 · 후보 유지",
    consensus_rejected: "3자 의견 일치 · 제외 후보",
  })[value] || value || "검토 대기";
}

function crossReviewMarkup() {
  const bundle = state.eventReviews;
  const summary = bundle?.summary || { status: "pending", latest_by_role: {}, pending_roles: ["guardian", "teacher", "institutional_social_worker"] };
  const roles = ["guardian", "teacher", "institutional_social_worker"];
  const roleCards = roles.map((role) => {
    const entry = summary.latest_by_role?.[role];
    return `<article class="role-review ${entry ? "complete" : "pending"}">
      <div class="role-review-head"><strong>${escapeHtml(reviewRoleName(role))}</strong><span>${entry ? escapeHtml(reviewDecisionName(entry.decision)) : "대기"}</span></div>
      ${entry ? `<p>${escapeHtml(entry.observed_facts)}</p>${entry.context_comment ? `<small>${escapeHtml(entry.context_comment)}</small>` : ""}<time>${escapeHtml(entry.reviewer_name)} · ${formatDate(entry.created_at, true)}</time>` : `<p>이 역할의 독립 검토가 아직 없습니다.</p>`}
    </article>`;
  }).join("");
  return `<section class="card cross-review-panel">
    <div class="cross-review-head"><div><p class="eyebrow">세 역할의 독립 교차 검토</p><h4>보호자 · 교사 · 기관 사회복지사 교차 검토</h4><p>각 역할이 먼저 독립 의견을 남깁니다. 일치 여부는 표시만 하며 기록철에는 자동 반영하지 않습니다.</p></div><span class="review-status status-${escapeHtml(summary.status)}">${escapeHtml(reviewStatusName(summary.status))}</span></div>
    <div class="role-review-grid">${roleCards}</div>
    <form id="event-review-form" class="event-review-form">
      <div class="review-form-grid">
        <label>검토 역할<select name="reviewer_role" required><option value="guardian">보호자</option><option value="teacher">교사</option><option value="institutional_social_worker">기관 사회복지사</option></select></label>
        <label>검토자 이름<input name="reviewer_name" required maxlength="120" placeholder="예: 보호자 김OO"></label>
        <label>판단<select name="decision" required><option value="accepted">의미 있는 움직임 후보</option><option value="rejected">이벤트 아님</option><option value="uncertain">추가 맥락 필요</option></select></label>
      </div>
      <label>영상에서 직접 확인한 사실<textarea name="observed_facts" required maxlength="2000" placeholder="감정 해석 없이, 언제 어떤 움직임이 보였는지 적습니다."></textarea></label>
      <label>상황 맥락·코멘트<textarea name="context_comment" maxlength="4000" placeholder="활동, 장소, 직전 사건, 평소 패턴과의 차이 등을 적습니다."></textarea></label>
      <div class="review-submit-row"><p>같은 역할이 다시 제출하면 이전 기록은 보존되고 최신 의견으로 표시됩니다.</p><button class="button button-primary" type="submit" ${state.clipBusy ? "disabled" : ""}>${state.clipBusy === "review" ? "검토 저장 중…" : "독립 검토 저장"}</button></div>
    </form>
  </section>`;
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
  const llm = state.integrations?.llm || { configured: false };
  const localLlm = llm.provider === "ollama" && llm.local_only;
  const structuredTriggerKeys = new Set(["movement_summary", "similarity_to_previous", "selection_explanation"]);
  const triggerMarkup = Object.entries(selected.trigger_values || {}).filter(([key]) => !structuredTriggerKeys.has(key)).map(([key, value]) => `
    <span><b>${escapeHtml(key)}</b>${escapeHtml(value)}</span>`).join("") || `<span><b>trigger</b>메타데이터 없음</span>`;
  const expressionEntries = Object.entries(state.clipAnalysis?.expression_label_counts || {});
  const movementSummary = state.clipAnalysis?.movement_summary || selected.movement_summary;
  const topMovementChanges = movementSummary?.top_changes || [];
  const movementMarkup = movementSummary ? `
    <div class="movement-summary-card">
      <div><span>가장 크게 움직인 부위</span><strong>${escapeHtml(movementSummary.dominant_region_label || "확인 필요")}</strong></div>
      <p>${escapeHtml(movementSummary.plain_summary || "부위별 상대 움직임을 확인했습니다.")}</p>
      ${topMovementChanges.length ? `<ul>${topMovementChanges.slice(0, 3).map((item) => `<li><b>${escapeHtml(item.label)}</b><span>구간 변화 ${Number(item.change_points).toFixed(1)}%p</span></li>`).join("")}</ul>` : ""}
      <small>${escapeHtml(movementSummary.non_diagnostic_notice || "상대 움직임 지표이며 감정이나 의도를 뜻하지 않습니다.")}</small>
    </div>` : `<div class="analysis-placeholder">이전 영상에는 부위별 움직임 요약이 저장되지 않았습니다. 아래 MediaPipe 분석을 실행하면 현재 영상에서 확인할 수 있습니다.</div>`;
  const similarity = selected.similarity_to_previous || {};
  const selectionMarkup = selected.selection_explanation ? `
    <section class="selection-reason-card ${selected.pattern_memory_status === "orphaned" ? "warning" : ""}">
      <div class="selection-reason-head"><span>이벤트 선정 이유</span><strong>${Number(similarity.occurrence_count || selected.trigger_values?.occurrence_count || 0)}회 반복</strong></div>
      <p>${escapeHtml(selected.selection_explanation)}</p>
      <div class="selection-facts">
        ${similarity.embedding_similarity_percent != null ? `<span><b>${Number(similarity.embedding_similarity_percent).toFixed(1)}%</b>시간 흐름 유사도</span>` : ""}
        ${similarity.regional_comparison?.similarity_percent != null ? `<span><b>${Number(similarity.regional_comparison.similarity_percent).toFixed(1)}%</b>움직인 부위 비율 유사도</span>` : ""}
        <span><b>${Number(selected.event_duration_seconds ?? selected.duration_seconds).toFixed(1)}초</b>실제 감지 구간</span>
      </div>
      ${selected.pattern_memory_notice ? `<small>${escapeHtml(selected.pattern_memory_notice)}</small>` : ""}
    </section>` : "";
  const analysisMarkup = state.clipAnalysis ? `
    <div class="analysis-result">
      <div class="analysis-quality ${state.clipAnalysis.review_data_quality === "usable" ? "usable" : "unusable"}">${escapeHtml(state.clipAnalysis.review_data_quality_notice || "로컬 분석 품질을 확인했습니다.")}</div>
      <div class="analysis-summary"><span>대표 움직임 힌트</span><strong>${escapeHtml(expressionHintName(state.clipAnalysis.dominant_expression_hint))}</strong></div>
      <div class="analysis-counts">${expressionEntries.length ? expressionEntries.map(([label, count]) => `<span>${escapeHtml(expressionHintName(label))}<b>${count}</b></span>`).join("") : `<span>분석한 프레임에서 얼굴 blendshape를 확인하지 못했습니다.</span>`}</div>
      ${movementMarkup}
      <p>${escapeHtml(state.clipAnalysis.non_diagnostic_notice || "얼굴 미세 움직임 힌트이며 감정 상태로 확정하지 않습니다.")}</p>
    </div>` : movementMarkup;
  const gptMarkup = state.gptReview ? `
    <div class="gpt-result"><div class="gpt-result-head"><strong>${state.gptReview.local_only ? "Ollama 로컬" : "GPT 원격"} 관찰 보조 초안</strong><span>${escapeHtml(state.gptReview.model || "LLM")}</span></div><div class="gpt-text">${escapeHtml(state.gptReview.review_text).replace(/\n/g, "<br>")}</div><p>전체 영상 전송 안 함 · 로컬 프레임 ${state.gptReview.local_frame_count || 0}장 · 원격 프레임 ${state.gptReview.remote_frame_count || 0}장 · 기록철 자동 반영 안 함</p></div>` : "";

  workspace.innerHTML = `
    <div class="integration-strip card">
      <div><span class="integration-dot ready"></span><strong>MediaPipe</strong><small>로컬 얼굴 blendshape</small></div>
      <div><span class="integration-dot ${llm.configured && llm.available !== false ? "ready" : "off"}"></span><strong>${localLlm ? "Ollama" : "GPT"}</strong><small>${llm.configured ? escapeHtml(llm.model || "연결됨") : "LLM 꺼짐"}</small></div>
      <p>${localLlm ? "Ollama는 추출 프레임을 이 실행 기기의 localhost에서만 처리합니다." : "GPT는 동의한 경우에만 최대 3장 프레임을 원격 전송합니다."}</p>
    </div>
    <div class="clip-review-grid">
      <aside class="card clip-list-panel">
        <div class="clip-panel-head"><span>이벤트 영상</span><b>${state.clips.length}</b></div>
        <div class="clip-list">${state.clips.map((clip) => `
          <button type="button" data-clip-id="${escapeHtml(clip.clip_id)}" class="clip-item ${clip.clip_id === selected.clip_id ? "selected" : ""}">
            <span class="clip-type">${escapeHtml(eventTypeName(clip.event_type))}</span>
            <strong>${escapeHtml(clip.event_id)}</strong>
            <small>영상 ${Number(clip.clip_duration_seconds ?? clip.duration_seconds).toFixed(1)}초 · 움직임 ${Number(clip.event_duration_seconds ?? clip.duration_seconds).toFixed(1)}초<br>${formatDate(clip.created_at, true)}</small>
          </button>`).join("")}</div>
      </aside>
      <article class="card clip-player-panel">
        <div class="clip-panel-head"><div><span>로컬 재생</span><strong>${escapeHtml(eventTypeName(selected.event_type))}</strong></div><span class="local-only-pill">이 기기에서만</span></div>
        <video controls playsinline preload="metadata" src="${escapeHtml(selected.media_url)}" aria-label="선택한 로컬 특이 이벤트 영상"></video>
        ${selectionMarkup}
        <div class="clip-trigger-grid">${triggerMarkup}</div>
        <p class="clip-path">${escapeHtml(selected.relative_path)}</p>
      </article>
      <article class="card clip-analysis-panel">
        <div class="analysis-block">
          <div class="clip-panel-head"><div><span>미세 움직임 힌트</span><strong>Google MediaPipe</strong></div><span class="local-only-pill">로컬</span></div>
          ${analysisMarkup}
          <button class="button button-secondary full-button" id="run-mediapipe-analysis" type="button" ${state.clipBusy || !mediaPipe.configured ? "disabled" : ""}>${state.clipBusy === "mediapipe" ? "로컬 분석 중…" : "MediaPipe로 분석"}</button>
        </div>
        <div class="analysis-block gpt-block">
          <div class="clip-panel-head"><div><span>${localLlm ? "선택적 로컬 검토" : "선택적 원격 검토"}</span><strong>${localLlm ? "Ollama" : "GPT"} 프레임 검토</strong></div><span class="${localLlm ? "local-only-pill" : "remote-pill"}">최대 3장</span></div>
          ${llm.configured && llm.available !== false ? `
            ${localLlm ? `<p class="local-review-note">추출 JPEG는 localhost Ollama에만 전달되며 디스크에 별도 저장하지 않습니다.</p>` : `<label class="consent-check"><input type="checkbox" id="gpt-frame-consent"><span>선택 영상에서 추출한 프레임을 OpenAI API로 전송하는 데 동의합니다. 전체 MP4는 전송하지 않습니다.</span></label>`}
            <button class="button button-quiet full-button" id="run-llm-review" type="button" ${state.clipBusy ? "disabled" : ""}>${state.clipBusy === "gpt" ? "LLM 검토 중…" : (localLlm ? "Ollama로 로컬 검토" : "동의 후 GPT 검토")}</button>` : `
            <div class="key-missing"><strong>LLM 연결 대기</strong><p>${localLlm ? "Ollama daemon과 모델 상태를 확인하세요." : "ONDAMM_LLM_PROVIDER와 provider 설정을 확인하세요."}</p></div>`}
          ${gptMarkup}
        </div>
      </article>
    </div>
    ${crossReviewMarkup()}`;
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
      <div class="result-head"><div><p class="eyebrow">검토용 초안 · 아직 저장 안 됨</p><h3>${escapeHtml(draft.goal)}</h3></div><span class="draft-banner">승인 전 초안</span></div>
      <p class="summary">${escapeHtml(draft.summary)}</p>
      <h4>제안 활동</h4><ol>${draft.suggested_activities.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ol>
      <h4>근거</h4><ul>${draft.rationale_lines.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
      <div class="approval-box"><label>최종 승인자<input id="recommendation-approver" value="${escapeHtml(state.recommendationPayload.drafted_by)}"></label><button class="button button-primary" id="approve-recommendation" type="button">검토 후 승인 저장</button></div>
    </div>`;
}

function renderLocalRag() {
  const status = $("#local-rag-status");
  const result = $("#local-rag-result");
  const form = $("#local-rag-form");
  if (!status || !result || !form) return;
  const llm = state.integrations?.llm || { configured: false };
  const ready = llm.configured && llm.available !== false && llm.provider === "ollama" && llm.rag_configured;
  status.textContent = ready ? `${llm.model || "Ollama"} · ${llm.context_length ? `${Math.round(llm.context_length / 1024)}K · ` : ""}로컬` : "Ollama RAG 꺼짐";
  form.querySelector("button").disabled = !ready || state.ragBusy || state.current?.canonical_status !== "active";
  form.querySelector("button").textContent = state.ragBusy ? "로컬 검색 중…" : "로컬에서 검색·대화";
  if (!state.ragResult && !state.ragHistory.length) {
    result.innerHTML = ready
      ? `<p>질문을 입력하면 답, source ID와 일치하는 영상 링크를 표시합니다. 대화와 vector는 영구 저장되지 않습니다.</p>`
      : `<p><code>ONDAMM_LLM_PROVIDER=ollama</code>와 로컬 Ollama 모델을 준비한 뒤 웹 서버를 다시 시작하세요.</p>`;
    return;
  }
  const latest = state.ragResult || { sources: [], video_results: [] };
  const sources = latest.sources || [];
  const videos = latest.video_results || [];
  const conversation = state.ragHistory.map((message) => `<article class="rag-message ${message.role}"><span>${message.role === "user" ? "질문" : "로컬 보조 답변"}</span><p>${escapeHtml(message.content).replace(/\n/g, "<br>")}</p></article>`).join("");
  result.innerHTML = `<div class="rag-conversation">${conversation}</div>
    <p class="rag-local-notice">로컬 처리 · 대화/vector 영구 저장 안 함 · 기록철 자동 반영 안 함</p>
    ${videos.length ? `<div class="rag-video-results"><strong>찾은 이벤트 영상</strong>${videos.map((video) => `<article><div><span>${escapeHtml(eventTypeName(video.event_type))}</span><strong>${escapeHtml(formatDate(video.created_at, true))}</strong><small>${Number(video.duration_seconds).toFixed(1)}초 · ${escapeHtml(video.source_id)}</small></div><button class="button button-quiet" type="button" data-rag-clip-id="${escapeHtml(video.clip_id)}">기존 플레이어에서 열기</button></article>`).join("")}</div>` : ""}
    <div class="rag-sources"><strong>검색 근거</strong>${sources.length ? sources.map((source) => `<article><code>${escapeHtml(source.source_id)}</code><span>${source.source_kind === "video" ? "영상 메타데이터 · 미검토" : "승인 기록"}</span><p>${escapeHtml(source.excerpt || "")}</p></article>`).join("") : `<p>사용할 로컬 근거가 없습니다.</p>`}</div>`;
}

async function refreshCurrent() {
  if (!state.current) return;
  await loadDossiers(state.current.child_id);
}

async function reloadAfterSuccessfulMutation(preferredId = null, tab = null) {
  try {
    await loadDossiers(preferredId);
    if (tab) setTab(tab);
  } catch (error) {
    console.error("Saved mutation could not be refreshed", error);
    toast("변경 사항은 저장됐지만 화면 갱신에 실패했습니다. 페이지를 새로고침해 주세요.", "error");
  }
}

async function refreshAfterSuccessfulMutation(tab = null) {
  if (!state.current) return;
  await reloadAfterSuccessfulMutation(state.current.child_id, tab);
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
  const form = event.currentTarget;
  const data = new FormData(form);
  const body = {
    child_id: data.get("child_id").trim(), display_name: data.get("display_name").trim(),
    age_band: data.get("age_band").trim(), communication_modality: data.get("communication_modality").trim(),
    confirmed_preferences: splitList(data.get("confirmed_preferences")),
    effective_strategies: splitList(data.get("effective_strategies")),
    triggers_and_calming_supports: splitList(data.get("triggers_and_calming_supports")),
  };
  let created;
  try {
    created = await api("/api/dossiers", { method: "POST", body });
  } catch (error) {
    toast(error.message, "error");
    return;
  }
  $("#create-dialog").close(); form.reset();
  toast(`${created.display_name}님의 로컬 지원 기록철을 만들었습니다.`);
  await reloadAfterSuccessfulMutation(created.child_id);
});

$("#session-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const data = new FormData(form);
  const body = {
    title: data.get("title").trim(), activity_name: data.get("activity_name").trim(),
    observed_response: data.get("observed_response").trim(), educator_interpretation: data.get("educator_interpretation").trim(),
    approved_by: data.get("approved_by").trim(), tags: splitList(data.get("tags")),
  };
  try {
    await api(`/api/dossiers/${encodeURIComponent(state.current.child_id)}/sessions`, { method: "POST", body });
  } catch (error) {
    toast(error.message, "error");
    return;
  }
  $("#session-dialog").close(); form.reset();
  form.elements.approved_by.value = "local-operator";
  toast("승인된 수업 기록을 저장했습니다.");
  await refreshAfterSuccessfulMutation("sessions");
});

$("#new-facial-profile-button").addEventListener("click", () => {
  if (!state.current || state.current.canonical_status !== "active") return;
  $("#facial-profile-dialog").showModal();
});

$("#facial-profile-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const data = new FormData(form);
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
  } catch (error) {
    toast(error.message, "error");
    return;
  }
  $("#facial-profile-dialog").close(); form.reset();
  form.elements.activation_threshold.value = "0.35";
  form.elements.approved_by.value = "local-operator";
  toast("승인된 얼굴 움직임 프로필을 기록철에 저장했습니다.");
  await refreshAfterSuccessfulMutation("dossier");
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
  } catch (error) {
    toast(error.message, "error");
    return;
  }
  toast("검토한 활동 계획을 승인 기록으로 저장했습니다.");
  await refreshAfterSuccessfulMutation("plan");
});

$("#local-rag-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const formData = new FormData(form);
  const question = formData.get("question").trim();
  const scope = formData.get("scope");
  if (!question) return toast("검색하거나 물어볼 내용을 입력해 주세요.", "error");
  state.ragBusy = true;
  state.ragResult = null;
  renderLocalRag();
  try {
    state.ragResult = await api(`/api/dossiers/${encodeURIComponent(state.current.child_id)}/assistant/query`, {
      method: "POST", body: {
        question,
        scope,
        top_k: 8,
        history: state.ragHistory.slice(-6).map((message) => ({
          role: message.role,
          content: message.content.slice(0, 1800),
        })),
      },
    });
    state.ragHistory.push(
      { role: "user", content: question },
      { role: "assistant", content: state.ragResult.answer },
    );
    state.ragHistory = state.ragHistory.slice(-8);
    form.elements.question.value = "";
    toast(`로컬 자료에서 답변과 영상 ${state.ragResult.video_results?.length || 0}개를 찾았습니다.`);
  } catch (error) {
    toast(error.message, "error");
  } finally {
    state.ragBusy = false;
    renderLocalRag();
  }
});

$("#local-rag-reset").addEventListener("click", () => {
  state.ragHistory = [];
  state.ragResult = null;
  renderLocalRag();
  toast("브라우저 메모리의 로컬 대화 내용을 지웠습니다.");
});

$("#local-rag-result").addEventListener("click", async (event) => {
  const button = event.target.closest("[data-rag-clip-id]");
  if (!button) return;
  const clipId = button.dataset.ragClipId;
  if (!state.clips.some((clip) => clip.clip_id === clipId)) {
    await loadLocalClips({ preserveSelection: true });
  }
  if (!state.clips.some((clip) => clip.clip_id === clipId)) {
    return toast("해당 영상이 현재 로컬 catalog에 없습니다.", "error");
  }
  state.selectedClipId = clipId;
  state.clipAnalysis = null;
  state.gptReview = null;
  await loadEventReviews({ silent: true });
  setTab("observation");
  renderClipReview();
  renderPatternMemory();
  document.querySelector("#clip-review-title")?.scrollIntoView({ behavior: "smooth", block: "start" });
  toast("찾은 영상을 기존 검토 플레이어에서 열었습니다.");
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
  await loadEventReviews({ silent: true });
  await loadPatternMemory({ silent: true });
  state.clipAnalysis = null;
  state.gptReview = null;
  renderClipReview();
  renderPatternMemory();
  toast(`${state.clips.length}개의 로컬 이벤트 영상을 확인했습니다.`);
});

$("#pattern-refresh-button").addEventListener("click", async () => {
  await loadLocalClips({ preserveSelection: true, silent: true });
  await loadEventReviews({ silent: true });
  await loadPatternMemory();
  renderPatternMemory();
  renderClipReview();
  toast("개인별 temporal pattern memory를 새로고침했습니다.");
});

$("#pattern-memory-workspace").addEventListener("click", async (event) => {
  const clipButton = event.target.closest("[data-pattern-clip]");
  if (clipButton) {
    state.selectedClipId = clipButton.dataset.patternClip;
    state.clipAnalysis = null;
    state.gptReview = null;
    await loadEventReviews();
    renderPatternMemory();
    renderClipReview();
    document.querySelector("#clip-review-title")?.scrollIntoView({ behavior: "smooth", block: "start" });
    return;
  }
  const watchButton = event.target.closest("[data-pattern-watch]");
  if (watchButton) {
    try {
      await api(`/api/dossiers/${encodeURIComponent(state.current.child_id)}/patterns/candidates/${encodeURIComponent(watchButton.dataset.patternWatch)}/watch`, { method: "POST", body: {} });
    } catch (error) {
      toast(error.message, "error");
      return;
    }
    toast("후보를 영상 추가 저장 없이 계속 관찰합니다.");
    await refreshAfterSuccessfulMutation("observation");
    renderPatternMemory();
  }
});

$("#pattern-memory-workspace").addEventListener("submit", async (event) => {
  const promotion = event.target.closest(".pattern-promotion-form");
  const suppression = event.target.closest(".pattern-suppression-form");
  if (!promotion && !suppression) return;
  event.preventDefault();
  const form = promotion || suppression;
  const data = new FormData(form);
  const action = promotion ? "promote" : "suppress";
  const body = promotion ? {
    clip_id: form.dataset.clipId,
    display_name: data.get("display_name").trim(),
    approved_by: data.get("approved_by").trim(),
  } : {
    clip_id: form.dataset.clipId,
    approved_by: data.get("approved_by").trim(),
    reason: data.get("reason").trim(),
  };
  state.patternBusy = action;
  try {
    await api(`/api/dossiers/${encodeURIComponent(state.current.child_id)}/patterns/candidates/${encodeURIComponent(form.dataset.candidateId)}/${action}`, { method: "POST", body });
  } catch (error) {
    toast(error.message, "error");
    state.patternBusy = null;
    return;
  }
  toast(promotion ? "승인된 Known pattern을 등록했습니다. TCN은 재학습하지 않았습니다." : "거절 패턴을 suppression memory에 등록했습니다.");
  await refreshAfterSuccessfulMutation("observation");
  state.patternBusy = null;
  renderPatternMemory();
});

$("#clip-review-workspace").addEventListener("click", async (event) => {
  const clipButton = event.target.closest("[data-clip-id]");
  if (clipButton) {
    state.selectedClipId = clipButton.dataset.clipId;
    state.clipAnalysis = null;
    state.gptReview = null;
    await loadEventReviews({ silent: true });
    renderClipReview();
    renderPatternMemory();
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
  if (event.target.closest("#run-llm-review")) {
    const llm = state.integrations?.llm || {};
    const consent = $("#gpt-frame-consent")?.checked === true;
    if (llm.requires_explicit_frame_consent && !consent) return toast("추출 프레임 원격 전송 동의를 먼저 확인해 주세요.", "error");
    state.clipBusy = "gpt";
    renderClipReview();
    try {
      state.gptReview = await api(`/api/dossiers/${encodeURIComponent(state.current.child_id)}/clips/${encodeURIComponent(state.selectedClipId)}/llm-review`, {
        method: "POST", body: { confirm_remote_frame_upload: consent },
      });
      toast(`${state.gptReview.local_only ? "Ollama 로컬" : "GPT 원격"} 관찰 보조 초안을 받았습니다. 공식 기록에는 반영되지 않았습니다.`);
    } catch (error) {
      toast(error.message, "error");
    } finally {
      state.clipBusy = null;
      renderClipReview();
    }
  }
});

$("#clip-review-workspace").addEventListener("submit", async (event) => {
  if (event.target.id !== "event-review-form") return;
  event.preventDefault();
  const data = new FormData(event.target);
  state.clipBusy = "review";
  renderClipReview();
  try {
    state.eventReviews = await api(`/api/dossiers/${encodeURIComponent(state.current.child_id)}/clips/${encodeURIComponent(state.selectedClipId)}/reviews`, {
      method: "POST",
      body: {
        reviewer_role: data.get("reviewer_role"),
        reviewer_name: data.get("reviewer_name").trim(),
        decision: data.get("decision"),
        observed_facts: data.get("observed_facts").trim(),
        context_comment: data.get("context_comment").trim(),
      },
    });
    renderPatternMemory();
    toast("독립 검토를 저장했습니다. 기록철에는 자동 반영되지 않았습니다.");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    state.clipBusy = null;
    renderClipReview();
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
      <div class="result-head"><div><p class="eyebrow">검토할 관찰 메모</p><h3>관찰 메모 초안</h3></div><span class="draft-stamp">초안</span></div>
      <div class="sensing-metrics"><div><span>프레임</span><strong>${result.frame_count}</strong></div><div><span>얼굴 존재</span><strong>${Math.round(result.face_present_ratio * 100)}%</strong></div><div><span>자세 추정</span><strong>${Math.round(result.pose_present_ratio * 100)}%</strong></div><div><span>미세 움직임</span><strong>${escapeHtml(expressionHintName(expressionHint))}</strong></div></div>
      <div class="draft-lines">${result.reviewed_note_draft.map((line) => `<p>${escapeHtml(line)}</p>`).join("")}</div>
      <div class="non-authoritative"><strong>비권위 초안</strong><br>${escapeHtml(result.non_authoritative_notice)}</div>`;
    toast("데모 관찰 초안을 만들었습니다. 공식 기록에는 반영되지 않았습니다.");
  } catch (error) { toast(error.message, "error"); }
});

$("#export-button").addEventListener("click", async () => {
  const actorId = $("#export-actor").value.trim();
  if (!actorId) return toast("담당자를 입력해 주세요.", "error");
  let result;
  try {
    result = await api(`/api/dossiers/${encodeURIComponent(state.current.child_id)}/handoff/export`, { method: "POST", body: { actor_id: actorId } });
  } catch (error) {
    toast(error.message, "error");
    return;
  }
  $("#export-result").innerHTML = `<div class="export-success"><strong>생성 완료 · ${escapeHtml(result.manifest.artifact_id)}</strong>문서: ${escapeHtml(result.export_path)}<br>서명: ${escapeHtml(result.manifest_path)}</div>`;
  toast("서명된 인수인계 자료를 만들었습니다.");
  await refreshAfterSuccessfulMutation("handoff");
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
    : `<div class="status-restore"><strong>기록철 활성 복구</strong><br>승인된 근거가 있을 때만 다시 활성화합니다. 이전 동의는 되살아나지 않으며, 새 목적별 동의와 교육 전 확인이 필요합니다.</div>`;
  $("#status-submit").textContent = active ? "동의 철회로 잠금" : "승인 근거로 활성 복구";
  $("#status-submit").className = `button ${active ? "button-danger" : "button-primary"}`;
  $("#purge-section").hidden = active;
  $("#purge-preview").innerHTML = "";
  $("#status-dialog").showModal();
});

$("#status-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = new FormData(event.currentTarget);
  try {
    await api(`/api/dossiers/${encodeURIComponent(state.current.child_id)}/status`, {
      method: "POST", body: Object.fromEntries(data.entries()),
    });
  } catch (error) {
    toast(error.message, "error");
    return;
  }
  $("#status-dialog").close();
  toast(data.get("status") === "active" ? "기록철을 활성화했습니다. 촬영 전 새 동의와 권리 확인이 필요합니다." : "동의를 철회하고 모든 촬영·분석을 잠갔습니다.");
  await refreshAfterSuccessfulMutation();
});

$("#purge-preview-button").addEventListener("click", async () => {
  if (!state.current) return;
  try {
    const preview = await api(`/api/dossiers/${encodeURIComponent(state.current.child_id)}/purge/preview`, { method: "POST", body: {} });
    $("#purge-preview").innerHTML = `
      <div class="purge-warning"><strong>${escapeHtml(preview.title)}</strong><p>${escapeHtml(preview.warning)}</p>
      <ul>${preview.targets.map((item) => `<li><strong>${escapeHtml(item.category_label)}</strong><small>${escapeHtml(item.path)}</small></li>`).join("") || "<li>현재 확인된 삭제 대상이 없습니다.</li>"}</ul>
      <label>삭제 실행 담당자<input id="purge-actor" value="guardian-admin"></label>
      <label>확인 문구<input id="purge-confirmation" placeholder="${escapeHtml(preview.confirmation_phrase)}"></label>
      <button class="button button-danger full-button" id="purge-execute-button" type="button">복구할 수 없는 실제 삭제 실행</button></div>`;
  } catch (error) { toast(error.message, "error"); }
});

$("#purge-preview").addEventListener("click", async (event) => {
  if (!event.target.closest("#purge-execute-button") || !state.current) return;
  const childId = state.current.child_id;
  let result;
  try {
    result = await api(`/api/dossiers/${encodeURIComponent(childId)}/purge/execute`, {
      method: "POST", body: { actor_id: $("#purge-actor").value.trim(), confirmation: $("#purge-confirmation").value },
    });
  } catch (error) {
    toast(error.message, "error");
    return;
  }
  $("#status-dialog").close();
  state.current = null;
  toast(`${result.message} 익명화된 삭제 확인서를 남겼습니다.`);
  await reloadAfterSuccessfulMutation();
});

$("#child-stop-button").addEventListener("click", async () => {
  if (!state.current) return;
  try {
    state.rights = await api(`/api/dossiers/${encodeURIComponent(state.current.child_id)}/rights/child-stop`, {
      method: "POST", body: { reason: "아동이 웹의 ‘촬영 싫어요·중단’ 버튼을 누름" },
    });
    state.current.subject_refusal_active = true;
    renderRights();
    toast("촬영 중단 요청을 보냈습니다. 실행 중인 카메라는 곧 멈춥니다.", "error");
  } catch (error) { toast(error.message, "error"); }
});

$("#camera-consent-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.current) return;
  const form = event.currentTarget;
  const data = new FormData(form);
  const body = Object.fromEntries(data.entries());
  body.guardian_consent_confirmed = data.get("guardian_consent_confirmed") === "on";
  body.subject_assent_confirmed = data.get("subject_assent_confirmed") === "on";
  try {
    state.rights = await api(`/api/dossiers/${encodeURIComponent(state.current.child_id)}/rights/consents`, { method: "POST", body });
    form.reset();
    form.elements.form_version.value = "1.0";
    renderRights();
    toast("선택한 목적의 서명 동의를 이 기기에 등록했습니다.");
  } catch (error) { toast(error.message, "error"); }
});

let acclimationInterval = null;
$("#acclimation-button").addEventListener("click", () => {
  if (acclimationInterval) return;
  let remaining = 180;
  state.acclimationCompleted = false;
  $("#acclimation-button").disabled = true;
  $("#acclimation-message").textContent = "카메라를 켜지 말고, 촬영 장치를 보여 주며 편안하게 적응해 주세요.";
  acclimationInterval = window.setInterval(() => {
    remaining -= 1;
    $("#acclimation-timer").textContent = `${String(Math.floor(remaining / 60)).padStart(2, "0")}:${String(remaining % 60).padStart(2, "0")}`;
    if (remaining <= 0) {
      window.clearInterval(acclimationInterval);
      acclimationInterval = null;
      state.acclimationCompleted = true;
      $("#acclimation-confirmed").disabled = false;
      $("#acclimation-confirmed").checked = true;
      $("#acclimation-message").textContent = "적응 시간이 끝났습니다. 아동이 편안한지 다시 묻고 나머지 항목을 확인해 주세요.";
      $("#acclimation-button").textContent = "카메라를 끈 적응 완료";
      toast("카메라를 끈 적응 시간이 끝났습니다.");
    }
  }, 1000);
});

$("#pre-session-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.current) return;
  if (!state.acclimationCompleted) return toast("먼저 카메라를 끈 3분 적응 시간을 마쳐 주세요.", "error");
  const data = new FormData(event.currentTarget);
  const body = {
    operator_id: data.get("operator_id").trim(),
    guardian_cross_checker: data.get("guardian_cross_checker").trim(),
    educator_cross_checker: data.get("educator_cross_checker").trim(),
    valid_minutes: Number(data.get("valid_minutes")),
  };
  ["explanation_confirmed", "recording_device_recognized", "camera_off_acclimation_completed", "stop_control_practiced", "subject_willing_now"].forEach((key) => {
    body[key] = data.get(key) === "on";
  });
  try {
    state.rights = await api(`/api/dossiers/${encodeURIComponent(state.current.child_id)}/rights/pre-session`, { method: "POST", body });
  } catch (error) {
    toast(error.message, "error");
    return;
  }
  renderRights();
  toast("교육 전 권리 확인을 마쳤습니다. 유효시간 동안만 촬영을 시작할 수 있습니다.");
  await refreshAfterSuccessfulMutation("observation");
});

loadDossiers().catch((error) => {
  console.error(error);
  toast(`앱을 불러오지 못했습니다: ${error.message}`, "error");
});
