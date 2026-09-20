const $ = (selector) => document.querySelector(selector);

const ui = {
  newChat: $("#new-chat"), newProject: $("#new-project"), newProjectForm: $("#new-project-form"), newProjectName: $("#new-project-name"),
  newProjectCancel: $("#new-project-cancel"), conversationList: $("#conversation-list"), sidebarProjectStatus: $("#sidebar-project-status"), importButton: $("#import-button"), importFile: $("#import-file"),
  profileSelect: $("#profile-select"), profileEdit: $("#profile-edit"), profileCreate: $("#profile-create"), profileDialog: $("#profile-dialog"),
  profileForm: $("#profile-form"), profileEditId: $("#profile-edit-id"), profileName: $("#profile-name"), profileLanguage: $("#profile-language"),
  profileTechnicalLevel: $("#profile-technical-level"), profileDetail: $("#profile-detail"), profileTone: $("#profile-tone"), profileStructure: $("#profile-structure"),
  profilePreferredFormats: $("#profile-preferred-formats"), profileAvoidFormats: $("#profile-avoid-formats"), profileInstructions: $("#profile-instructions"), profileSave: $("#profile-save"), profileError: $("#profile-error"),
  chatTitle: $("#chat-title"), subtitle: $("#chat-subtitle"), exportButton: $("#export-button"), deleteChat: $("#delete-chat"),
  messages: $("#messages"), form: $("#message-form"), input: $("#message-input"), send: $("#send-button"), error: $("#form-error"),
  voiceButton: $("#voice-button"), voiceStatus: $("#voice-status"),
  presetSelect: $("#preset-select"), settingsToggle: $("#settings-toggle"), settingsPanel: $("#settings-panel"), settingsClose: $("#settings-close"),
  settingsSummary: $("#settings-summary"), resetCustom: $("#reset-custom"), saveAsPreset: $("#save-as-preset"), presetsButton: $("#presets-button"),
  requestStatus: $("#request-status"), tokenOverview: $("#token-overview"),
  contextMode: $("#context-mode"), contextStatus: $("#context-status"), contextMetrics: $("#context-metrics"),
  taskDescription: $("#task-description"), taskStage: $("#task-stage"), taskCurrentStep: $("#task-current-step"),
  taskTransitionMode: $("#task-transition-mode"), taskAutopilotStop: $("#task-autopilot-stop"),
  taskExpectedAction: $("#task-expected-action"), taskPlan: $("#task-plan"), taskStateSave: $("#task-state-save"),
  taskActivityToggle: $("#task-activity-toggle"), taskActivityBadge: $("#task-activity-badge"), taskStateNote: $("#task-state-note"),
  taskStateShow: $("#task-state-show"), taskStateReport: $("#task-state-report"), taskTransitionActions: $("#task-transition-actions"),
  taskHistoryCount: $("#task-history-count"), taskHistoryList: $("#task-history-list"),
  taskHandoffCount: $("#task-handoff-count"), taskHandoffNote: $("#task-handoff-note"), taskHandoffList: $("#task-handoff-list"),
  invariantScope: $("#invariant-scope"), invariantSetId: $("#invariant-set-id"), invariantName: $("#invariant-name"),
  invariantRules: $("#invariant-rules"), invariantEnabled: $("#invariant-enabled"), invariantSave: $("#invariant-save"),
  invariantCancel: $("#invariant-cancel"), invariantError: $("#invariant-error"), invariantLayers: $("#invariant-layers"), invariantsCount: $("#invariants-count"),
  summaryCurrent: $("#summary-current"), summaryCurrentMeta: $("#summary-current-meta"), summaryHistory: $("#summary-history"), summaryCount: $("#summary-count"),
  summarySection: $("#summary-section"), summaryHistorySection: $("#summary-history-section"),
  windowSetting: $("#window-setting"), windowExchanges: $("#window-exchanges"), contextRuleText: $("#context-rule-text"),
  factsSection: $("#facts-section"), factsList: $("#facts-list"), factsCount: $("#facts-count"), factForm: $("#fact-form"),
  factKey: $("#fact-key"), factValue: $("#fact-value"), factsError: $("#facts-error"),
  factsHelpButton: $("#facts-help-button"), factsHelpDialog: $("#facts-help-dialog"),
  branchingSection: $("#branching-section"), branchCount: $("#branch-count"),
  presetDialog: $("#preset-dialog"), presetList: $("#preset-list"), presetName: $("#preset-name"), presetDescription: $("#preset-description"),
  presetReasoning: $("#preset-reasoning"), presetEditId: $("#preset-edit-id"), presetSave: $("#preset-save"), presetCancelEdit: $("#preset-cancel-edit"), presetError: $("#preset-error"),
  model: $("#setting-model"), system: $("#setting-system"), temperature: $("#setting-temperature"), topP: $("#setting-top-p"),
  reasoning: $("#setting-reasoning"), effort: $("#setting-effort"), maxTokens: $("#setting-max-tokens"), stop: $("#setting-stop"),
  format: $("#setting-format"), logprobs: $("#setting-logprobs"), topLogprobs: $("#setting-top-logprobs"), template: $("#message-template"),
  memoryButton: $("#memory-button"), memoryDialog: $("#memory-dialog"), memoryClose: $("#memory-close"), memoryError: $("#memory-error"),
  memoryConversationSummary: $("#memory-conversation-summary"), memoryProjectSelect: $("#memory-project-select"), memoryNewProject: $("#memory-new-project"),
  memoryProjectEditor: $("#memory-project-editor"), memoryProjectName: $("#memory-project-name"), memoryProjectDescription: $("#memory-project-description"),
  memoryProjectAuto: $("#memory-project-auto"), memorySaveProject: $("#memory-save-project"), memoryProjectConversations: $("#memory-project-conversations"),
  memoryProjectForm: $("#memory-project-form"), memoryProjectList: $("#memory-project-list"), memoryUserAuto: $("#memory-user-auto"),
  memoryUserForm: $("#memory-user-form"), memoryUserList: $("#memory-user-list"), memoryPolicyList: $("#memory-policy-list"),
  memorySnapshotDialog: $("#memory-snapshot-dialog"), memorySnapshotClose: $("#memory-snapshot-close"), memorySnapshotContent: $("#memory-snapshot-content"),
  projectShareDialog: $("#project-share-dialog"), projectShareClose: $("#project-share-close"), projectShareOwner: $("#project-share-owner"),
  projectShareList: $("#project-share-list"), projectShareSave: $("#project-share-save"), projectShareError: $("#project-share-error"),
};

let appState = { conversations: [], projects: [], profiles: [], activeProfileId: null, presets: [], defaults: null, provider: null };
let activeConversation = null;
let sharingProjectId = null;
let selectedPresetId = "";
let sending = false;
let autopilotRunning = false;
let autopilotStopRequested = false;
const MAX_AUTOPILOT_TRANSITIONS = 6;
let voiceState = { phase: "starting", ready: false, message: "Whisper запускается…" };
let mediaRecorder = null;
let microphoneStream = null;
let audioChunks = [];
let recordingTimer = null;
let voiceBusy = false;
let voiceSubmitAfterTranscription = false;
let invariantContext = { bundle: { layers: {}, rules: [] }, sets: { user: [], project: [], task: [] } };

start().catch(showFatal);

async function start() {
  const data = await api("/api/state");
  appState = { conversations: data.conversations, projects: data.projects || [], profiles: data.profiles || [], activeProfileId: data.active_profile_id, presets: data.presets, defaults: data.default_settings, provider: data.provider };
  renderProfileSelect();
  populateModels();
  renderPresetSelect();
  applySettings(appState.defaults);
  renderConversationList();
  if (appState.conversations.length) await openConversation(appState.conversations[0].id);
  else renderEmptyWorkspace();
  startVoiceStatusPolling();
}

function startVoiceStatusPolling() {
  refreshVoiceStatus().catch(() => {});
  window.setInterval(() => refreshVoiceStatus().catch(() => {}), 3000);
}

async function refreshVoiceStatus() {
  const data = await api("/api/voice/status");
  voiceState = data.voice;
  renderVoiceStatus();
}

function renderVoiceStatus() {
  if (mediaRecorder?.state === "recording" || voiceBusy) return;
  ui.voiceStatus.textContent = voiceState.message || "Whisper недоступен.";
  ui.voiceStatus.className = `voice-status ${voiceState.phase || "error"}`;
  ui.voiceButton.disabled = sending || taskIsPaused() || !voiceState.ready;
  ui.voiceButton.classList.remove("recording", "processing");
  ui.voiceButton.title = voiceState.ready ? "Надиктовать сообщение" : ui.voiceStatus.textContent;
}

function populateModels() {
  ui.model.replaceChildren();
  appState.provider.models.forEach((model) => ui.model.add(new Option(model, model)));
}

async function createConversation(projectId = null) {
  const data = await api("/api/conversations", { method: "POST", body: { title: "Новый диалог", project_id: projectId } });
  await refreshState();
  await openConversation(data.conversation.id);
  ui.input.focus();
}

async function refreshState() {
  const data = await api("/api/state");
  appState.conversations = data.conversations;
  appState.projects = data.projects || [];
  appState.profiles = data.profiles || [];
  appState.activeProfileId = data.active_profile_id;
  appState.presets = data.presets;
  renderConversationList();
  renderProfileSelect();
  renderPresetSelect();
}

function profileById(id) { return appState.profiles.find((item) => item.id === id); }

function renderProfileSelect() {
  ui.profileSelect.replaceChildren();
  appState.profiles.forEach((profile) => ui.profileSelect.add(new Option(profile.name, profile.id)));
  ui.profileSelect.value = appState.activeProfileId || "";
}

function renderEmptyWorkspace() {
  activeConversation = null;
  ui.chatTitle.value = "Нет выбранного диалога";
  ui.messages.replaceChildren();
  const welcome = document.createElement("div"); welcome.className = "welcome";
  welcome.innerHTML = '<span class="welcome-mark">D</span><h1>У этого профиля пока нет чатов</h1><p>Создайте диалог или проект. Расшаренные владельцами проекты также появятся здесь.</p>';
  ui.messages.append(welcome);
  ui.form.querySelectorAll("textarea, button").forEach((item) => { item.disabled = true; });
  ui.deleteChat.disabled = true; ui.exportButton.disabled = true; ui.memoryButton.disabled = true;
  setTaskStateControlsDisabled(true);
  renderConversationList();
}

function enableConversationControls() {
  ui.form.querySelectorAll("textarea, button").forEach((item) => { item.disabled = false; });
  ui.deleteChat.disabled = false; ui.exportButton.disabled = false; ui.memoryButton.disabled = false;
  setTaskStateControlsDisabled(false);
  renderVoiceStatus();
}

async function changeActiveProfile() {
  await api("/api/profiles/active", { method: "PATCH", body: { profile_id: ui.profileSelect.value } });
  activeConversation = null;
  await refreshState();
  if (appState.conversations.length) await openConversation(appState.conversations[0].id);
  else renderEmptyWorkspace();
}

function openProfileDialog(profile = null) {
  const preferences = profile?.preferences || {};
  ui.profileEditId.value = profile?.id || "";
  ui.profileName.value = profile?.name || "";
  ui.profileLanguage.value = preferences.language || "ru";
  ui.profileTechnicalLevel.value = preferences.technical_level || "intermediate";
  ui.profileDetail.value = preferences.detail_level || "medium";
  ui.profileTone.value = preferences.tone || "neutral";
  ui.profileStructure.value = preferences.response_structure || "result_then_explanation";
  ui.profilePreferredFormats.value = (preferences.preferred_formats || []).join(", ");
  ui.profileAvoidFormats.value = (preferences.avoid_formats || []).join(", ");
  ui.profileInstructions.value = profile?.custom_instructions || "";
  ui.profileError.textContent = "";
  $("#profile-dialog-title").textContent = profile ? "Изменить профиль" : "Новый профиль";
  ui.profileDialog.showModal();
  ui.profileName.focus();
}

function profileFormBody() {
  const split = (value) => value.split(",").map((item) => item.trim()).filter(Boolean);
  return {
    name: ui.profileName.value,
    preferences: {
      language: ui.profileLanguage.value,
      technical_level: ui.profileTechnicalLevel.value,
      detail_level: ui.profileDetail.value,
      tone: ui.profileTone.value,
      response_structure: ui.profileStructure.value,
      preferred_formats: split(ui.profilePreferredFormats.value),
      avoid_formats: split(ui.profileAvoidFormats.value),
    },
    custom_instructions: ui.profileInstructions.value,
  };
}

async function saveProfile() {
  const id = ui.profileEditId.value;
  try {
    const data = await api(id ? `/api/profiles/${id}` : "/api/profiles", { method: id ? "PATCH" : "POST", body: profileFormBody() });
    if (!id) await api("/api/profiles/active", { method: "PATCH", body: { profile_id: data.profile.id } });
    ui.profileDialog.close(); activeConversation = null; await refreshState();
    if (appState.conversations.length) await openConversation(appState.conversations[0].id); else renderEmptyWorkspace();
  } catch (error) { ui.profileError.textContent = error.message; }
}

async function openConversation(id) {
  const data = await api(`/api/conversations/${id}`);
  activeConversation = data.conversation;
  enableConversationControls();
  ui.chatTitle.value = activeConversation.title;
  renderMessages();
  await loadInvariantContext();
  renderConversationList();
  const lastUser = [...conversationMessages()].reverse().find((item) => item.role === "user");
  if (lastUser?.technical?.settings) {
    applySettings(lastUser.technical.settings);
    selectedPresetId = lastUser.technical.configuration_source?.preset_id || "";
    if (!appState.presets.some((item) => item.id === selectedPresetId)) selectedPresetId = "";
    ui.presetSelect.value = selectedPresetId;
  }
  updateSettingsSummary();
}

function renderConversationList() {
  ui.conversationList.replaceChildren();
  appState.projects.forEach((project) => {
    const section = document.createElement("section"); section.className = "sidebar-project"; section.dataset.projectId = project.id;
    const header = document.createElement("div"); header.className = "sidebar-project-header";
    const name = document.createElement("strong"); name.textContent = project.name;
    name.title = `Владелец: ${profileById(project.owner_profile_id)?.name || "Основной пользователь"}`;
    const add = button("＋", "project-add-chat", () => createConversation(project.id).catch(showError));
    add.title = `Новый диалог в проекте «${project.name}»`;
    add.setAttribute("aria-label", add.title);
    header.append(name);
    if (project.owner_profile_id === appState.activeProfileId) {
      const share = button("⌘", "project-share-button", () => openProjectShare(project.id).catch(showError));
      share.title = "Открыть доступ к проекту"; share.setAttribute("aria-label", share.title); header.append(share);
    }
    header.append(add); section.append(header);
    const conversations = appState.conversations.filter((item) => item.project_id === project.id);
    const list = document.createElement("div"); list.className = "project-conversations";
    conversations.forEach((conversation) => list.append(conversationSidebarItem(conversation)));
    if (!conversations.length) list.append(memoryEmpty("Перетащите сюда диалог или нажмите ＋."));
    section.append(list);
    section.addEventListener("dragover", (event) => { event.preventDefault(); section.classList.add("drag-over"); });
    section.addEventListener("dragleave", () => section.classList.remove("drag-over"));
    section.addEventListener("drop", (event) => dropConversationOnProject(event, project.id, section));
    ui.conversationList.append(section);
  });
  const unassigned = document.createElement("section"); unassigned.className = "sidebar-project unassigned";
  const header = document.createElement("div"); header.className = "sidebar-project-header";
  const name = document.createElement("strong"); name.textContent = "Без проекта"; header.append(name); unassigned.append(header);
  const list = document.createElement("div"); list.className = "project-conversations";
  const conversations = appState.conversations.filter((item) => !item.project_id);
  conversations.forEach((conversation) => list.append(conversationSidebarItem(conversation)));
  if (!conversations.length) list.append(memoryEmpty("Непривязанных диалогов нет."));
  unassigned.append(list); ui.conversationList.append(unassigned);
}

function conversationSidebarItem(conversation) {
  const row = document.createElement("div"); row.className = "conversation-item-row";
  const open = document.createElement("button");
  open.className = `conversation-item${activeConversation?.id === conversation.id ? " active" : ""}`;
  open.draggable = true; open.dataset.conversationId = conversation.id; open.dataset.projectId = conversation.project_id || "";
  const title = document.createElement("strong"); title.textContent = conversation.title;
  const owner = profileById(conversation.owner_profile_id)?.name || "Основной пользователь";
  const meta = document.createElement("small"); meta.textContent = `${owner} · ${conversation.message_count} сообщ. · ${formatDate(conversation.updated_at)}`;
  open.append(title, meta);
  open.addEventListener("click", () => openConversation(conversation.id).catch(showError));
  open.addEventListener("dragstart", (event) => {
    event.dataTransfer.effectAllowed = conversation.project_id ? "copy" : "move";
    event.dataTransfer.setData("application/json", JSON.stringify({ id: conversation.id, project_id: conversation.project_id || null }));
    row.classList.add("dragging");
  });
  open.addEventListener("dragend", () => row.classList.remove("dragging"));
  row.append(open);
  if (conversation.project_id) {
    const extract = button("↻", "extract-history-button", () => extractHistoryMemory(conversation.id, extract));
    extract.title = "Извлечь память проекта из успешной истории активной ветки. Использует отдельный запрос DeepSeek.";
    extract.setAttribute("aria-label", `Извлечь память из истории: ${conversation.title}`);
    row.append(extract);
  }
  return row;
}

async function dropConversationOnProject(event, targetProjectId, section) {
  event.preventDefault(); section.classList.remove("drag-over");
  let dragged;
  try { dragged = JSON.parse(event.dataTransfer.getData("application/json")); } catch { return; }
  if (!dragged?.id || dragged.project_id === targetProjectId) return;
  ui.sidebarProjectStatus.textContent = dragged.project_id ? "Создаём независимую копию диалога…" : "Подключаем диалог к проекту…";
  try {
    const data = dragged.project_id
      ? await api(`/api/conversations/${dragged.id}/copy-to-project`, { method: "POST", body: { project_id: targetProjectId } })
      : await api(`/api/conversations/${dragged.id}/project`, { method: "PATCH", body: { project_id: targetProjectId } });
    await refreshState(); await openConversation(data.conversation.id);
    ui.sidebarProjectStatus.textContent = dragged.project_id ? "Копия создана; исходный диалог сохранён." : "Диалог подключён к проекту.";
  } catch (error) { ui.sidebarProjectStatus.textContent = error.message; showError(error); }
}

async function extractHistoryMemory(conversationId, control) {
  const original = control.textContent; control.disabled = true; control.textContent = "…";
  ui.sidebarProjectStatus.textContent = "Извлекаем память из истории через DeepSeek…";
  try {
    const data = await api(`/api/conversations/${conversationId}/extract-project-memory`, { method: "POST", body: {} });
    await refreshState();
    ui.sidebarProjectStatus.textContent = `Извлечение завершено: сохранено записей — ${data.revision.saved_count || 0}.`;
  } catch (error) { ui.sidebarProjectStatus.textContent = error.message; showError(error); }
  finally { control.disabled = false; control.textContent = original; }
}

async function createSidebarProject(event) {
  event.preventDefault();
  const data = await api("/api/projects", { method: "POST", body: {
    name: ui.newProjectName.value, description: "", create_dialog: true,
  }});
  ui.newProjectForm.hidden = true; ui.newProjectName.value = "";
  await refreshState(); await openConversation(data.conversation.id); ui.input.focus();
}

async function openProjectShare(projectId) {
  const data = await api(`/api/projects/${projectId}`);
  const project = data.project;
  sharingProjectId = projectId;
  ui.projectShareOwner.textContent = `Владелец: ${profileById(project.owner_profile_id)?.name || "—"}. Участники видят весь проект и могут работать в его чатах.`;
  ui.projectShareList.replaceChildren();
  appState.profiles.filter((profile) => profile.id !== project.owner_profile_id).forEach((profile) => {
    const label = document.createElement("label"); label.className = "share-profile-row";
    const checkbox = document.createElement("input"); checkbox.type = "checkbox"; checkbox.value = profile.id;
    checkbox.checked = (project.participant_profile_ids || []).includes(profile.id);
    const span = document.createElement("span"); span.textContent = profile.name;
    label.append(checkbox, span); ui.projectShareList.append(label);
  });
  if (!ui.projectShareList.children.length) ui.projectShareList.append(memoryEmpty("Сначала создайте ещё один профиль."));
  ui.projectShareError.textContent = "";
  ui.projectShareDialog.showModal();
}

async function saveProjectShare() {
  if (!sharingProjectId) return;
  const participantProfileIds = [...ui.projectShareList.querySelectorAll('input[type="checkbox"]:checked')].map((item) => item.value);
  try {
    await api(`/api/projects/${sharingProjectId}`, { method: "PATCH", body: { participant_profile_ids: participantProfileIds } });
    ui.projectShareDialog.close(); sharingProjectId = null; await refreshState();
  } catch (error) { ui.projectShareError.textContent = error.message; }
}

function renderMessages() {
  ui.messages.replaceChildren();
  renderTokenOverview();
  renderContextPanel();
  renderTaskState();
  const visibleMessages = conversationMessages();
  if (!visibleMessages.length) {
    const welcome = document.createElement("div"); welcome.className = "welcome";
    welcome.innerHTML = '<span class="welcome-mark">D</span><h1>Первый агент готов к диалогу</h1><p>Выберите пресет или настройте один запрос вручную. Контекст и технические данные сохраняются после каждого сообщения.</p>';
    ui.messages.append(welcome); return;
  }
  const tokenIndex = buildTokenIndex(visibleMessages);
  const points = new Map((activeConversation.branch_points || []).map((point) => [point.checkpoint_id, point]));
  visibleMessages.forEach((message) => {
    if (message.role === "event") {
      const event = document.createElement("div"); event.className = "event";
      const span = document.createElement("span"); span.textContent = message.content; event.append(span); ui.messages.append(event); return;
    }
    const element = ui.template.content.firstElementChild.cloneNode(true);
    element.classList.add(message.role);
    const localCommand = message.technical?.local_command;
    if (message.technical?.request_status === "failed") element.classList.add("failed");
    const appliedProfile = message.role === "user"
      ? profileById(message.author_profile_id)
      : (message.technical?.profile_snapshot || message.technical?.memory_context?.profile_snapshot);
    const profileName = appliedProfile?.name || appliedProfile?.profile_name || "Основной пользователь";
    const automaticContinuation = message.role === "user" && message.technical?.automatic_continuation === true;
    element.querySelector(".message-avatar").textContent = automaticContinuation ? "A" : (message.role === "user" ? profileName.slice(0, 1).toUpperCase() : "D");
    element.querySelector("header strong").textContent = message.role === "user"
      ? (automaticContinuation ? "Автопилот" : profileName)
      : (localCommand ? "Состояние задачи · сервер" : `DeepSeek · профиль «${profileName}»`);
    element.querySelector("time").textContent = formatDate(message.created_at, true);
    element.querySelector(".message-text").textContent = message.content;
    const audit = message.technical?.policy_audit;
    if (message.role === "assistant" && audit) {
      const auditNode = document.createElement("div");
      auditNode.className = `policy-audit${audit.accepted ? "" : " blocked"}`;
      const checked = Array.isArray(audit.checked_invariants) ? audit.checked_invariants.length : 0;
      auditNode.textContent = `Контроль: ${audit.accepted ? "пройден" : "заблокирован"} · этап ${audit.stage || "—"} · действие ${audit.detected_action_type || audit.action_type || "—"} · проверено инвариантов ${checked}${audit.stage_complete ? ` · этап завершён · переход ${audit.recommended_event || "—"}` : ""}`;
      element.querySelector(".message-text").after(auditNode);
    }
    renderMessageArtifacts(element.querySelector(".message-body"), message);
    const actions = element.querySelector(".message-actions");
    const fork = element.querySelector(".fork-button");
    const policyRetry = element.querySelector(".policy-retry-button");
    const fromUser = message.role === "user";
    if (localCommand) {
      actions.hidden = true;
    } else {
      fork.textContent = fromUser ? "Разветвить от запроса" : "Разветвить от ответа";
      fork.title = fromUser
        ? "Сразу получить другой ответ модели на этот запрос"
        : "Начать новое пользовательское продолжение после этого ответа";
      const messageRunId = message.technical?.task_state?.stage_run_id;
      const currentRunId = activeConversation?.task_state?.stage_run_id;
      const messageStage = message.technical?.task_state?.stage;
      const currentStage = activeConversation?.task_state?.stage;
      const previousStage = Boolean(
        (activeConversation?.task_handoffs || []).length
          ? messageRunId !== currentRunId
          : (messageStage && messageStage !== currentStage)
      );
      if (previousStage) fork.title = "Сообщение относится к предыдущему этапу; используйте handoff текущего этапа";
      fork.setAttribute("aria-label", fork.title);
      fork.disabled = taskIsPaused() || previousStage;
      fork.addEventListener("click", () => createBranch(message.id));
      if (!previousStage && message.role === "assistant" && message.technical?.request_status === "blocked" && message.parent_id) {
        fork.hidden = true;
        policyRetry.hidden = false;
        policyRetry.disabled = taskIsPaused();
        policyRetry.title = "Повторно отправить исходный запрос и заново проверить ответ";
        policyRetry.setAttribute("aria-label", policyRetry.title);
        policyRetry.addEventListener("click", () => createBranch(message.parent_id));
      }
      const point = points.get(message.id);
      if (point) renderBranchSwitch(element.querySelector(".branch-switch"), point);
    }
    renderMessageTokens(element.querySelector(".token-strip"), message, tokenIndex);
    const reasoning = element.querySelector(".reasoning");
    if (message.reasoning_content) { reasoning.hidden = false; reasoning.querySelector("div").textContent = message.reasoning_content; }
    const technical = element.querySelector(".technical");
    technical.querySelector("pre").textContent = JSON.stringify(message.technical || {}, null, 2);
    const memoryUsed = element.querySelector(".memory-used-button");
    if (message.role === "assistant" && message.technical?.request_status === "completed" && message.technical?.memory_context) {
      memoryUsed.hidden = false;
      memoryUsed.addEventListener("click", () => showMemorySnapshot(message.technical.memory_context));
    }
    ui.messages.append(element);
  });
  ui.messages.scrollTop = ui.messages.scrollHeight;
}

function renderMessageArtifacts(container, message) {
  const manifest = message.technical?.result_manifest;
  const artifacts = Array.isArray(manifest?.artifacts)
    ? manifest.artifacts
    : (Array.isArray(message.technical?.artifacts) ? message.technical.artifacts : []);
  if (!artifacts.length || !activeConversation?.id) return;
  const panel = document.createElement("section");
  panel.className = `artifact-panel${manifest ? " final-result" : ""}`;
  const heading = document.createElement("strong");
  heading.textContent = manifest ? "Итоговый результат" : "Созданные артефакты";
  panel.append(heading);
  artifacts.forEach((artifact) => {
    const row = document.createElement("div");
    row.className = "artifact-row";
    const info = document.createElement("div");
    const name = document.createElement("b");
    name.textContent = artifact.path || artifact.filename || "Артефакт";
    const meta = document.createElement("small");
    meta.textContent = `версия ${artifact.version || 1} · ${formatArtifactSize(artifact.size_bytes)}`;
    info.append(name, meta);
    const actions = document.createElement("div");
    const base = `/api/conversations/${encodeURIComponent(activeConversation.id)}/artifacts/${encodeURIComponent(artifact.id)}`;
    const open = document.createElement("a");
    open.href = `${base}?disposition=inline`;
    open.target = "_blank";
    open.rel = "noopener";
    open.textContent = "Открыть";
    const download = document.createElement("a");
    download.href = base;
    download.textContent = "Скачать";
    actions.append(open, download);
    row.append(info, actions);
    panel.append(row);
  });
  container.append(panel);
}

function formatArtifactSize(value) {
  const bytes = Number(value) || 0;
  if (bytes < 1024) return `${bytes} Б`;
  return `${(bytes / 1024).toFixed(bytes < 10240 ? 1 : 0)} КБ`;
}

function conversationMessages() {
  if (!activeConversation) return [];
  return Array.isArray(activeConversation.visible_messages) ? activeConversation.visible_messages : (activeConversation.messages || []);
}

function taskIsPaused() {
  return activeConversation?.task_state?.activity === "paused";
}

function taskStateBody() {
  const body = {
    description: ui.taskDescription.value,
    transition_mode: ui.taskTransitionMode.value,
    current_step: ui.taskCurrentStep.value,
    expected_action: ui.taskExpectedAction.value,
    plan: ui.taskPlan.value,
  };
  return body;
}

const taskEventLabels = {
  approve_plan: "Утвердить план → выполнение",
  complete_execution: "Завершить выполнение → проверка",
  pass_validation: "Валидация пройдена → завершить",
  return_to_planning: "Вернуть в планирование",
  validation_failed: "Валидация не пройдена → выполнение",
};
function renderTaskState() {
  if (!activeConversation) return;
  const state = activeConversation.task_state || {};
  ui.taskDescription.value = state.description || "";
  ui.taskTransitionMode.value = state.transition_mode || "manual";
  ui.taskStage.value = state.stage || "planning";
  ui.taskCurrentStep.value = state.current_step || "";
  ui.taskExpectedAction.value = state.expected_action || "";
  ui.taskPlan.value = state.plan || "";
  ui.taskStage.disabled = true;
  const paused = state.activity === "paused";
  ui.taskActivityBadge.textContent = paused ? "Приостановлена" : "Активна";
  ui.taskActivityBadge.classList.toggle("paused", paused);
  ui.taskActivityToggle.textContent = paused ? "Продолжить" : "Приостановить";
  ui.taskStateNote.textContent = paused
    ? "Выполнение и переходы заблокированы. Продолжение восстановит тот же этап и шаг."
    : (state.stage === "done"
      ? "Жизненный цикл завершён: переходов к следующему этапу нет."
      : state.transition_mode === "automatic"
      ? (autopilotRunning ? "Автопилот выполняет цепочку этапов…" : "Автопилот продолжит работу после завершённого ответа этапа.")
      : "Переход выполняется вручную только разрешённой кнопкой.");
  ui.taskAutopilotStop.disabled = state.transition_mode !== "automatic" && !autopilotRunning;
  renderTaskTransitionActions(state);
  renderTaskTransitionHistory(state.transition_history || []);
  renderTaskHandoffs(state);
  setSendingState(sending);
}

function renderTaskHandoffs(state) {
  const handoffs = Array.isArray(activeConversation?.active_task_handoffs) ? activeConversation.active_task_handoffs : [];
  const totals = activeConversation?.task_handoff_token_totals || {};
  ui.taskHandoffCount.textContent = handoffs.length;
  ui.taskHandoffList.replaceChildren();
  ui.taskHandoffNote.textContent = handoffs.length
    ? `История других этапов отсечена · текущий запуск ${state.stage_run_id || "—"} · токены handoff ${number(totals.total_tokens)}`
    : "До первого handoff используется последний непрерывный фрагмент текущего этапа; после перехода его заменит структурированный handoff.";
  const fieldLabels = {
    approved_plan: "Утверждённый план", decisions: "Решения", constraints: "Ограничения",
    acceptance_criteria: "Критерии готовности", completed_work: "Выполнено",
    validation_findings: "Результаты проверки", open_questions: "Открытые вопросы",
  };
  handoffs.forEach((handoff) => {
    const card = document.createElement("article"); card.className = "task-handoff-item";
    const title = document.createElement("strong");
    title.textContent = `${handoff.source_stage || "—"} → ${handoff.target_stage || "—"}`;
    const summary = document.createElement("p"); summary.textContent = handoff.summary || "";
    card.append(title, summary);
    Object.entries(fieldLabels).forEach(([field, label]) => {
      const values = Array.isArray(handoff[field]) ? handoff[field] : [];
      if (!values.length) return;
      const section = document.createElement("div");
      const caption = document.createElement("small"); caption.textContent = label;
      const list = document.createElement("ul");
      values.forEach((value) => { const item = document.createElement("li"); item.textContent = value; list.append(item); });
      section.append(caption, list); card.append(section);
    });
    ui.taskHandoffList.append(card);
  });
}

function renderTaskTransitionActions(state) {
  ui.taskTransitionActions.replaceChildren();
  if (state.transition_mode === "automatic") {
    const note = document.createElement("small");
    note.textContent = "Переходы выполняет автопилот после независимой проверки завершённости этапа.";
    ui.taskTransitionActions.append(note);
    return;
  }
  const events = Array.isArray(state.allowed_events) ? state.allowed_events : [];
  events.filter((event) => taskEventLabels[event]).forEach((event) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = event.startsWith("return") || event === "validation_failed" ? "secondary" : "primary";
    button.textContent = taskEventLabels[event];
    button.addEventListener("click", () => applyTaskEvent(event));
    ui.taskTransitionActions.append(button);
  });
}

function renderTaskTransitionHistory(history) {
  ui.taskHistoryCount.textContent = history.length;
  ui.taskHistoryList.replaceChildren();
  [...history].reverse().slice(0, 20).forEach((item) => {
    const row = document.createElement("div");
    const from = item.from || {}; const to = item.to || {};
    row.className = "task-history-row";
    row.textContent = `${formatDate(item.created_at, true)} · ${item.event_label || item.event}: ${from.stage || "—"}/${from.activity || "—"} → ${to.stage || "—"}/${to.activity || "—"}`;
    ui.taskHistoryList.append(row);
  });
  if (!history.length) ui.taskHistoryList.textContent = "Изменений пока нет.";
}

function setTaskStateControlsDisabled(disabled) {
  [ui.taskDescription, ui.taskTransitionMode, ui.taskCurrentStep, ui.taskExpectedAction, ui.taskPlan, ui.taskStateSave, ui.taskActivityToggle, ui.taskStateShow]
    .forEach((control) => { control.disabled = disabled; });
  ui.taskStage.disabled = true;
  ui.taskTransitionActions.querySelectorAll("button").forEach((button) => { button.disabled = disabled; });
}

async function saveTaskState() {
  if (!activeConversation || sending) return;
  const enablingAutopilot = activeConversation.task_state?.transition_mode !== "automatic" && ui.taskTransitionMode.value === "automatic";
  ui.taskStateSave.disabled = true; ui.taskActivityToggle.disabled = true; ui.error.textContent = "";
  let saved = false;
  try {
    const data = await api(`/api/conversations/${activeConversation.id}/task-state`, {
      method: "PATCH", body: taskStateBody(),
    });
    activeConversation = data.conversation;
    renderTaskState();
    await refreshState();
    saved = true;
  } catch (error) { showError(error); }
  finally { ui.taskStateSave.disabled = false; ui.taskActivityToggle.disabled = false; }
  if (saved && activeConversation?.task_state?.transition_mode === "automatic") {
    if (enablingAutopilot) autopilotStopRequested = false;
    await runAutomaticLifecycle();
  }
}

function toggleTaskActivity() {
  return applyTaskEvent(taskIsPaused() ? "resume" : "pause");
}

async function applyTaskEvent(event, { automatic = false } = {}) {
  if (!activeConversation || sending) return;
  sending = true;
  ui.error.textContent = "";
  setSendingState(true);
  setTaskStateControlsDisabled(true);
  if (!["pause", "resume"].includes(event)) ui.taskStateNote.textContent = "Формируется структурированный handoff этапа…";
  try {
    const data = await api(`/api/conversations/${activeConversation.id}/task-state/events`, {
      method: "POST", body: { event, automatic },
    });
    activeConversation = data.conversation;
    ui.taskStateReport.hidden = true;
    renderTaskState();
    await refreshState();
    return true;
  } catch (error) {
    if (error.data?.conversation) { activeConversation = error.data.conversation; renderTaskState(); }
    showError(error);
    return false;
  } finally {
    sending = false;
    setSendingState(false);
    setTaskStateControlsDisabled(false);
  }
}

async function showTaskStateReport() {
  if (!activeConversation) return;
  try {
    const data = await api(`/api/conversations/${activeConversation.id}/task-state`);
    ui.taskStateReport.textContent = data.report;
    ui.taskStateReport.hidden = false;
  } catch (error) { showError(error); }
}

function renderBranchSwitch(container, point) {
  container.hidden = false;
  const previous = container.querySelector(".branch-prev");
  const next = container.querySelector(".branch-next");
  container.querySelector("span").textContent = `${point.active_index + 1} / ${point.options.length}`;
  previous.disabled = point.active_index <= 0;
  next.disabled = point.active_index >= point.options.length - 1;
  previous.addEventListener("click", () => switchBranch(point, point.active_index - 1));
  next.addEventListener("click", () => switchBranch(point, point.active_index + 1));
}

function renderContextPanel() {
  if (!activeConversation) return;
  const context = activeConversation.context_management || { mode: "full", keep_recent_exchanges: 5, summary_batch_exchanges: 5 };
  const summaries = Array.isArray(activeConversation.summaries) ? activeConversation.summaries : [];
  const current = [...summaries].reverse().find((item) => item.id === context.active_summary_id) || null;
  const summaryTotals = activeConversation.summary_token_totals || {};
  const factsTotals = activeConversation.facts_token_totals || {};
  const completed = Number(activeConversation.current_stage_exchange_count) || 0;
  const covered = Number(current?.covered_exchange_count) || 0;
  const labels = { full: "Полная история", summary: "Summary", sliding: "Sliding Window", facts: "Sticky Facts" };
  const rules = {
    full: "В запрос отправляется вся успешная история только текущего посещения этапа.",
    summary: "Внутри текущего этапа последние 5 обменов остаются дословно, а старые сворачиваются моделью deepseek-v4-flash.",
    sliding: `В запрос отправляются только последние ${context.sliding_window_exchanges || 5} обменов текущего этапа.`,
    facts: `Facts текущего этапа обновляются после каждого сообщения и отправляются вместе с последними ${context.facts_window_exchanges || 5} его обменами.`,
  };

  ui.contextMode.value = context.mode;
  ui.contextStatus.textContent = labels[context.mode] || labels.full;
  ui.contextStatus.classList.toggle("enabled", context.mode !== "full");
  ui.contextRuleText.textContent = rules[context.mode] || rules.full;
  ui.windowSetting.hidden = !["sliding", "facts"].includes(context.mode);
  ui.windowExchanges.value = context.mode === "facts" ? (context.facts_window_exchanges || 5) : (context.sliding_window_exchanges || 5);
  ui.summarySection.hidden = context.mode !== "summary";
  ui.summaryHistorySection.hidden = context.mode !== "summary";
  ui.factsSection.hidden = context.mode !== "facts";
  ui.branchingSection.hidden = false;
  ui.contextMetrics.replaceChildren();
  addContextMetric("Обменов в пути", completed);
  if (context.mode === "summary") {
    addContextMetric("Свёрнуто", covered);
    addContextMetric("Версий", summaries.length);
    addContextMetric("Токены summary", summaryTotals.total_tokens || 0);
  } else if (context.mode === "facts") {
    addContextMetric("Фактов этапа", (activeConversation.active_stage_facts || []).length);
    addContextMetric("Обновлений", factsTotals.request_count || 0);
    addContextMetric("Токены facts", factsTotals.total_tokens || 0);
  } else if (context.mode === "sliding") {
    addContextMetric("Размер окна", context.sliding_window_exchanges || 5);
  }
  addContextMetric("Развилок в пути", (activeConversation.branch_points || []).length);
  addContextMetric("Всего сообщений", (activeConversation.messages || []).filter((item) => ["user", "assistant"].includes(item.role)).length);

  ui.summaryCurrent.textContent = current?.content || (context.mode === "summary"
    ? "Summary появится, когда за последними 5 обменами накопятся ещё 5 старых."
    : "Сжатие выключено. Ранее созданные версии остаются в JSON.");
  ui.summaryCurrentMeta.textContent = current ? `${current.covered_exchange_count} обменов · ${formatDate(current.created_at, true)}` : "";
  ui.summaryCount.textContent = `${summaries.length} ${plural(summaries.length, "версия", "версии", "версий")}`;
  ui.summaryHistory.replaceChildren();
  if (!summaries.length) {
    const empty = document.createElement("div"); empty.className = "empty-note"; empty.textContent = "Сохранённых версий пока нет."; ui.summaryHistory.append(empty);
  } else {
    [...summaries].reverse().forEach((summary, index) => {
      const details = document.createElement("details"); details.className = "summary-version"; details.open = index === 0;
      const heading = document.createElement("summary");
      const label = document.createElement("strong"); label.textContent = `Версия ${summaries.length - index}`;
      const meta = document.createElement("small"); meta.textContent = `${summary.covered_exchange_count || 0} обменов · ${formatDate(summary.created_at, true)}`;
      heading.append(label, meta);
      const content = document.createElement("div"); content.className = "summary-version-text"; content.textContent = summary.content || "";
      const usage = document.createElement("div"); usage.className = "summary-usage";
      const tokens = summary.technical?.usage || {};
      usage.textContent = `Вход ${number(tokens.input_tokens)} · выход ${number(tokens.output_tokens)} · всего ${number(tokens.total_tokens)}`;
      details.append(heading, content, usage); ui.summaryHistory.append(details);
    });
  }
  renderFacts();
  ui.branchCount.textContent = `${(activeConversation.branch_points || []).length} развилок`;
}

function renderFacts() {
  const facts = Array.isArray(activeConversation?.active_stage_facts) ? activeConversation.active_stage_facts : [];
  ui.factsCount.textContent = `${facts.length} ${plural(facts.length, "факт", "факта", "фактов")}`;
  ui.factsList.replaceChildren();
  if (!facts.length) {
    const empty = document.createElement("div"); empty.className = "empty-note"; empty.textContent = "Facts пока не сохранены.";
    ui.factsList.append(empty);
    return;
  }
  facts.forEach((fact) => {
    const row = document.createElement("div"); row.className = "fact-row";
    const key = document.createElement("input"); key.value = fact.key; key.maxLength = 120;
    const value = document.createElement("textarea"); value.value = fact.value; value.maxLength = 2000; value.rows = 2;
    const state = document.createElement("small"); state.textContent = fact.locked ? "🔒 закреплён" : "автоматический";
    const actions = document.createElement("div"); actions.className = "fact-actions";
    actions.append(
      button("Сохранить", "secondary", () => saveFact(fact.id, key.value, value.value, true)),
      button(fact.locked ? "Открепить" : "Закрепить", "secondary", () => saveFact(fact.id, key.value, value.value, !fact.locked)),
      button("×", "danger", () => removeFact(fact.id)),
    );
    row.append(key, value, state, actions); ui.factsList.append(row);
  });
}

function addContextMetric(label, value) {
  const item = document.createElement("div");
  const caption = document.createElement("small"); caption.textContent = label;
  const numberElement = document.createElement("strong"); numberElement.textContent = number(value);
  item.append(caption, numberElement); ui.contextMetrics.append(item);
}

function plural(value, one, few, many) {
  const numberValue = Math.abs(Number(value)) % 100;
  const last = numberValue % 10;
  if (numberValue > 10 && numberValue < 20) return many;
  if (last === 1) return one;
  if (last > 1 && last < 5) return few;
  return many;
}

function number(value) { return new Intl.NumberFormat("ru-RU").format(Number(value) || 0); }

function addTokenBadge(container, label, value, title = "") {
  const badge = document.createElement("span");
  badge.className = "token-badge"; badge.title = title;
  const caption = document.createElement("small"); caption.textContent = label;
  const count = document.createElement("strong"); count.textContent = number(value);
  badge.append(caption, count); container.append(badge);
}

function normalizedUsage(message) {
  const usage = message?.technical?.usage || {};
  const input = Number(usage.input_tokens) || 0;
  const output = Number(usage.output_tokens) || 0;
  return {
    input_tokens: input, output_tokens: output,
    total_tokens: Number(usage.total_tokens) || input + output,
    cached_input_tokens: Number(usage.cached_input_tokens) || 0,
    uncached_input_tokens: Number(usage.uncached_input_tokens) || 0,
    reasoning_tokens: Number(usage.reasoning_tokens) || 0,
  };
}

function buildTokenIndex(messages) {
  const exchanges = new Map();
  const cumulative = { input_tokens: 0, output_tokens: 0, total_tokens: 0, cached_input_tokens: 0, uncached_input_tokens: 0, reasoning_tokens: 0, request_count: 0 };
  messages.forEach((message) => {
    if (message.role !== "assistant" || !message.technical?.usage || message.technical?.request_status === "failed") return;
    const usage = normalizedUsage(message);
    Object.keys(usage).forEach((key) => { cumulative[key] += usage[key]; });
    cumulative.request_count += 1;
    exchanges.set(message.exchange_id, { usage, cumulative: { ...cumulative } });
  });
  return { exchanges, totals: cumulative };
}

function renderMessageTokens(container, message, tokenIndex) {
  const exchange = tokenIndex.exchanges.get(message.exchange_id);
  if (!exchange) return;
  container.hidden = false;
  if (message.role === "user") {
    addTokenBadge(container, "Контекст вызова", exchange.usage.input_tokens, "Системный промпт, предыдущая история и этот запрос");
    addTokenBadge(container, "Из кэша", exchange.usage.cached_input_tokens, "Входные токены, найденные в кэше DeepSeek");
    addTokenBadge(container, "Без кэша", exchange.usage.uncached_input_tokens, "Входные токены, не найденные в кэше DeepSeek");
  } else if (message.role === "assistant") {
    addTokenBadge(container, "Выход с reasoning", exchange.usage.output_tokens, "Все выходные токены, включая reasoning");
    addTokenBadge(container, "Reasoning внутри", exchange.usage.reasoning_tokens, "Часть выходных токенов; отдельно к итогу не прибавляется");
    addTokenBadge(container, "Всего за вызов", exchange.usage.total_tokens, "Входные плюс все выходные токены, уже включая reasoning");
    addTokenBadge(container, "Σ диалога", exchange.cumulative.total_tokens, "Сумма токенов всех вызовов к этому месту диалога");
  }
}

function renderTokenOverview() {
  ui.tokenOverview.replaceChildren();
  const messages = conversationMessages();
  const totals = buildTokenIndex(messages).totals;
  if (!totals.request_count) { ui.tokenOverview.hidden = true; return; }
  ui.tokenOverview.hidden = false;
  addTokenBadge(ui.tokenOverview, "Σ диалога", totals.total_tokens, "Входные плюс выходные токены всех вызовов; reasoning уже входит в выходные");
  addTokenBadge(ui.tokenOverview, "Σ контекст", totals.input_tokens);
  addTokenBadge(ui.tokenOverview, "Σ выход", totals.output_tokens, "Все выходные токены, включая reasoning");
  addTokenBadge(ui.tokenOverview, "Σ reasoning", totals.reasoning_tokens, "Часть Σ выхода; отдельно к Σ диалога не прибавляется");
  addTokenBadge(ui.tokenOverview, "Σ кэш", totals.cached_input_tokens);
  addTokenBadge(ui.tokenOverview, "Вызовов", totals.request_count);
  renderContextWindowUsage(messages);
}

function renderContextWindowUsage(messages) {
  const lastAssistant = [...messages].reverse().find((message) => (
    message.role === "assistant" && message.technical?.usage && message.technical?.request_status !== "failed"
  ));
  if (!lastAssistant) return;
  const usage = normalizedUsage(lastAssistant);
  const model = lastAssistant.technical?.settings?.model || lastAssistant.technical?.model || ui.model.value;
  const windowTokens = Number(appState.provider?.context_windows?.[model]) || 0;
  if (!windowTokens) return;

  const percent = Math.min(100, (usage.input_tokens / windowTokens) * 100);
  const precision = percent > 0 && percent < 0.01 ? 4 : percent < 1 ? 2 : 1;
  const wrapper = document.createElement("div");
  wrapper.className = "context-window-usage";
  wrapper.title = "Последний фактически отправленный вход: системный промпт, выбранная история и сообщение пользователя. Выход модели в эту долю не входит.";
  const label = document.createElement("div");
  label.className = "context-window-label";
  const caption = document.createElement("small");
  caption.textContent = `Контекст последнего вызова · ${model}`;
  const value = document.createElement("strong");
  value.textContent = `${number(usage.input_tokens)} / ${number(windowTokens)} · ${percent.toFixed(precision)}%`;
  label.append(caption, value);
  const track = document.createElement("div");
  track.className = "context-window-track";
  track.setAttribute("role", "progressbar");
  track.setAttribute("aria-label", "Заполнение контекстного окна последним запросом");
  track.setAttribute("aria-valuemin", "0");
  track.setAttribute("aria-valuemax", String(windowTokens));
  track.setAttribute("aria-valuenow", String(Math.min(usage.input_tokens, windowTokens)));
  const fill = document.createElement("span");
  fill.style.width = `${percent}%`;
  track.append(fill);
  wrapper.append(label, track);
  ui.tokenOverview.append(wrapper);
}

function renderPresetSelect() {
  const current = selectedPresetId;
  ui.presetSelect.replaceChildren(new Option("Разовые настройки", ""));
  appState.presets.forEach((preset) => ui.presetSelect.add(new Option(preset.name, preset.id)));
  ui.presetSelect.value = appState.presets.some((item) => item.id === current) ? current : "";
}

function renderPresetList() {
  ui.presetList.replaceChildren();
  if (!appState.presets.length) { const empty = document.createElement("div"); empty.className = "empty-note"; empty.textContent = "Сохранённых пресетов пока нет."; ui.presetList.append(empty); return; }
  appState.presets.forEach((preset) => {
    const row = document.createElement("div"); row.className = "preset-row";
    const info = document.createElement("div"); const name = document.createElement("strong"); name.textContent = preset.name;
    const description = document.createElement("p");
    const details = `${preset.settings.model} · t=${preset.settings.temperature} · Reasoning: ${preset.settings.reasoning_enabled ? "Да" : "Нет"}`;
    description.textContent = preset.description ? `${preset.description} · ${details}` : details;
    info.append(name, description);
    const actions = document.createElement("div"); actions.className = "actions";
    const use = button("Выбрать", "secondary", () => {
      selectPreset(preset.id);
      ui.presetDialog.close();
      ui.settingsPanel.classList.remove("open");
    });
    const edit = button("Изменить", "secondary", () => editPreset(preset));
    const remove = button("×", "danger", () => deletePreset(preset.id));
    actions.append(use, edit, remove); row.append(info, actions); ui.presetList.append(row);
  });
}

function button(label, className, handler) { const value = document.createElement("button"); value.type = "button"; value.className = className; value.textContent = label; value.addEventListener("click", handler); return value; }

function selectPreset(id) {
  selectedPresetId = id;
  ui.presetSelect.value = id;
  const preset = appState.presets.find((item) => item.id === id);
  if (preset) applySettings(preset.settings);
  else applySettings(appState.defaults);
  updateSettingsSummary();
}

function applySettings(settings) {
  ui.model.value = settings.model; ui.system.value = settings.system_prompt;
  ui.temperature.value = settings.temperature; ui.topP.value = settings.top_p;
  ui.reasoning.value = String(settings.reasoning_enabled); ui.effort.value = settings.reasoning_effort;
  ui.maxTokens.value = settings.max_tokens ?? ""; ui.stop.value = (settings.stop || []).join("\n");
  ui.format.value = settings.response_format; ui.logprobs.checked = settings.logprobs;
  ui.topLogprobs.value = settings.top_logprobs ?? "";
  updateSettingsSummary();
}

function readSettings() {
  return {
    model: ui.model.value, system_prompt: ui.system.value.trim(), temperature: Number(ui.temperature.value), top_p: Number(ui.topP.value),
    reasoning_enabled: ui.reasoning.value === "true", reasoning_effort: ui.effort.value,
    max_tokens: ui.maxTokens.value === "" ? null : Number(ui.maxTokens.value),
    stop: ui.stop.value.split("\n").map((value) => value.trim()).filter(Boolean), response_format: ui.format.value,
    logprobs: ui.logprobs.checked, top_logprobs: ui.topLogprobs.value === "" ? null : Number(ui.topLogprobs.value),
  };
}

function updateSettingsSummary() {
  if (!appState.defaults) return;
  const value = readSettings();
  const preset = appState.presets.find((item) => item.id === selectedPresetId);
  const changed = preset && JSON.stringify(value) !== JSON.stringify(preset.settings);
  const source = preset ? `${preset.name}${changed ? " · изменён" : ""}` : "Разовые";
  ui.settingsSummary.textContent = `${source} · ${value.model} · t=${value.temperature} · reasoning ${value.reasoning_enabled ? "on" : "off"}`;
}

async function submitMessage(event) {
  event.preventDefault(); if (sending || !activeConversation) return;
  if (taskIsPaused()) { ui.error.textContent = "Задача приостановлена. Нажмите «Продолжить»."; return; }
  if (mediaRecorder?.state === "recording") {
    voiceSubmitAfterTranscription = true;
    voiceBusy = true;
    ui.voiceStatus.textContent = "Запись завершена — распознаю и сразу отправлю…";
    mediaRecorder.stop();
    return;
  }
  if (voiceBusy) {
    voiceSubmitAfterTranscription = true;
    ui.voiceStatus.textContent = "После распознавания текст будет сразу отправлен…";
    return;
  }
  const content = ui.input.value.trim(); if (!content) return;
  autopilotStopRequested = false;
  const sent = await sendMessageContent(content);
  if (sent) await runAutomaticLifecycle();
}

async function sendMessageContent(content, { automatic = false } = {}) {
  content = String(content || "").trim();
  if (!content || sending || !activeConversation) return false;
  sending = true; setSendingState(true); ui.error.textContent = "";
  const pending = renderPendingExchange(content);
  try {
    const data = await api(`/api/conversations/${activeConversation.id}/messages`, { method: "POST", body: { content, automatic, preset_id: selectedPresetId || null, settings: readSettings() } });
    activeConversation = data.conversation; ui.input.value = ""; resizeInput(); await refreshState(); renderMessages(); ui.chatTitle.value = activeConversation.title;
    return true;
  } catch (error) {
    if (error.data?.conversation) { activeConversation = error.data.conversation; renderMessages(); await refreshState(); }
    showError(error);
    return false;
  } finally { sending = false; setSendingState(false); pending.forEach((element) => element.remove()); ui.input.focus(); }
}

function automaticDecision() {
  const state = activeConversation?.task_state || {};
  if (state.transition_mode !== "automatic" || state.activity === "paused" || state.stage === "done") return null;
  const messages = Array.isArray(activeConversation?.visible_messages) ? activeConversation.visible_messages : [];
  const assistant = [...messages].reverse().find((message) =>
    message.role === "assistant"
    && message.technical?.request_status === "completed"
    && message.technical?.task_state?.stage_run_id === state.stage_run_id
  );
  const audit = assistant?.technical?.policy_audit || {};
  const event = audit.stage_complete === true ? audit.recommended_event : null;
  return event && (state.allowed_events || []).includes(event) ? event : null;
}

function automaticPrompt() {
  const state = activeConversation?.task_state || {};
  return `[Автопилот] Продолжить этап «${state.stage || "—"}»: ${state.current_step || "выполнить текущий шаг"}`;
}

async function setTransitionMode(mode) {
  if (!activeConversation) return;
  const data = await api(`/api/conversations/${activeConversation.id}/task-state`, {
    method: "PATCH", body: { transition_mode: mode },
  });
  activeConversation = data.conversation;
  renderTaskState();
  await refreshState();
}

async function runAutomaticLifecycle() {
  if (autopilotRunning || activeConversation?.task_state?.transition_mode !== "automatic") return;
  if (autopilotStopRequested) {
    await setTransitionMode("manual");
    return;
  }
  autopilotRunning = true;
  let transitions = 0;
  renderTaskState();
  try {
    while (!autopilotStopRequested) {
      const event = automaticDecision();
      if (!event) break;
      if (transitions >= MAX_AUTOPILOT_TRANSITIONS) {
        autopilotStopRequested = true;
        ui.error.textContent = `Автопилот остановлен после ${MAX_AUTOPILOT_TRANSITIONS} переходов: требуется проверка пользователя.`;
        break;
      }
      const transitioned = await applyTaskEvent(event, { automatic: true });
      if (!transitioned || autopilotStopRequested || activeConversation?.task_state?.stage === "done") break;
      transitions += 1;
      const continued = await sendMessageContent(automaticPrompt(), { automatic: true });
      if (!continued) break;
    }
  } finally {
    autopilotRunning = false;
    if (autopilotStopRequested && activeConversation?.task_state?.transition_mode === "automatic") {
      try { await setTransitionMode("manual"); }
      catch (error) { showError(error); }
    }
    renderTaskState();
  }
}

async function stopAutopilot() {
  autopilotStopRequested = true;
  ui.taskAutopilotStop.disabled = true;
  ui.taskStateNote.textContent = sending
    ? "Остановка запрошена: текущий ответ сохранится, новые сообщения не отправятся."
    : "Автопилот остановлен.";
  if (!autopilotRunning && !sending && activeConversation?.task_state?.transition_mode === "automatic") {
    await setTransitionMode("manual");
  }
}

function renderPendingExchange(content) {
  const welcome = ui.messages.querySelector(".welcome"); if (welcome) welcome.remove();
  const user = ui.template.content.firstElementChild.cloneNode(true); user.classList.add("user", "pending-message");
  user.querySelector(".message-avatar").textContent = "Вы"; user.querySelector("header strong").textContent = "Вы";
  user.querySelector("time").textContent = "только что"; ui.messages.append(user);
  user.querySelector(".message-text").textContent = content;
  user.querySelector(".technical").remove(); user.querySelector(".reasoning").remove(); user.querySelector(".message-actions").remove();

  const waiting = document.createElement("article"); waiting.className = "message assistant waiting-message";
  waiting.innerHTML = '<div class="message-avatar">D</div><div class="message-body"><header><strong>DeepSeek</strong><span class="sent-badge">Запрос получен</span></header><div class="thinking-line"><i></i><i></i><i></i><span>Формирует ответ…</span></div></div>';
  ui.messages.append(waiting); ui.messages.scrollTop = ui.messages.scrollHeight;
  return [user, waiting];
}

function setSendingState(active) {
  const paused = taskIsPaused();
  ui.send.disabled = active || paused; ui.input.disabled = active || paused; ui.form.classList.toggle("sending", active);
  ui.form.classList.toggle("paused", paused);
  ui.send.textContent = active ? "…" : (paused ? "⏸" : "↑"); ui.requestStatus.hidden = !active;
  ui.settingsToggle.disabled = active; ui.presetSelect.disabled = active;
  ui.contextMode.disabled = active;
  ui.windowExchanges.disabled = active;
  ui.voiceButton.disabled = active || paused || !voiceState.ready;
  ui.messages.querySelectorAll(".policy-retry-button").forEach((button) => { button.disabled = active || paused; });
}

async function toggleVoiceRecording() {
  if (mediaRecorder?.state === "recording") {
    mediaRecorder.stop();
    return;
  }
  if (!voiceState.ready || voiceBusy) return;
  if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
    ui.error.textContent = "Этот браузер не поддерживает запись с микрофона.";
    return;
  }
  ui.error.textContent = "";
  voiceSubmitAfterTranscription = false;
  try {
    microphoneStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
    const mimeType = candidates.find((item) => MediaRecorder.isTypeSupported(item)) || "";
    mediaRecorder = mimeType ? new MediaRecorder(microphoneStream, { mimeType }) : new MediaRecorder(microphoneStream);
    audioChunks = [];
    mediaRecorder.addEventListener("dataavailable", (event) => { if (event.data.size) audioChunks.push(event.data); });
    mediaRecorder.addEventListener("stop", finishVoiceRecording, { once: true });
    mediaRecorder.start(250);
    ui.voiceButton.classList.add("recording");
    ui.voiceButton.disabled = false;
    ui.voiceButton.title = "Остановить запись";
    ui.voiceStatus.textContent = "Идёт запись… микрофон — проверить текст, стрелка — сразу отправить";
    ui.voiceStatus.className = "voice-status recording";
    recordingTimer = window.setTimeout(() => { if (mediaRecorder?.state === "recording") mediaRecorder.stop(); }, 5 * 60 * 1000);
  } catch (error) {
    stopMicrophoneTracks();
    ui.error.textContent = error?.name === "NotAllowedError"
      ? "Доступ к микрофону запрещён. Разрешите его для локальной страницы."
      : `Не удалось включить микрофон: ${error.message || error}`;
    renderVoiceStatus();
  }
}

async function finishVoiceRecording() {
  window.clearTimeout(recordingTimer);
  stopMicrophoneTracks();
  const mimeType = mediaRecorder?.mimeType || "audio/webm";
  const chunks = audioChunks;
  mediaRecorder = null;
  audioChunks = [];
  if (!chunks.length) { voiceBusy = false; voiceSubmitAfterTranscription = false; ui.error.textContent = "Микрофон не записал звук."; renderVoiceStatus(); return; }

  voiceBusy = true;
  ui.voiceButton.disabled = true;
  ui.voiceButton.classList.remove("recording");
  ui.voiceButton.classList.add("processing");
  ui.voiceStatus.textContent = "Whisper распознаёт запись на RX 6600…";
  ui.voiceStatus.className = "voice-status processing";
  let directText = "";
  try {
    const extension = mimeType.includes("ogg") ? "ogg" : mimeType.includes("mp4") ? "mp4" : "webm";
    const body = new FormData();
    body.append("audio", new Blob(chunks, { type: mimeType }), `recording.${extension}`);
    const response = await fetch("/api/voice/transcribe", { method: "POST", body });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.ok === false) throw new Error(data.error || `Ошибка HTTP ${response.status}`);
    const recognizedText = String(data.text || "").trim();
    if (!recognizedText) throw new Error("Whisper не распознал текст.");
    if (voiceSubmitAfterTranscription) {
      directText = recognizedText;
      ui.voiceStatus.textContent = "Текст распознан — отправляю…";
    } else {
      insertRecognizedText(recognizedText);
      ui.voiceStatus.textContent = "Текст распознан — проверьте и отправьте";
    }
    ui.voiceStatus.className = "voice-status success";
  } catch (error) {
    ui.error.textContent = error.message || String(error);
  } finally {
    voiceBusy = false;
    voiceSubmitAfterTranscription = false;
    ui.voiceButton.classList.remove("processing");
    ui.voiceButton.disabled = sending || !voiceState.ready;
    if (directText) {
      autopilotStopRequested = false;
      const sent = await sendMessageContent(directText);
      ui.voiceStatus.textContent = sent ? "Голосовое сообщение отправлено" : "Текст распознан, но не отправлен";
      ui.voiceStatus.className = sent ? "voice-status success" : "voice-status error";
      if (!sent) insertRecognizedText(directText);
      if (sent) await runAutomaticLifecycle();
    } else ui.input.focus();
  }
}

function stopMicrophoneTracks() {
  if (microphoneStream) microphoneStream.getTracks().forEach((track) => track.stop());
  microphoneStream = null;
}

function insertRecognizedText(text) {
  const start = ui.input.selectionStart ?? ui.input.value.length;
  const end = ui.input.selectionEnd ?? start;
  const before = ui.input.value.slice(0, start);
  const after = ui.input.value.slice(end);
  const prefix = before && !/\s$/.test(before) ? " " : "";
  const suffix = after && !/^\s/.test(after) ? " " : "";
  ui.input.value = `${before}${prefix}${String(text).trim()}${suffix}${after}`;
  const cursor = before.length + prefix.length + String(text).trim().length;
  ui.input.setSelectionRange(cursor, cursor);
  resizeInput();
}

async function changeContextMode() {
  if (!activeConversation || sending) return;
  const requestedMode = ui.contextMode.value;
  ui.contextMode.disabled = true; ui.error.textContent = "";
  try {
    const size = Math.max(1, Math.min(50, Number(ui.windowExchanges.value) || 5));
    const body = { mode: requestedMode };
    if (requestedMode === "sliding") body.sliding_window_exchanges = size;
    if (requestedMode === "facts") body.facts_window_exchanges = size;
    const data = await api(`/api/conversations/${activeConversation.id}/context`, { method: "PATCH", body });
    activeConversation = data.conversation;
    renderMessages();
    await refreshState();
  } catch (error) {
    ui.contextMode.value = activeConversation.context_management?.mode || "full";
    showError(error);
  } finally { ui.contextMode.disabled = false; }
}

async function createBranch(messageId) {
  if (!activeConversation || sending) return;
  sending = true; setSendingState(true); ui.error.textContent = "";
  try {
    const data = await api(`/api/conversations/${activeConversation.id}/branches`, {
      method: "POST", body: { checkpoint_id: messageId },
    });
    activeConversation = data.conversation;
    renderMessages();
    ui.input.focus();
  } catch (error) {
    showError(error);
  } finally {
    sending = false; setSendingState(false);
  }
}

async function switchBranch(point, index) {
  if (!activeConversation || sending || index < 0 || index >= point.options.length) return;
  try {
    const data = await api(`/api/conversations/${activeConversation.id}/branches/active`, {
      method: "PATCH",
      body: { checkpoint_id: point.checkpoint_id, child_id: point.options[index].child_id },
    });
    activeConversation = data.conversation;
    renderMessages();
  } catch (error) { showError(error); }
}

async function addManualFact(event) {
  event.preventDefault();
  if (!activeConversation) return;
  ui.factsError.textContent = "";
  try {
    const data = await api(`/api/conversations/${activeConversation.id}/facts`, {
      method: "POST", body: { key: ui.factKey.value, value: ui.factValue.value },
    });
    activeConversation = data.conversation;
    ui.factKey.value = ""; ui.factValue.value = "";
    renderContextPanel();
  } catch (error) { ui.factsError.textContent = error.message; }
}

async function saveFact(id, key, value, locked) {
  ui.factsError.textContent = "";
  try {
    const data = await api(`/api/conversations/${activeConversation.id}/facts/${id}`, {
      method: "PUT", body: { key, value, locked },
    });
    activeConversation = data.conversation;
    renderContextPanel();
  } catch (error) { ui.factsError.textContent = error.message; }
}

async function removeFact(id) {
  if (!confirm("Удалить этот факт? Он сможет появиться снова при следующем автоматическом обновлении.")) return;
  try {
    const data = await api(`/api/conversations/${activeConversation.id}/facts/${id}`, { method: "DELETE" });
    activeConversation = data.conversation;
    renderContextPanel();
  } catch (error) { ui.factsError.textContent = error.message; }
}

async function renameChat() {
  if (!activeConversation) return;
  try { const data = await api(`/api/conversations/${activeConversation.id}`, { method: "PATCH", body: { title: ui.chatTitle.value } }); activeConversation = data.conversation; await refreshState(); }
  catch (error) { showError(error); }
}

async function deleteChat() {
  if (!activeConversation || !confirm(`Удалить диалог «${activeConversation.title}»?`)) return;
  await api(`/api/conversations/${activeConversation.id}`, { method: "DELETE" }); activeConversation = null; await refreshState();
  if (appState.conversations.length) await openConversation(appState.conversations[0].id); else renderEmptyWorkspace();
}

function openPresetDialog(forSave = false) { renderPresetList(); clearPresetEditor(); if (forSave) ui.presetName.focus(); ui.presetDialog.showModal(); }
function clearPresetEditor() { ui.presetEditId.value = ""; ui.presetName.value = ""; ui.presetDescription.value = ""; ui.presetReasoning.value = ui.reasoning.value; ui.presetError.textContent = ""; $("#preset-editor-title").textContent = "Новый пресет из текущих настроек"; }
function editPreset(preset) { ui.presetEditId.value = preset.id; ui.presetName.value = preset.name; ui.presetDescription.value = preset.description; ui.presetReasoning.value = String(preset.settings.reasoning_enabled); applySettings(preset.settings); $("#preset-editor-title").textContent = "Изменить пресет"; }

async function savePreset() {
  ui.presetError.textContent = "";
  const presetSettings = readSettings(); presetSettings.reasoning_enabled = ui.presetReasoning.value === "true";
  const body = { name: ui.presetName.value, description: ui.presetDescription.value, settings: presetSettings };
  const id = ui.presetEditId.value;
  try {
    const data = await api(id ? `/api/presets/${id}` : "/api/presets", { method: id ? "PUT" : "POST", body });
    selectedPresetId = data.preset.id;
    applySettings(data.preset.settings);
    await refreshState();
    clearPresetEditor();
    ui.presetSelect.value = selectedPresetId;
    updateSettingsSummary();
    ui.presetDialog.close();
    ui.settingsPanel.classList.remove("open");
  } catch (error) { ui.presetError.textContent = error.message; }
}

async function deletePreset(id) {
  const preset = appState.presets.find((item) => item.id === id); if (!confirm(`Удалить пресет «${preset?.name || ""}»?`)) return;
  await api(`/api/presets/${id}`, { method: "DELETE" }); if (selectedPresetId === id) selectedPresetId = ""; await refreshState(); renderPresetList(); updateSettingsSummary();
}

async function importFile(file) {
  try { const bundle = JSON.parse(await file.text()); const data = await api("/api/import", { method: "POST", body: bundle }); await refreshState(); await openConversation(data.conversation.id); }
  catch (error) { showError(error); } finally { ui.importFile.value = ""; }
}

const MEMORY_KINDS = ["goal", "context", "preference", "decision", "requirement", "constraint", "resource", "environment", "definition", "open_question", "result", "risk"];

async function openMemoryDialog() {
  ui.memoryError.textContent = "";
  document.querySelectorAll(".memory-form select[name=kind]").forEach((select) => {
    if (!select.options.length) MEMORY_KINDS.forEach((kind) => select.add(new Option(kind, kind)));
  });
  await refreshMemoryDialog();
  ui.memoryDialog.showModal();
}

async function refreshMemoryDialog() {
  const context = activeConversation?.context_management || {};
  const summary = activeConversation?.summaries?.find((item) => item.id === context.active_summary_id);
  const activeCount = conversationMessages().filter((item) => item.role === "user" || item.role === "assistant").length;
  ui.memoryConversationSummary.replaceChildren(
    memoryInfo("Стратегия", context.mode || "full"),
    memoryInfo("Сообщений активной ветки", String(activeCount)),
    memoryInfo("Summary", summary?.content || "Не создан"),
    memoryInfo("Sticky Facts", `${activeConversation?.facts?.length || 0}`),
  );
  ui.memoryProjectSelect.replaceChildren(new Option("Без проекта", ""));
  appState.projects.forEach((project) => ui.memoryProjectSelect.add(new Option(project.name, project.id)));
  ui.memoryProjectSelect.value = activeConversation?.project_id || "";
  await Promise.all([loadProjectMemory(), loadUserMemory(), loadPolicy()]);
}

function memoryInfo(label, value) {
  const item = document.createElement("div"); item.className = "memory-info";
  const strong = document.createElement("strong"); strong.textContent = label;
  const span = document.createElement("span"); span.textContent = value;
  item.append(strong, span); return item;
}

async function loadProjectMemory() {
  const projectId = activeConversation?.project_id;
  ui.memoryProjectEditor.hidden = !projectId; ui.memoryProjectForm.hidden = !projectId;
  if (!projectId) { ui.memoryProjectList.replaceChildren(memoryEmpty("Выберите или создайте проект.")); return; }
  const data = await api(`/api/projects/${projectId}`);
  const project = data.project;
  ui.memoryProjectName.value = project.name; ui.memoryProjectDescription.value = project.description || "";
  ui.memoryProjectAuto.checked = project.settings?.automatic_extraction !== false;
  const ownsProject = project.owner_profile_id === appState.activeProfileId;
  ui.memoryProjectName.disabled = !ownsProject; ui.memoryProjectDescription.disabled = !ownsProject;
  ui.memoryProjectAuto.disabled = !ownsProject; ui.memorySaveProject.hidden = !ownsProject;
  ui.memoryProjectConversations.textContent = project.conversations?.map((item) => item.title).join(" · ") || "Связанных диалогов пока нет.";
  renderMemoryEntries(ui.memoryProjectList, project.manual_memory || [], project.learned_memory || [], "project", projectId);
}

async function loadUserMemory() {
  const data = await api("/api/memory/user");
  ui.memoryUserAuto.checked = data.settings?.automatic_extraction === true;
  renderMemoryEntries(ui.memoryUserList, data.manual || [], data.learned || [], "user", null);
}

async function loadPolicy() {
  const data = await api("/api/memory/policy");
  ui.memoryPolicyList.replaceChildren();
  (data.policy.rules || []).forEach((rule) => { const li = document.createElement("li"); li.textContent = rule.text; ui.memoryPolicyList.append(li); });
}

function renderMemoryEntries(container, manual, learned, scope, projectId) {
  container.replaceChildren();
  [["Ручная подтверждённая память", manual], ["Автоматическая подтверждённая память", learned]].forEach(([title, entries]) => {
    const section = document.createElement("section"); section.className = "memory-group";
    const heading = document.createElement("h3"); heading.textContent = `${title} · ${entries.length}`; section.append(heading);
    if (!entries.length) section.append(memoryEmpty("Записей нет."));
    entries.forEach((entry) => section.append(memoryEntry(entry, scope, projectId)));
    container.append(section);
  });
}

function memoryEmpty(text) { const value = document.createElement("p"); value.className = "empty-note"; value.textContent = text; return value; }

function memoryEntry(entry, scope, projectId) {
  const row = document.createElement("article"); row.className = "memory-entry";
  const header = document.createElement("div"); header.className = "memory-entry-head";
  const key = document.createElement("strong"); key.textContent = `${entry.kind} · ${entry.key}`;
  const badge = document.createElement("small"); badge.textContent = entry.locked ? "manual · защищено" : "автоматически извлечено · подтверждено";
  header.append(key, badge);
  const value = document.createElement("p"); value.textContent = entry.value;
  const origin = document.createElement("small"); origin.textContent = entry.source === "automatic" ? `Источник: диалог ${entry.source_conversation_id || "—"}, обмен ${entry.source_exchange_id || "—"}` : "Источник: пользователь";
  const actions = document.createElement("div"); actions.className = "fact-actions";
  if (!entry.locked) actions.append(button("Закрепить", "secondary", () => mutateMemory(scope, projectId, entry.id, { pin: true })));
  const editButton = button("Изменить", "secondary", () => editMemoryInline(row, value, actions, editButton, scope, projectId, entry));
  actions.append(editButton);
  actions.append(button(entry.status === "inactive" ? "Включить" : "Отключить", "secondary", () => mutateMemory(scope, projectId, entry.id, { status: entry.status === "inactive" ? "active" : "inactive" })));
  actions.append(button("Удалить", "danger", () => deleteMemory(scope, projectId, entry.id)));
  row.append(header, value, origin, actions); return row;
}

async function mutateMemory(scope, projectId, memoryId, body) {
  const url = scope === "project" ? `/api/projects/${projectId}/memory/${memoryId}` : `/api/memory/user/${memoryId}`;
  await api(url, { method: "PUT", body }); await refreshState(); await refreshMemoryDialog();
}

async function editMemoryInline(row, valueNode, actions, editButton, scope, projectId, entry) {
  if (editButton.dataset.editing === "true") {
    const textarea = row.querySelector(".memory-edit-value");
    await mutateMemory(scope, projectId, entry.id, { kind: entry.kind, key: entry.key, value: textarea.value });
    return;
  }
  editButton.dataset.editing = "true"; editButton.textContent = "Сохранить";
  const textarea = document.createElement("textarea"); textarea.className = "memory-edit-value";
  textarea.maxLength = 2000; textarea.rows = 3; textarea.value = entry.value;
  valueNode.replaceWith(textarea); textarea.focus();
  actions.append(button("Отмена", "secondary", () => refreshMemoryDialog().catch(showError)));
}

async function deleteMemory(scope, projectId, memoryId) {
  if (!window.confirm("Удалить эту запись памяти? Снимки ранее использованной памяти останутся.")) return;
  const url = scope === "project" ? `/api/projects/${projectId}/memory/${memoryId}` : `/api/memory/user/${memoryId}`;
  await api(url, { method: "DELETE" }); await refreshState(); await refreshMemoryDialog();
}

async function addMemory(event, scope) {
  event.preventDefault(); const form = event.currentTarget;
  const body = { kind: form.elements.kind.value, key: form.elements.key.value, value: form.elements.value.value };
  const url = scope === "project" ? `/api/projects/${activeConversation.project_id}/memory` : "/api/memory/user";
  await api(url, { method: "POST", body }); form.reset(); await refreshMemoryDialog();
}

async function createMemoryProject() {
  const data = await api("/api/projects", { method: "POST", body: { name: "Новый проект", description: "" } });
  await refreshState(); await api(`/api/conversations/${activeConversation.id}/project`, { method: "PATCH", body: { project_id: data.project.id } });
  await openConversation(activeConversation.id); await refreshMemoryDialog();
}

async function changeMemoryProject() {
  await api(`/api/conversations/${activeConversation.id}/project`, { method: "PATCH", body: { project_id: ui.memoryProjectSelect.value || null } });
  await openConversation(activeConversation.id); await refreshState(); await refreshMemoryDialog();
}

async function saveMemoryProject() {
  await api(`/api/projects/${activeConversation.project_id}`, { method: "PATCH", body: {
    name: ui.memoryProjectName.value, description: ui.memoryProjectDescription.value, automatic_extraction: ui.memoryProjectAuto.checked,
  }}); await refreshState(); await refreshMemoryDialog();
}

async function changeUserExtraction() {
  await api("/api/memory/user", { method: "POST", body: { automatic_extraction: ui.memoryUserAuto.checked } });
  await loadUserMemory();
}

async function loadInvariantContext() {
  if (!activeConversation) return;
  invariantContext = await api(`/api/invariants/context/${activeConversation.id}`);
  renderInvariantContext();
}

function invariantOwner(scope) {
  if (scope === "user") return appState.activeProfileId;
  if (scope === "project") return activeConversation?.project_id || "";
  return activeConversation?.id || "";
}

function resetInvariantEditor() {
  ui.invariantSetId.value = "";
  ui.invariantName.value = "";
  ui.invariantRules.value = "";
  ui.invariantEnabled.checked = true;
  ui.invariantError.textContent = "";
  ui.invariantSave.textContent = "Сохранить набор";
}

function renderInvariantContext() {
  const bundle = invariantContext.bundle || { layers: {}, rules: [] };
  const sets = invariantContext.sets || { user: [], project: [], task: [] };
  ui.invariantsCount.textContent = `${(bundle.rules || []).length} активных`;
  const projectOption = ui.invariantScope.querySelector('option[value="project"]');
  projectOption.disabled = !activeConversation?.project_id;
  if (ui.invariantScope.value === "project" && projectOption.disabled) ui.invariantScope.value = "task";
  ui.invariantLayers.replaceChildren();
  const labels = { system: "Системная политика · только чтение", user: "Профиль пользователя", project: "Текущий проект", task: "Конкретная задача" };
  ["system", "user", "project", "task"].forEach((scope) => {
    const section = document.createElement("section"); section.className = "invariant-layer";
    const title = document.createElement("strong"); title.textContent = labels[scope]; section.append(title);
    if (scope === "system") {
      const rules = bundle.layers?.system || [];
      const row = document.createElement("div"); row.className = "invariant-set-row";
      const list = document.createElement("ul"); rules.forEach((rule) => { const li = document.createElement("li"); li.textContent = rule.text; list.append(li); });
      row.append(list); section.append(row);
      if (!rules.length) section.append(memoryEmpty("Правил нет."));
    } else {
      (sets[scope] || []).forEach((item) => section.append(renderInvariantSet(item)));
      if (!(sets[scope] || []).length) section.append(memoryEmpty(scope === "project" && !activeConversation?.project_id ? "Диалог не входит в проект." : "Наборов пока нет."));
    }
    ui.invariantLayers.append(section);
  });
}

function renderInvariantSet(item) {
  const row = document.createElement("article"); row.className = `invariant-set-row${item.enabled === false ? " disabled" : ""}`;
  const header = document.createElement("header");
  const name = document.createElement("strong"); name.textContent = item.name;
  const state = document.createElement("span"); state.textContent = item.enabled === false ? "выключен" : "активен";
  header.append(name, state);
  const list = document.createElement("ul");
  (item.rules || []).forEach((rule) => { const li = document.createElement("li"); li.textContent = rule.text; list.append(li); });
  const actions = document.createElement("div"); actions.className = "invariant-set-buttons";
  actions.append(button("Изменить", "secondary", () => editInvariantSet(item)));
  actions.append(button(item.enabled === false ? "Включить" : "Отключить", "secondary", () => toggleInvariantSet(item).catch(showError)));
  actions.append(button("Удалить", "danger", () => deleteInvariantSet(item).catch(showError)));
  row.append(header, list, actions); return row;
}

function editInvariantSet(item) {
  ui.invariantScope.value = item.scope;
  ui.invariantSetId.value = item.id;
  ui.invariantName.value = item.name;
  ui.invariantRules.value = (item.rules || []).map((rule) => rule.text).join("\n");
  ui.invariantEnabled.checked = item.enabled !== false;
  ui.invariantSave.textContent = "Обновить набор";
  ui.invariantName.focus();
}

function invariantBody(item = null) {
  const scope = item?.scope || ui.invariantScope.value;
  return {
    name: item?.name || ui.invariantName.value,
    description: item?.description || "",
    scope,
    owner_id: item?.owner_id || invariantOwner(scope),
    enabled: item ? item.enabled !== false : ui.invariantEnabled.checked,
    rules: item?.rules || ui.invariantRules.value.split("\n").map((text) => text.trim()).filter(Boolean),
  };
}

async function saveInvariantSet() {
  if (!activeConversation) return;
  ui.invariantError.textContent = "";
  const id = ui.invariantSetId.value;
  try {
    await api(id ? `/api/invariants/${id}` : "/api/invariants", { method: id ? "PUT" : "POST", body: invariantBody() });
    resetInvariantEditor(); await loadInvariantContext();
  } catch (error) { ui.invariantError.textContent = error.message; }
}

async function toggleInvariantSet(item) {
  const body = invariantBody({ ...item, enabled: item.enabled === false });
  await api(`/api/invariants/${item.id}`, { method: "PUT", body }); await loadInvariantContext();
}

async function deleteInvariantSet(item) {
  if (!window.confirm(`Удалить набор «${item.name}»?`)) return;
  await api(`/api/invariants/${item.id}`, { method: "DELETE" }); resetInvariantEditor(); await loadInvariantContext();
}

function showMemorySnapshot(snapshot) {
  ui.memorySnapshotContent.replaceChildren(
    memoryInfo("Профиль", snapshot.profile_snapshot?.profile_name || "Основной пользователь"),
    memoryInfo("Стратегия краткосрочного контекста", snapshot.context_mode || "—"),
  );
  const addGroup = (title, items, formatter) => {
    const section = document.createElement("section"); section.className = "memory-group";
    const heading = document.createElement("h3"); heading.textContent = title; section.append(heading);
    (items || []).forEach((item) => section.append(memoryInfo(formatter(item), item.value || item.text || "")));
    if (!(items || []).length) section.append(memoryEmpty("Не использовалась."));
    ui.memorySnapshotContent.append(section);
  };
  addGroup("Политика", snapshot.policy_rules, (item) => item.id || "Правило");
  addGroup("Память проекта", snapshot.project_memories, (item) => `${item.kind} · ${item.key} · ${item.review_status === "confirmed" ? "подтверждено" : "не подтверждено"}`);
  addGroup("Память пользователя", snapshot.user_memories, (item) => `${item.kind} · ${item.key} · ${item.review_status === "confirmed" ? "подтверждено" : "не подтверждено"}`);
  ui.memorySnapshotDialog.showModal();
}

async function api(url, options = {}) {
  const init = { method: options.method || "GET", headers: {} };
  if (options.body !== undefined) { init.headers["Content-Type"] = "application/json"; init.body = JSON.stringify(options.body); }
  const response = await fetch(url, init); const data = await response.json().catch(() => ({}));
  if (!response.ok || data.ok === false) { const error = new Error(data.error || `Ошибка HTTP ${response.status}`); error.data = data; throw error; }
  return data;
}

function formatDate(value, withTime = false) { if (!value) return ""; const date = new Date(value); return date.toLocaleString("ru-RU", withTime ? { dateStyle: "short", timeStyle: "short" } : { day: "2-digit", month: "2-digit" }); }
function showError(error) { ui.error.textContent = error.message || String(error); }
function showFatal(error) { ui.messages.textContent = `Не удалось запустить интерфейс: ${error.message || error}`; }
function resizeInput() { ui.input.style.height = "auto"; ui.input.style.height = `${Math.min(ui.input.scrollHeight, 180)}px`; }

ui.newChat.addEventListener("click", () => createConversation().catch(showError));
ui.profileSelect.addEventListener("change", () => changeActiveProfile().catch(showError));
ui.profileCreate.addEventListener("click", () => openProfileDialog());
ui.profileEdit.addEventListener("click", () => openProfileDialog(profileById(appState.activeProfileId)));
ui.profileSave.addEventListener("click", () => saveProfile());
ui.projectShareClose.addEventListener("click", () => ui.projectShareDialog.close());
ui.projectShareSave.addEventListener("click", () => saveProjectShare());
ui.newProject.addEventListener("click", () => { ui.newProjectForm.hidden = false; ui.newProjectName.focus(); });
ui.newProjectCancel.addEventListener("click", () => { ui.newProjectForm.hidden = true; ui.newProjectName.value = ""; });
ui.newProjectForm.addEventListener("submit", (event) => createSidebarProject(event).catch(showError));
ui.form.addEventListener("submit", submitMessage); ui.input.addEventListener("input", resizeInput);
ui.voiceButton.addEventListener("click", toggleVoiceRecording);
ui.input.addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); ui.form.requestSubmit(); } });
ui.chatTitle.addEventListener("change", renameChat); ui.chatTitle.addEventListener("blur", renameChat);
ui.deleteChat.addEventListener("click", () => deleteChat().catch(showError));
ui.exportButton.addEventListener("click", () => { if (activeConversation) window.location.href = `/api/conversations/${activeConversation.id}/export`; });
ui.importButton.addEventListener("click", () => ui.importFile.click()); ui.importFile.addEventListener("change", () => { if (ui.importFile.files[0]) importFile(ui.importFile.files[0]); });
ui.settingsToggle.addEventListener("click", () => ui.settingsPanel.classList.add("open")); ui.settingsClose.addEventListener("click", () => ui.settingsPanel.classList.remove("open"));
ui.contextMode.addEventListener("change", changeContextMode);
ui.windowExchanges.addEventListener("change", changeContextMode);
ui.taskStateSave.addEventListener("click", () => saveTaskState());
ui.taskActivityToggle.addEventListener("click", () => toggleTaskActivity());
ui.taskAutopilotStop.addEventListener("click", () => stopAutopilot().catch(showError));
ui.taskStateShow.addEventListener("click", () => showTaskStateReport());
ui.taskTransitionMode.addEventListener("change", () => {
  ui.taskStateNote.textContent = ui.taskTransitionMode.value === "automatic"
    ? "После сохранения автопилот сможет переходить между этапами и отправлять продолжения."
    : "После сохранения переходы выполняются вручную разрешёнными кнопками.";
});
ui.invariantSave.addEventListener("click", () => saveInvariantSet());
ui.invariantCancel.addEventListener("click", () => resetInvariantEditor());
ui.invariantScope.addEventListener("change", () => resetInvariantEditor());
ui.factForm.addEventListener("submit", addManualFact);
ui.factsHelpButton.addEventListener("click", () => ui.factsHelpDialog.showModal());
ui.memoryButton.addEventListener("click", () => openMemoryDialog().catch(showError));
ui.memoryClose.addEventListener("click", () => ui.memoryDialog.close());
ui.memorySnapshotClose.addEventListener("click", () => ui.memorySnapshotDialog.close());
document.querySelectorAll(".memory-tab").forEach((tab) => tab.addEventListener("click", () => {
  document.querySelectorAll(".memory-tab").forEach((item) => item.classList.toggle("active", item === tab));
  document.querySelectorAll(".memory-pane").forEach((pane) => pane.classList.toggle("active", pane.dataset.pane === tab.dataset.tab));
}));
ui.memoryNewProject.addEventListener("click", () => createMemoryProject().catch(showError));
ui.memoryProjectSelect.addEventListener("change", () => changeMemoryProject().catch(showError));
ui.memorySaveProject.addEventListener("click", () => saveMemoryProject().catch(showError));
ui.memoryProjectForm.addEventListener("submit", (event) => addMemory(event, "project").catch(showError));
ui.memoryUserForm.addEventListener("submit", (event) => addMemory(event, "user").catch(showError));
ui.memoryUserAuto.addEventListener("change", () => changeUserExtraction().catch(showError));
ui.presetSelect.addEventListener("change", () => {
  selectPreset(ui.presetSelect.value);
  if (!ui.presetSelect.value) ui.settingsPanel.classList.add("open");
  else ui.settingsPanel.classList.remove("open");
});
ui.resetCustom.addEventListener("click", () => { selectedPresetId = ""; ui.presetSelect.value = ""; applySettings(appState.defaults); });
ui.saveAsPreset.addEventListener("click", () => openPresetDialog(true)); ui.presetsButton.addEventListener("click", () => openPresetDialog());
ui.presetSave.addEventListener("click", savePreset); ui.presetCancelEdit.addEventListener("click", clearPresetEditor);
document.querySelectorAll("#settings-panel input, #settings-panel textarea, #settings-panel select").forEach((control) => control.addEventListener("input", updateSettingsSummary));
