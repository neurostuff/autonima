const { useEffect, useMemo, useRef, useState } = React;
const DEFAULT_ANNOTATION_METADATA_FIELDS = [
  "analysis_name",
  "analysis_description",
  "table_caption",
  "study_title",
  "study_fulltext",
];
const BUILD_STEPS = [
  ["search", "Find studies"],
  ["screening", "Screening"],
  ["parsing_annotation", "Parsing + Annotation"],
  ["review", "Review"],
];

function formatApiError(payload) {
  if (payload == null) {
    return "Request failed";
  }
  if (typeof payload === "string") {
    return payload;
  }
  if (Array.isArray(payload)) {
    const parts = payload
      .map((item) => formatApiError(item))
      .filter(Boolean);
    return parts.join(" | ") || "Request failed";
  }
  if (typeof payload === "object") {
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
    if (payload.detail != null) {
      return formatApiError(payload.detail);
    }
    if (typeof payload.message === "string") {
      return payload.message;
    }
    if (typeof payload.msg === "string") {
      const loc = Array.isArray(payload.loc) ? payload.loc.join(".") : null;
      return loc ? `${loc}: ${payload.msg}` : payload.msg;
    }
    try {
      return JSON.stringify(payload);
    } catch (_) {
      return String(payload);
    }
  }
  return String(payload);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const detail = formatApiError(payload);
    throw new Error(detail);
  }
  return payload;
}

function setNested(obj, path, value) {
  const copy = JSON.parse(JSON.stringify(obj || {}));
  let cursor = copy;
  for (let i = 0; i < path.length - 1; i += 1) {
    const key = path[i];
    if (typeof cursor[key] !== "object" || cursor[key] === null) {
      cursor[key] = {};
    }
    cursor = cursor[key];
  }
  cursor[path[path.length - 1]] = value;
  return copy;
}

function getNested(obj, path, defaultValue = "") {
  let cursor = obj;
  for (const key of path) {
    if (!cursor || typeof cursor !== "object") {
      return defaultValue;
    }
    cursor = cursor[key];
  }
  return cursor ?? defaultValue;
}

function parseLines(value) {
  return String(value || "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

function stringifyLines(value) {
  if (Array.isArray(value)) {
    return value.join("\n");
  }
  return "";
}

function pickPreferredDefaultModel(models, preferred) {
  if (!Array.isArray(models) || !models.length) {
    return "";
  }
  if (typeof preferred === "string" && models.includes(preferred)) {
    return preferred;
  }
  return models[0] || "";
}

function formatBytes(value) {
  const size = Number(value || 0);
  if (!Number.isFinite(size) || size < 1024) {
    return `${Math.max(0, Math.round(size))} B`;
  }
  const units = ["KB", "MB", "GB", "TB"];
  let current = size / 1024;
  let unitIndex = 0;
  while (current >= 1024 && unitIndex < units.length - 1) {
    current /= 1024;
    unitIndex += 1;
  }
  return `${current.toFixed(current >= 10 ? 0 : 1)} ${units[unitIndex]}`;
}

function buildMetaArtifactUrl(runId, relativePath) {
  const normalizedRunId = String(runId || "").trim();
  const normalizedPath = String(relativePath || "").trim();
  if (!normalizedRunId || !normalizedPath) return "";
  return `/api/runs/${encodeURIComponent(normalizedRunId)}/meta-artifact?path=${encodeURIComponent(normalizedPath)}`;
}

function metaArtifactGroupId(file) {
  const relativePath = String(file?.relative_path || "").trim();
  const parts = relativePath.split("/").filter(Boolean);
  return parts.length > 1 ? parts[0] : "maps";
}

function metaArtifactGroupLabel(groupId) {
  return formatCounterLabel(String(groupId || "maps").replaceAll("-", "_"));
}

function metaArtifactSortRank(file) {
  const name = String(file?.name || "").toLowerCase();
  if (name.startsWith("z_corr")) return 0;
  if (name === "z.nii" || name === "z.nii.gz") return 1;
  if (name.startsWith("z")) return 2;
  return 10;
}

function metaArtifactGroupSortRank(groupId) {
  const normalized = String(groupId || "").toLowerCase();
  if (normalized.startsWith("all_") || normalized === "maps") return 1;
  return 0;
}

function sortMetaArtifacts(files) {
  return [...(files || [])].sort((a, b) => {
    const groupA = metaArtifactGroupId(a);
    const groupB = metaArtifactGroupId(b);
    const groupRankCompare = metaArtifactGroupSortRank(groupA) - metaArtifactGroupSortRank(groupB);
    if (groupRankCompare) return groupRankCompare;
    const groupCompare = groupA.localeCompare(groupB);
    if (groupCompare) return groupCompare;
    const rankCompare = metaArtifactSortRank(a) - metaArtifactSortRank(b);
    if (rankCompare) return rankCompare;
    return String(a?.name || "").localeCompare(String(b?.name || ""));
  });
}

function preferredMetaArtifact(files) {
  const sorted = sortMetaArtifacts(files);
  return sorted[0] || null;
}

function groupMetaArtifacts(files) {
  const groups = new Map();
  for (const file of sortMetaArtifacts(files)) {
    const groupId = metaArtifactGroupId(file);
    if (!groups.has(groupId)) {
      groups.set(groupId, {
        id: groupId,
        label: metaArtifactGroupLabel(groupId),
        files: [],
      });
    }
    groups.get(groupId).files.push(file);
  }
  return Array.from(groups.values());
}

function buildMissingFulltextsUrl(runId, extension) {
  const normalizedRunId = String(runId || "").trim();
  const normalizedExt = String(extension || "").trim().toLowerCase();
  if (!normalizedRunId || !["txt", "csv"].includes(normalizedExt)) return "";
  return `/api/runs/${encodeURIComponent(normalizedRunId)}/missing-fulltexts.${normalizedExt}`;
}

function statusClass(status) {
  return `badge ${status || "queued"}`;
}

function isActiveRunStatus(status) {
  return ["queued", "running", "canceling"].includes(String(status || "").toLowerCase());
}

function runTimestampValue(run) {
  const timestamp = Date.parse(run?.created_at || run?.started_at || run?.updated_at || "");
  return Number.isFinite(timestamp) ? timestamp : 0;
}

function newestRunsFirst(runsList) {
  return [...(runsList || [])].sort((a, b) => runTimestampValue(b) - runTimestampValue(a));
}

function formatCounterLabel(key) {
  return String(key || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function counterTone(key) {
  const normalized = String(key || "").toLowerCase();
  if (normalized.includes("included") || normalized === "decisions") return "include";
  if (normalized.includes("excluded")) return "exclude";
  if (normalized.includes("incomplete")) return "incomplete";
  return "total";
}

function toDateInputValue(value) {
  const raw = String(value || "").trim();
  if (!raw) return "";
  if (/^\d{4}-\d{2}-\d{2}$/.test(raw)) return raw;
  if (/^\d{4}\/\d{2}\/\d{2}$/.test(raw)) return raw.replaceAll("/", "-");
  return "";
}

function fromDateInputValue(value) {
  const raw = String(value || "").trim();
  if (!raw) return "";
  return raw.replaceAll("-", "/");
}

function MetaNiftiViewer({ fileUrl, fileName }) {
  const canvasRef = useRef(null);
  const viewerRef = useRef(null);
  const [viewerError, setViewerError] = useState("");
  const [viewerReady, setViewerReady] = useState(false);

  useEffect(() => {
    if (!fileUrl) {
      setViewerError("");
      setViewerReady(false);
      return;
    }

    let canceled = false;

    async function loadVolume() {
      try {
        setViewerError("");
        setViewerReady(false);

        const niivueGlobal = window.niivue;
        const NiivueCtor = niivueGlobal?.Niivue;
        if (!NiivueCtor) {
          throw new Error("NiiVue library was not loaded.");
        }
        if (!canvasRef.current) {
          throw new Error("Viewer canvas is unavailable.");
        }

        if (!viewerRef.current) {
          const nextViewer = new NiivueCtor({
            show3Dcrosshair: true,
          });
          await nextViewer.attachToCanvas(canvasRef.current);
          if (niivueGlobal?.SHOW_RENDER?.ALWAYS != null) {
            nextViewer.opts.multiplanarShowRender = niivueGlobal.SHOW_RENDER.ALWAYS;
          }
          nextViewer.opts.isColorbar = true;
          nextViewer.setSliceMM(false);
          viewerRef.current = nextViewer;
        }

        const viewer = viewerRef.current;
        while ((viewer.volumes || []).length) {
          viewer.removeVolume(viewer.volumes[viewer.volumes.length - 1]);
        }

        await viewer.addVolumeFromUrl({
          url: "https://neurovault.org/static/images/GenericMNI.nii.gz",
          colormap: "gray",
          opacity: 1,
          colorbarVisible: false,
        });

        await viewer.addVolumeFromUrl({
          url: fileUrl,
          colormap: "warm",
          opacity: 1,
          cal_min: 0,
          cal_max: 6,
          cal_minNeg: -6,
          cal_maxNeg: 0,
        });

        viewer.setInterpolation(true);
        viewer.updateGLVolume();

        if (!canceled) {
          setViewerReady(true);
        }
      } catch (err) {
        if (!canceled) {
          setViewerError(err?.message || String(err));
          setViewerReady(false);
        }
      }
    }

    loadVolume();
    return () => {
      canceled = true;
    };
  }, [fileUrl, fileName]);

  return (
    <div className="meta-live-viewer-shell">
      {viewerError ? (
        <div className="status-msg error">{viewerError}</div>
      ) : null}
      {!viewerReady && !viewerError ? (
        <div className="status-msg">Loading viewer…</div>
      ) : null}
      <div className="meta-live-viewer-canvas-wrap">
        <canvas ref={canvasRef} className="meta-live-viewer-canvas" />
      </div>
      <div className="meta-live-viewer-actions">
        <a className="secondary" href={fileUrl} target="_blank" rel="noreferrer">
          Download NIfTI
        </a>
      </div>
    </div>
  );
}

function App() {
  const [view, setView] = useState("projects");
  const [editorTab, setEditorTab] = useState("build");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [workspace, setWorkspace] = useState(null);
  const [projects, setProjects] = useState([]);
  const [selectedProjectId, setSelectedProjectId] = useState(null);

  const [specForm, setSpecForm] = useState({});
  const [yamlText, setYamlText] = useState("");
  const [yamlMode, setYamlMode] = useState(false);
  const [buildStep, setBuildStep] = useState("search");
  const [runsSubTab, setRunsSubTab] = useState("screening");
  const [specPath, setSpecPath] = useState("");

  const [runs, setRuns] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState(null);
  const [selectedRun, setSelectedRun] = useState(null);
  const [metaArtifacts, setMetaArtifacts] = useState([]);
  const [metaArtifactsLoading, setMetaArtifactsLoading] = useState(false);
  const [metaArtifactsError, setMetaArtifactsError] = useState("");
  const [selectedMetaArtifactPath, setSelectedMetaArtifactPath] = useState("");
  const [selectedMetaArtifactGroup, setSelectedMetaArtifactGroup] = useState("");
  const [logs, setLogs] = useState([]);
  const [logOffset, setLogOffset] = useState(0);
  const logOffsetRef = useRef(0);
  const yamlEditorRef = useRef(null);

  const [secrets, setSecrets] = useState({});
  const [maskedSecrets, setMaskedSecrets] = useState({});
  const [preferredModels, setPreferredModels] = useState([]);
  const [preferredModelsText, setPreferredModelsText] = useState("");
  const [globalPreferredModel, setGlobalPreferredModel] = useState("");
  const [yamlModelSelection, setYamlModelSelection] = useState("");
  const [searchAdvancedTouched, setSearchAdvancedTouched] = useState({
    maxResults: false,
    email: false,
  });

  const [statusMsg, setStatusMsg] = useState(null);
  const [deletePreview, setDeletePreview] = useState(null);
  const [deleteMode, setDeleteMode] = useState("metadata_only");
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [createBusy, setCreateBusy] = useState(false);
  const [createName, setCreateName] = useState("");
  const [createDescription, setCreateDescription] = useState("");
  const [importModalOpen, setImportModalOpen] = useState(false);
  const [importBusy, setImportBusy] = useState(false);
  const [importConfigPath, setImportConfigPath] = useState("");
  const [importName, setImportName] = useState("");
  const [importDescription, setImportDescription] = useState("");
  const [editModalOpen, setEditModalOpen] = useState(false);
  const [editBusy, setEditBusy] = useState(false);
  const [editProjectId, setEditProjectId] = useState("");
  const [editName, setEditName] = useState("");
  const [editDescription, setEditDescription] = useState("");
  const [cloneModalOpen, setCloneModalOpen] = useState(false);
  const [cloneBusy, setCloneBusy] = useState(false);
  const [cloneProjectId, setCloneProjectId] = useState("");
  const [cloneName, setCloneName] = useState("");
  const [cloneDescription, setCloneDescription] = useState("");
  const [cloneMode, setCloneMode] = useState("schema_only");
  const [projectModelSettingsOpen, setProjectModelSettingsOpen] = useState(false);
  const [studyListModalOpen, setStudyListModalOpen] = useState(false);
  const [studyListBusy, setStudyListBusy] = useState(false);
  const [studyListFileName, setStudyListFileName] = useState("");
  const [pubmedCountBusy, setPubmedCountBusy] = useState(false);
  const [pubmedCount, setPubmedCount] = useState(null);
  const [sourceModalOpen, setSourceModalOpen] = useState(false);
  const [sourceModalIndex, setSourceModalIndex] = useState(null);
  const [sourceType, setSourceType] = useState("custom");
  const [sourceForm, setSourceForm] = useState({
    root_path: "",
    pmid_source: "folder_name",
    text_path_templates: "",
    coordinates_path_templates: "",
    allowed_extensions: "",
    processed_data_path: "",
    json_filename: "",
    json_pmid_key: "",
  });

  const [runForm, setRunForm] = useState({
    mode: "run",
    output_folder: "",
    verbose: false,
    dry_run: false,
    debug: false,
    num_workers: 1,
    force_reextract_incomplete_fulltext: false,
    apply_default_email: true,
  });

  const [metaForm, setMetaForm] = useState({
    output_folder: "",
    estimator: "mkdadensity",
    estimator_args: "{}",
    corrector: "fdr",
    corrector_args: "{}",
    include_ids: "",
    run_reports: false,
    fail_fast: false,
    debug: false,
  });

  const selectedProject = useMemo(
    () => projects.find((project) => project.id === selectedProjectId) || null,
    [projects, selectedProjectId]
  );
  const buildStepIndex = useMemo(
    () => Math.max(0, BUILD_STEPS.findIndex(([id]) => id === buildStep)),
    [buildStep]
  );
  const isFirstBuildStep = buildStepIndex <= 0;
  const isLastBuildStep = buildStepIndex >= BUILD_STEPS.length - 1;
  const eligibleMetaRun = useMemo(
    () => {
      const candidates = [];
      if (selectedRun) candidates.push(selectedRun);
      for (const run of runs) {
        if (!candidates.some((item) => item?.id && item.id === run?.id)) {
          candidates.push(run);
        }
      }

      const completedRuns = candidates.filter(
        (run) => run?.status === "completed" && Boolean(run.output_folder)
      );
      if (!completedRuns.length) {
        return null;
      }

      const preferredCompletedScreeningRun = completedRuns.find((run) => {
        const outputStage = (run.progress?.timeline || []).find(
          (stage) => stage.stage === "output"
        );
        const isMetaRun = run.kind === "meta" || run.mode === "meta";
        const isScreeningRun = !isMetaRun;
        const hasNimads = Boolean(
          run.progress?.nimads_available || run.progress?.nimads_export_logged
        );
        return isScreeningRun && outputStage?.status === "completed" && hasNimads;
      });

      return preferredCompletedScreeningRun || completedRuns[0];
    },
    [runs, selectedRun]
  );
  const metaAnalysisEnabled = Boolean(eligibleMetaRun);
  const selectedOutputFolder = selectedRun?.output_folder || eligibleMetaRun?.output_folder || runForm.output_folder || "";
  const activeScreeningRun = useMemo(
    () => {
      const candidates = [];
      if (selectedRun) candidates.push(selectedRun);
      for (const run of runs) {
        if (!candidates.some((item) => item?.id && item.id === run?.id)) {
          candidates.push(run);
        }
      }
      return candidates.find((run) => {
        const isMetaRun = run?.kind === "meta" || run?.mode === "meta";
        return !isMetaRun && isActiveRunStatus(run?.status);
      }) || null;
    },
    [runs, selectedRun]
  );
  const activeMetaRun = useMemo(
    () => {
      const candidates = [];
      if (selectedRun) candidates.push(selectedRun);
      for (const run of runs) {
        if (!candidates.some((item) => item?.id && item.id === run?.id)) {
          candidates.push(run);
        }
      }
      return candidates.find((run) => {
        const isMetaRun = run?.kind === "meta" || run?.mode === "meta";
        return isMetaRun && isActiveRunStatus(run?.status);
      }) || null;
    },
    [runs, selectedRun]
  );
  const screeningRunInProgress = Boolean(activeScreeningRun);
  const metaRunInProgress = Boolean(activeMetaRun);
  const runsForActiveTab = useMemo(
    () => newestRunsFirst(runs.filter((run) => {
      const isMetaRun = run?.kind === "meta" || run?.mode === "meta";
      return runsSubTab === "meta" ? isMetaRun : !isMetaRun;
    })),
    [runs, runsSubTab]
  );
  const selectedRunForActiveTab = useMemo(
    () => runsForActiveTab.find((run) => run.id === selectedRunId) || null,
    [runsForActiveTab, selectedRunId]
  );
  const activeRunForActiveTab = runsSubTab === "meta" ? activeMetaRun : activeScreeningRun;
  const currentExecutionRun = useMemo(
    () => activeRunForActiveTab || selectedRunForActiveTab || runsForActiveTab[0] || null,
    [activeRunForActiveTab, selectedRunForActiveTab, runsForActiveTab]
  );
  const previousRunsForActiveTab = useMemo(
    () => runsForActiveTab.filter((run) => run.id !== currentExecutionRun?.id),
    [runsForActiveTab, currentExecutionRun?.id]
  );
  const viewingPreviousRun = Boolean(
    currentExecutionRun
      && selectedRunForActiveTab
      && currentExecutionRun.id === selectedRunForActiveTab.id
      && !activeRunForActiveTab
      && runsForActiveTab[0]?.id !== currentExecutionRun.id
  );
  const selectedMetaRun = useMemo(
    () => (runsSubTab === "meta" ? currentExecutionRun : null),
    [runsSubTab, currentExecutionRun]
  );
  const resolvedMetaRun = useMemo(
    () => {
      if (runsSubTab !== "meta") return null;
      return selectedMetaRun;
    },
    [runsSubTab, selectedMetaRun]
  );
  const metaArtifactGroups = useMemo(
    () => groupMetaArtifacts(metaArtifacts),
    [metaArtifacts]
  );
  const activeMetaArtifactGroup = useMemo(
    () => {
      if (metaArtifactGroups.some((group) => group.id === selectedMetaArtifactGroup)) {
        return selectedMetaArtifactGroup;
      }
      const selectedFile = metaArtifacts.find((item) => item.relative_path === selectedMetaArtifactPath);
      if (selectedFile) {
        return metaArtifactGroupId(selectedFile);
      }
      return metaArtifactGroups[0]?.id || "";
    },
    [metaArtifactGroups, metaArtifacts, selectedMetaArtifactGroup, selectedMetaArtifactPath]
  );
  const activeMetaArtifacts = useMemo(
    () => metaArtifactGroups.find((group) => group.id === activeMetaArtifactGroup)?.files || [],
    [metaArtifactGroups, activeMetaArtifactGroup]
  );
  const selectedMetaArtifact = useMemo(
    () => activeMetaArtifacts.find((item) => item.relative_path === selectedMetaArtifactPath)
      || preferredMetaArtifact(activeMetaArtifacts)
      || null,
    [activeMetaArtifacts, selectedMetaArtifactPath]
  );
  const selectedMetaArtifactUrl = useMemo(
    () => buildMetaArtifactUrl(resolvedMetaRun?.id, selectedMetaArtifact?.relative_path),
    [resolvedMetaRun?.id, selectedMetaArtifact?.relative_path]
  );

  async function refreshWorkspace() {
    const data = await api("/api/workspace");
    setWorkspace(data);
  }

  async function refreshProjects() {
    const data = await api("/api/projects");
    const nextProjects = data.projects || [];
    setProjects(nextProjects);
    setSelectedProjectId((current) => {
      if (!current) return null;
      const stillExists = nextProjects.some((item) => item.id === current);
      return stillExists ? current : null;
    });
  }

  async function refreshRuns() {
    const projectQuery = selectedProjectId ? `?project_id=${encodeURIComponent(selectedProjectId)}` : "";
    const data = await api(`/api/runs${projectQuery}`);
    setRuns(data.runs || []);
  }

  async function loadSpec(projectId) {
    if (!projectId) return;
    const data = await api(`/api/projects/${projectId}/spec`);
    setSpecForm(data.form || {});
    setYamlText(data.yaml_text || "");
    setSpecPath(data.config_path || "");
    setSearchAdvancedTouched({ maxResults: false, email: false });
    setMetaForm((prev) => ({
      ...prev,
      output_folder: data.form?.output?.directory || prev.output_folder,
    }));
  }

  async function refreshSecrets() {
    const data = await api("/api/settings/secrets");
    setSecrets(data.values || {});
    setMaskedSecrets(data.masked || {});
  }

  async function refreshPreferences() {
    const data = await api("/api/settings/preferences");
    const models = Array.isArray(data?.preferred_models) ? data.preferred_models : [];
    const preferredDefault = typeof data?.default_model === "string" ? data.default_model : "";
    const resolvedDefault = pickPreferredDefaultModel(models, preferredDefault);
    setPreferredModels(models);
    setPreferredModelsText(stringifyLines(models));
    setGlobalPreferredModel(resolvedDefault);
    setYamlModelSelection((prev) => {
      if (models.includes(prev)) return prev;
      return resolvedDefault || "";
    });
  }

  async function saveGlobalModelPreference(nextModel) {
    const normalized = pickPreferredDefaultModel(preferredModels, nextModel);
    setGlobalPreferredModel(normalized);
    try {
      const data = await api("/api/settings/preferences", {
        method: "PUT",
        body: JSON.stringify({ default_model: normalized }),
      });
      const models = Array.isArray(data?.preferred_models) ? data.preferred_models : [];
      const preferredDefault = typeof data?.default_model === "string" ? data.default_model : "";
      const resolvedDefault = pickPreferredDefaultModel(models, preferredDefault);
      setPreferredModels(models);
      setPreferredModelsText(stringifyLines(models));
      setGlobalPreferredModel(resolvedDefault);
      setYamlModelSelection((prev) => (models.includes(prev) ? prev : (resolvedDefault || "")));
    } catch (err) {
      setStatusMsg({ type: "error", text: err.message });
    }
  }

  function insertIntoYaml(textToInsert) {
    const editor = yamlEditorRef.current;
    if (!editor) {
      setYamlText((prev) => `${prev}${textToInsert}`);
      return;
    }

    const start = editor.selectionStart || 0;
    const end = editor.selectionEnd || 0;
    const current = yamlText || "";
    const nextText = `${current.slice(0, start)}${textToInsert}${current.slice(end)}`;
    setYamlText(nextText);

    window.requestAnimationFrame(() => {
      const caret = start + textToInsert.length;
      editor.focus();
      editor.setSelectionRange(caret, caret);
    });
  }

  function insertSelectedModelValue() {
    if (!yamlModelSelection) return;
    insertIntoYaml(`"${yamlModelSelection}"`);
  }

  function insertSelectedModelLine() {
    if (!yamlModelSelection) return;
    const editor = yamlEditorRef.current;
    const cursor = editor?.selectionStart || 0;
    const current = yamlText || "";
    const lineStart = current.lastIndexOf("\n", Math.max(0, cursor - 1)) + 1;
    const linePrefix = current.slice(lineStart, cursor);
    const indentMatch = linePrefix.match(/^\s*/);
    const indent = indentMatch ? indentMatch[0] : "";
    insertIntoYaml(`${indent}model: "${yamlModelSelection}"\n`);
  }

  useEffect(() => {
    (async () => {
      try {
        await refreshWorkspace();
        await refreshProjects();
        await refreshSecrets();
        await refreshPreferences();
      } catch (err) {
        setStatusMsg({ type: "error", text: err.message });
      }
    })();
  }, []);

  useEffect(() => {
    if (selectedProjectId) {
      setBuildStep("search");
      loadSpec(selectedProjectId).catch((err) => {
        setStatusMsg({ type: "error", text: err.message });
      });
      refreshRuns().catch((err) => setStatusMsg({ type: "error", text: err.message }));
    }
  }, [selectedProjectId]);

  useEffect(() => {
    if (editorTab !== "runs" || !currentExecutionRun?.id) return;
    if (selectedRunId === currentExecutionRun.id) return;
    setSelectedRunId(currentExecutionRun.id);
    setSelectedRun(currentExecutionRun);
    setLogs([]);
    logOffsetRef.current = 0;
    setLogOffset(0);
  }, [editorTab, currentExecutionRun?.id, selectedRunId]);

  useEffect(() => {
    if (!eligibleMetaRun?.output_folder) return;
    setMetaForm((prev) => {
      if (prev.output_folder === eligibleMetaRun.output_folder) return prev;
      return { ...prev, output_folder: eligibleMetaRun.output_folder };
    });
  }, [eligibleMetaRun?.output_folder]);

  useEffect(() => {
    const timer = setInterval(() => {
      if (!selectedProjectId) return;
      refreshRuns().catch(() => {});
    }, 2500);
    return () => clearInterval(timer);
  }, [selectedProjectId]);

  useEffect(() => {
    if (!selectedRunId) return;
    let mounted = true;

    async function tick() {
      try {
        const runData = await api(`/api/runs/${selectedRunId}`);
        if (!mounted) return;
        setSelectedRun(runData);

        const logData = await api(`/api/runs/${selectedRunId}/logs?offset=${logOffsetRef.current}`);
        if (!mounted) return;
        if (Array.isArray(logData.lines) && logData.lines.length) {
          setLogs((prev) => [...prev, ...logData.lines]);
        }
        const nextOffset = logData.next_offset || 0;
        logOffsetRef.current = nextOffset;
        setLogOffset(nextOffset);
      } catch (err) {
        if (mounted) {
          setStatusMsg({ type: "error", text: err.message });
        }
      }
    }

    tick();
    const timer = setInterval(tick, 2000);
    return () => {
      mounted = false;
      clearInterval(timer);
    };
  }, [selectedRunId]);

  useEffect(() => {
    if (!resolvedMetaRun?.id) {
      setMetaArtifacts([]);
      setMetaArtifactsError("");
      setMetaArtifactsLoading(false);
      setSelectedMetaArtifactPath("");
      setSelectedMetaArtifactGroup("");
      return;
    }

    let mounted = true;

    async function refreshMetaArtifacts() {
      try {
        if (mounted) {
          setMetaArtifactsLoading(true);
          setMetaArtifactsError("");
        }
        const data = await api(`/api/runs/${resolvedMetaRun.id}/meta-artifacts`);
        if (!mounted) return;
        const files = sortMetaArtifacts(Array.isArray(data?.files) ? data.files : []);
        setMetaArtifacts(files);
        setSelectedMetaArtifactPath((current) => {
          if (files.some((item) => item.relative_path === current)) {
            return current;
          }
          return preferredMetaArtifact(files)?.relative_path || "";
        });
        setSelectedMetaArtifactGroup((current) => {
          if (files.some((item) => metaArtifactGroupId(item) === current)) {
            return current;
          }
          const preferred = preferredMetaArtifact(files);
          return preferred ? metaArtifactGroupId(preferred) : "";
        });
      } catch (err) {
        if (!mounted) return;
        setMetaArtifacts([]);
        setSelectedMetaArtifactPath("");
        setSelectedMetaArtifactGroup("");
        setMetaArtifactsError(err.message || String(err));
      } finally {
        if (mounted) {
          setMetaArtifactsLoading(false);
        }
      }
    }

    refreshMetaArtifacts();
    const shouldPoll = !["completed", "failed", "canceled"].includes(
      String(resolvedMetaRun.status || "").toLowerCase()
    );
    if (!shouldPoll) {
      return () => {
        mounted = false;
      };
    }

    const timer = setInterval(refreshMetaArtifacts, 5000);
    return () => {
      mounted = false;
      clearInterval(timer);
    };
  }, [resolvedMetaRun?.id, resolvedMetaRun?.status]);

  function updateField(path, value) {
    setSpecForm((prev) => setNested(prev, path, value));
  }

  function getModelFieldValue(path) {
    const explicit = String(getNested(specForm, path, "") || "").trim();
    if (explicit) return explicit;
    const defaultsModel = String(getNested(specForm, ["defaults", "model"], "") || "").trim();
    if (defaultsModel) return defaultsModel;
    return globalPreferredModel || "";
  }

  function renderModelSelect(label, path, optional = true) {
    const value = getModelFieldValue(path);
    const options = Array.from(new Set([
      ...preferredModels,
      ...(value && !preferredModels.includes(value) ? [value] : []),
    ]));
    return (
      <>
        <label>{label}</label>
        <select
          value={value}
          onChange={(e) => updateField(path, e.target.value)}
        >
          {optional ? (
            <option value="">None</option>
          ) : null}
          {options.map((modelName) => (
            <option key={modelName} value={modelName}>
              {modelName}
            </option>
          ))}
        </select>
      </>
    );
  }

  function renderToggleControl(label, checked, onChange, disabled = false) {
    const enabled = Boolean(checked);
    return (
      <div className={`toggle-field ${disabled ? "disabled" : ""}`}>
        <span className="toggle-label">{label}</span>
        <button
          type="button"
          className={`toggle-switch ${enabled ? "on" : ""}`}
          role="switch"
          aria-checked={enabled}
          disabled={disabled}
          onClick={() => onChange(!enabled)}
        >
          <span className="toggle-knob" />
          <span className="toggle-text">{enabled ? "On" : "Off"}</span>
        </button>
      </div>
    );
  }

  function buildFunnelStages(counters, liveProgress = null) {
    const data = counters && typeof counters === "object" ? counters : {};
    const stages = [];
    const addStage = (id, title, width, items) => {
      const filteredItems = items.filter((item) => item.value !== undefined && item.value !== null);
      if (filteredItems.length) {
        stages.push({ id, title, width, items: filteredItems });
      }
    };

    if (data.search) {
      addStage("search", "Search", 100, [
        { key: "studies_found", label: "Studies found", value: data.search.studies_found, tone: "total" },
      ]);
    }

    if (data.abstract) {
      addStage("abstract", "Abstract Screening", 86, [
        { key: "screened", label: "Screened", value: data.abstract.screened, tone: "total" },
        { key: "included", label: "Included", value: data.abstract.included, tone: "include" },
        { key: "excluded", label: "Excluded", value: data.abstract.excluded, tone: "exclude" },
      ]);
    }

    if (data.retrieval) {
      addStage("retrieval", "Full Text Retrieval", 72, [
        { key: "fulltext_candidates", label: "Full-text candidates", value: data.retrieval.fulltext_candidates, tone: "total" },
      ]);
    }

    if (data.fulltext) {
      addStage("fulltext", "Full Text Screening", 58, [
        { key: "screened", label: "Screened", value: data.fulltext.screened, tone: "total" },
        { key: "included", label: "Included", value: data.fulltext.included, tone: "include" },
        { key: "excluded", label: "Excluded", value: data.fulltext.excluded, tone: "exclude" },
        { key: "incomplete", label: "Incomplete", value: data.fulltext.incomplete, tone: "incomplete" },
      ]);
    }

    if (data.annotation) {
      addStage("annotation", "Annotation", 44, [
        { key: "decisions", label: "Decisions", value: data.annotation.decisions, tone: "include" },
      ]);
    }

    if (data.output && typeof data.output === "object") {
      const outputEntries = Object.entries(data.output);
      const finalIncluded =
        outputEntries.find(([key]) => key === "final_included") ||
        outputEntries.find(([key]) => {
          const normalizedKey = String(key || "").toLowerCase();
          return normalizedKey.includes("final") && normalizedKey.includes("included");
        });
      if (finalIncluded) {
        addStage("output", "Final Output", 34, [
          {
            key: finalIncluded[0],
            label: "Final included",
            value: finalIncluded[1],
            tone: "include",
          },
        ]);
      }
    }

    if (liveProgress?.stage && liveProgress.total && !stages.some((stage) => stage.id === liveProgress.stage)) {
      const fallbackStages = {
        abstract: ["Abstract Screening", 86],
        fulltext: ["Full Text Screening", 58],
        parsing: ["Parsing", 50],
        annotation: ["Annotation", 44],
      };
      const [title, width] = fallbackStages[liveProgress.stage] || [formatCounterLabel(liveProgress.stage), 64];
      stages.push({
        id: liveProgress.stage,
        title,
        width,
        items: [],
      });
    }

    if (liveProgress?.stage) {
      for (const stage of stages) {
        if (stage.id === liveProgress.stage) {
          stage.liveProgress = liveProgress;
        }
      }
    }

    return stages;
  }

  function renderCounterFunnel(counters, liveProgress = null) {
    const stages = buildFunnelStages(counters, liveProgress);
    if (!stages.length) {
      return (
        <div className="run-results-empty">
          Results will appear here as Autonima writes stage outputs.
        </div>
      );
    }

    return (
      <div className="run-funnel">
        {stages.map((stage, index) => (
          <div className="run-funnel-stage-wrap" key={stage.id}>
            <div className={`run-funnel-stage ${stage.id}`} style={{ width: `${stage.width}%` }}>
              <div className="run-funnel-title">{stage.title}</div>
              {stage.liveProgress ? (
                <div className="run-funnel-live">
                  <div className="run-funnel-live-top">
                    <span>{stage.liveProgress.label || "Running"}</span>
                    <strong>{stage.liveProgress.current} / {stage.liveProgress.total}</strong>
                  </div>
                  <div className="run-funnel-live-track" aria-label={`${stage.liveProgress.percent}% complete`}>
                    <div
                      className="run-funnel-live-fill"
                      style={{ width: `${Math.max(0, Math.min(100, stage.liveProgress.percent || 0))}%` }}
                    />
                  </div>
                </div>
              ) : null}
              <div className="run-funnel-metrics">
                {stage.items.map((item) => (
                  <div className={`run-funnel-metric ${item.tone}`} key={item.key}>
                    <span className="run-result-value">{item.value}</span>
                    <span className="run-result-label">{item.label}</span>
                  </div>
                ))}
              </div>
            </div>
            {index < stages.length - 1 ? <div className="run-funnel-arrow">↓</div> : null}
          </div>
        ))}
      </div>
    );
  }

  function reviewValue(value) {
    const text = String(value || "").trim();
    return text || "Not set";
  }

  function reviewCount(path) {
    return getCriteriaList(path).length;
  }

  function renderReviewChip(label, tone = "info") {
    return <span className={`review-chip ${tone}`}>{label}</span>;
  }

  function renderReviewRow(label, value) {
    return (
      <div className="review-row">
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    );
  }

  function renderReviewCard(title, children, chips = []) {
    return (
      <div className="review-card">
        <div className="review-card-header">
          <h4>{title}</h4>
          {chips.length ? <div className="review-chip-row">{chips}</div> : null}
        </div>
        <div className="review-card-body">{children}</div>
      </div>
    );
  }

  function renderSpecificationReview() {
    const searchQuery = String(getNested(specForm, ["search", "query"], "") || "");
    const dateFrom = reviewValue(getNested(specForm, ["search", "date_from"], ""));
    const dateTo = reviewValue(getNested(specForm, ["search", "date_to"], ""));
    const retrievalSources = getRetrievalSources();
    const fullTextSources = getFullTextSources();
    const annotationEntries = getNested(specForm, ["annotation", "annotations"], []);
    const annotations = Array.isArray(annotationEntries) ? annotationEntries : [];
    const metadataFields = getNested(specForm, ["annotation", "metadata_fields"], DEFAULT_ANNOTATION_METADATA_FIELDS);
    const metadataFieldCount = Array.isArray(metadataFields) ? metadataFields.length : 0;
    const hasPmidTerms = /\[PMID\]|\bPMID\b/i.test(searchQuery);

    return (
      <div className="review-summary">
        {renderReviewCard(
          "Find Studies",
          <>
            {renderReviewRow("Query", reviewValue(searchQuery))}
            {renderReviewRow("Date range", `${dateFrom} → ${dateTo}`)}
            {renderReviewRow("Database", reviewValue(getNested(specForm, ["search", "database"], "pubmed")))}
          </>,
          [
            renderReviewChip(searchQuery.trim() ? "Query set" : "Query missing", searchQuery.trim() ? "ok" : "muted"),
            hasPmidTerms ? renderReviewChip("PMID list", "info") : null,
          ].filter(Boolean)
        )}

        {renderReviewCard(
          "Retrieval",
          <>
            {renderReviewRow("Built-in sources", retrievalSources.length ? retrievalSources.join(", ") : "None")}
            {renderReviewRow("Local sources", fullTextSources.length)}
            {renderReviewRow("Load excluded", getNested(specForm, ["retrieval", "load_excluded"], false) ? "On" : "Off")}
          </>,
          [
            renderReviewChip(`${retrievalSources.length + fullTextSources.length} source${retrievalSources.length + fullTextSources.length === 1 ? "" : "s"}`, "info"),
          ]
        )}

        {renderReviewCard(
          "Screening",
          <>
            {renderReviewRow("Abstract objective", reviewValue(getNested(specForm, ["screening", "abstract", "objective"], "")))}
            {renderReviewRow("Abstract criteria", `${reviewCount(["screening", "abstract", "inclusion_criteria"])} inclusion · ${reviewCount(["screening", "abstract", "exclusion_criteria"])} exclusion`)}
            {renderReviewRow("Full-text objective", reviewValue(getNested(specForm, ["screening", "fulltext", "objective"], "")))}
            {renderReviewRow("Full-text criteria", `${reviewCount(["screening", "fulltext", "inclusion_criteria"])} inclusion · ${reviewCount(["screening", "fulltext", "exclusion_criteria"])} exclusion`)}
          </>,
          [
            renderReviewChip(`${reviewCount(["screening", "abstract", "inclusion_criteria"]) + reviewCount(["screening", "fulltext", "inclusion_criteria"])} inclusion`, "ok"),
            renderReviewChip(`${reviewCount(["screening", "abstract", "exclusion_criteria"]) + reviewCount(["screening", "fulltext", "exclusion_criteria"])} exclusion`, "muted"),
          ]
        )}

        {renderReviewCard(
          "Parsing",
          <>
            {renderReviewRow("Parse coordinates", isParsingEnabled() ? "On" : "Off")}
            {renderReviewRow("Coordinate model", reviewValue(getNested(specForm, ["parsing", "coordinate_model"], "")))}
          </>,
          [renderReviewChip(isParsingEnabled() ? "Enabled" : "Disabled", isParsingEnabled() ? "ok" : "muted")]
        )}

        {renderReviewCard(
          "Annotation",
          <>
            {renderReviewRow("Annotation", isAnnotationEnabled() ? "On" : "Off")}
            {renderReviewRow("Common criteria", `${reviewCount(["annotation", "inclusion_criteria"])} inclusion · ${reviewCount(["annotation", "exclusion_criteria"])} exclusion`)}
            {renderReviewRow("Named annotation rules", annotations.length)}
            {renderReviewRow("Metadata fields", metadataFieldCount)}
          </>,
          [
            renderReviewChip(isAnnotationEnabled() ? "Enabled" : "Disabled", isAnnotationEnabled() ? "ok" : "muted"),
            renderReviewChip(`${annotations.length} rule${annotations.length === 1 ? "" : "s"}`, "info"),
          ]
        )}
      </div>
    );
  }

  function updateParseCoordinates(value) {
    setSpecForm((prev) => {
      let next = setNested(prev, ["parsing", "parse_coordinates"], value);
      if (!value) {
        next = setNested(next, ["annotation", "enabled"], false);
      }
      return next;
    });
  }

  function isParsingEnabled() {
    return Boolean(getNested(specForm, ["parsing", "parse_coordinates"], true));
  }

  function isAnnotationEnabled() {
    return isParsingEnabled() && Boolean(getNested(specForm, ["annotation", "enabled"], true));
  }

  function enterEditor(projectId, tabName = "build") {
    if (!projectId) return;
    setSelectedProjectId(projectId);
    setEditorTab(tabName);
    if (tabName === "build") {
      setBuildStep("search");
    } else if (tabName === "runs") {
      setRunsSubTab("screening");
    }
    setView("editor");
    setSettingsOpen(false);
  }

  function backToProjects() {
    setView("projects");
    setEditorTab("build");
    setBuildStep("search");
    setRunsSubTab("screening");
    setSettingsOpen(false);
  }

  function updateAnnotationEntry(index, key, value) {
    setSpecForm((prev) => {
      const existing = getNested(prev, ["annotation", "annotations"], []);
      const entries = Array.isArray(existing) ? [...existing] : [];
      while (entries.length <= index) {
        entries.push({
          name: "",
          description: "",
          inclusion_criteria: [],
          exclusion_criteria: [],
        });
      }
      const nextEntry = { ...(entries[index] || {}) };
      nextEntry[key] = value;
      entries[index] = nextEntry;
      return setNested(prev, ["annotation", "annotations"], entries);
    });
  }

  function addAnnotationEntry() {
    setSpecForm((prev) => {
      const existing = getNested(prev, ["annotation", "annotations"], []);
      const entries = Array.isArray(existing) ? [...existing] : [];
      entries.push({
        name: "",
        description: "",
        inclusion_criteria: [],
        exclusion_criteria: [],
      });
      return setNested(prev, ["annotation", "annotations"], entries);
    });
  }

  function removeAnnotationEntry(index) {
    setSpecForm((prev) => {
      const existing = getNested(prev, ["annotation", "annotations"], []);
      const entries = Array.isArray(existing) ? [...existing] : [];
      const nextEntries = entries.filter((_, itemIndex) => itemIndex !== index);
      return setNested(prev, ["annotation", "annotations"], nextEntries);
    });
  }

  function getCriteriaList(path) {
    const criteria = getNested(specForm, path, []);
    return Array.isArray(criteria) ? criteria : [];
  }

  function updateCriteriaItem(path, index, value) {
    setSpecForm((prev) => {
      const existing = getNested(prev, path, []);
      const entries = Array.isArray(existing) ? [...existing] : [];
      entries[index] = value;
      return setNested(prev, path, entries);
    });
  }

  function addCriteriaItem(path) {
    setSpecForm((prev) => {
      const existing = getNested(prev, path, []);
      const entries = Array.isArray(existing) ? [...existing] : [];
      entries.push("");
      return setNested(prev, path, entries);
    });
  }

  function removeCriteriaItem(path, index) {
    setSpecForm((prev) => {
      const existing = getNested(prev, path, []);
      const entries = Array.isArray(existing) ? [...existing] : [];
      return setNested(prev, path, entries.filter((_, itemIndex) => itemIndex !== index));
    });
  }

  function renderCriteriaPanel(path, title, tone) {
    const criteria = getCriteriaList(path);
    const prefix = tone === "include" ? "I" : "E";
    const keyPrefix = path.join("-");
    return (
      <div className={`criteria-panel ${tone}`}>
        <div className="criteria-panel-header">
          <h4>{title}</h4>
          <span className="criteria-count">{criteria.length}</span>
        </div>
        {criteria.length ? (
          <div className="criteria-list">
            {criteria.map((item, index) => (
              <div className="criteria-row" key={`${keyPrefix}-${index}`}>
                <span className="criteria-id">{prefix}{index + 1}</span>
                <input
                  value={item || ""}
                  onChange={(e) => updateCriteriaItem(path, index, e.target.value)}
                />
                <button
                  className="icon-button danger-icon"
                  type="button"
                  title={`Remove ${prefix}${index + 1}`}
                  aria-label={`Remove ${prefix}${index + 1}`}
                  onClick={() => removeCriteriaItem(path, index)}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        ) : (
          <div className="criteria-empty">No criteria yet.</div>
        )}
        <button
          className="secondary criteria-add"
          type="button"
          onClick={() => addCriteriaItem(path)}
        >
          + Criterion
        </button>
      </div>
    );
  }

  function getRetrievalSources() {
    const retrieval = getNested(specForm, ["retrieval"], {});
    if (
      retrieval
      && typeof retrieval === "object"
      && Object.prototype.hasOwnProperty.call(retrieval, "sources")
    ) {
      return Array.isArray(retrieval.sources)
        ? retrieval.sources.map((source) => String(source || "").trim()).filter(Boolean)
        : [];
    }
    return ["pubget"];
  }

  function getFullTextSources() {
    const sources = getNested(specForm, ["retrieval", "full_text_sources"], []);
    return Array.isArray(sources) ? sources : [];
  }

  function updateRetrievalSources(nextSources) {
    const uniqueSources = [];
    for (const source of nextSources || []) {
      const normalized = String(source || "").trim();
      if (normalized && !uniqueSources.includes(normalized)) {
        uniqueSources.push(normalized);
      }
    }
    updateField(["retrieval", "sources"], uniqueSources);
  }

  function openAddSourceModal() {
    const currentSources = getRetrievalSources();
    setSourceModalIndex(null);
    setSourceType(currentSources.includes("pubget") ? "custom" : "pubget");
    setSourceForm({
      root_path: "",
      pmid_source: "folder_name",
      text_path_templates: "",
      coordinates_path_templates: "",
      allowed_extensions: "",
      processed_data_path: "",
      json_filename: "",
      json_pmid_key: "",
    });
    setSourceModalOpen(true);
  }

  function openEditSourceModal(index) {
    const source = getFullTextSources()[index] || {};
    setSourceModalIndex(index);
    setSourceType("custom");
    setSourceForm({
      root_path: source.root_path || "",
      pmid_source: source.pmid_source || "folder_name",
      text_path_templates: stringifyLines(source.text_path_templates || []),
      coordinates_path_templates: stringifyLines(source.coordinates_path_templates || []),
      allowed_extensions: stringifyLines(source.allowed_extensions || []),
      processed_data_path: source.processed_data_path || "",
      json_filename: source.json_filename || "",
      json_pmid_key: source.json_pmid_key || "",
    });
    setSourceModalOpen(true);
  }

  function updateSourceForm(key, value) {
    setSourceForm((prev) => ({ ...prev, [key]: value }));
  }

  function buildCustomSourcePayload() {
    const payload = {
      root_path: String(sourceForm.root_path || "").trim(),
      pmid_source: String(sourceForm.pmid_source || "folder_name").trim() || "folder_name",
    };
    for (const key of ["text_path_templates", "coordinates_path_templates", "allowed_extensions"]) {
      const values = parseLines(sourceForm[key]);
      if (values.length) {
        payload[key] = values;
      }
    }
    for (const key of ["processed_data_path", "json_filename", "json_pmid_key"]) {
      const value = String(sourceForm[key] || "").trim();
      if (value) {
        payload[key] = value;
      }
    }
    return payload;
  }

  function saveSourceModal() {
    if (sourceType === "pubget") {
      const currentSources = getRetrievalSources();
      if (currentSources.includes("pubget")) {
        setStatusMsg({ type: "error", text: "PubGet is already configured." });
        return;
      }
      updateRetrievalSources([...currentSources, "pubget"]);
      setSourceModalOpen(false);
      return;
    }

    const payload = buildCustomSourcePayload();
    if (!payload.root_path) {
      setStatusMsg({ type: "error", text: "Root path is required for a custom source." });
      return;
    }

    const currentSources = getFullTextSources();
    const nextSources = [...currentSources];
    if (sourceModalIndex == null) {
      nextSources.push(payload);
    } else {
      nextSources[sourceModalIndex] = payload;
    }
    updateField(["retrieval", "full_text_sources"], nextSources);
    setSourceModalOpen(false);
  }

  function removeFullTextSource(index) {
    const currentSources = getFullTextSources();
    updateField(
      ["retrieval", "full_text_sources"],
      currentSources.filter((_, itemIndex) => itemIndex !== index)
    );
  }

  function extractPmids(text) {
    const rawLines = String(text || "")
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
    const ids = [];
    const seen = new Set();
    for (const raw of rawLines) {
      const cleaned = raw.replace(/[^0-9]/g, "");
      if (!cleaned) continue;
      if (!seen.has(cleaned)) {
        seen.add(cleaned);
        ids.push(cleaned);
      }
    }
    return ids;
  }

  function handleStudyListFileSelected(file) {
    if (!file) return;
    setStudyListBusy(true);
    setStudyListFileName(file.name || "");
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const content = String(reader.result || "");
        const pmids = extractPmids(content);
        if (!pmids.length) {
          setStatusMsg({ type: "error", text: "No PMIDs found in the selected file." });
          return;
        }
        const pmidQuery = pmids.map((pmid) => `${pmid}[PMID]`).join(" OR ");
        const existing = String(getNested(specForm, ["search", "query"], "") || "").trim();
        const nextQuery = existing
          ? `${existing} OR (${pmidQuery})`
          : pmidQuery;
        updateField(["search", "query"], nextQuery);
        setStatusMsg({ type: "ok", text: `Imported ${pmids.length} PMIDs from ${file.name}.` });
        setStudyListModalOpen(false);
        setStudyListFileName("");
      } catch (err) {
        setStatusMsg({ type: "error", text: `Could not parse file: ${err.message || err}` });
      } finally {
        setStudyListBusy(false);
      }
    };
    reader.onerror = () => {
      setStudyListBusy(false);
      setStatusMsg({ type: "error", text: "Failed to read file." });
    };
    reader.readAsText(file);
  }

  function buildPubMedSearchTerm() {
    const query = String(getNested(specForm, ["search", "query"], "") || "").trim();
    if (!query) return "";
    const dateFrom = String(getNested(specForm, ["search", "date_from"], "") || "").trim();
    const dateTo = String(getNested(specForm, ["search", "date_to"], "") || "").trim();
    if (!dateFrom && !dateTo) return query;

    const startDate = dateFrom || "1800/01/01";
    const endDate = dateTo || "3000/12/31";
    return `(${query}) AND ("${startDate}"[Date - Publication] : "${endDate}"[Date - Publication])`;
  }

  function openSearchInPubMed() {
    const term = buildPubMedSearchTerm();
    if (!term) {
      setStatusMsg({ type: "error", text: "Enter a search query before opening PubMed." });
      return;
    }
    const url = `https://pubmed.ncbi.nlm.nih.gov/?term=${encodeURIComponent(term)}`;
    window.open(url, "_blank", "noopener,noreferrer");
  }

  async function fetchPubMedCount() {
    const term = buildPubMedSearchTerm();
    if (!term) {
      setStatusMsg({ type: "error", text: "Enter a search query before counting PubMed results." });
      return;
    }

    try {
      setPubmedCountBusy(true);
      setPubmedCount(null);
      const params = new URLSearchParams({
        db: "pubmed",
        term,
        retmode: "json",
        rettype: "count",
      });
      const email = String(getNested(specForm, ["search", "email"], "") || "").trim();
      if (email) {
        params.set("email", email);
      }
      const response = await fetch(`https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?${params.toString()}`);
      if (!response.ok) {
        throw new Error(`PubMed count request failed (${response.status})`);
      }
      const payload = await response.json();
      const count = payload?.esearchresult?.count;
      if (count == null) {
        throw new Error("PubMed response did not include a result count.");
      }
      setPubmedCount(String(count));
    } catch (err) {
      setStatusMsg({ type: "error", text: err.message || String(err) });
    } finally {
      setPubmedCountBusy(false);
    }
  }

  function sanitizeWizardForm(form) {
    const cleaned = JSON.parse(JSON.stringify(form || {}));

    const search = (cleaned.search && typeof cleaned.search === "object") ? cleaned.search : {};
    const db = String(search.database || "").trim();
    search.database = db || "pubmed";

    if (searchAdvancedTouched.maxResults) {
      if (search.max_results === "" || search.max_results == null) {
        delete search.max_results;
      } else {
        const parsed = Number(search.max_results);
        if (Number.isFinite(parsed) && parsed > 0) {
          search.max_results = parsed;
        } else {
          delete search.max_results;
        }
      }
    } else if (search.max_results === "" || search.max_results == null) {
      delete search.max_results;
    }

    if (searchAdvancedTouched.email) {
      const email = String(search.email || "").trim();
      if (email) {
        search.email = email;
      } else {
        delete search.email;
      }
    } else if (!String(search.email || "").trim()) {
      delete search.email;
    }
    cleaned.search = search;

    const retrieval = (cleaned.retrieval && typeof cleaned.retrieval === "object")
      ? cleaned.retrieval
      : {};
    if (!Array.isArray(retrieval.sources)) {
      retrieval.sources = ["pubget"];
    } else {
      retrieval.sources = retrieval.sources.map((source) => String(source || "").trim()).filter(Boolean);
    }
    if (!Array.isArray(retrieval.full_text_sources)) {
      retrieval.full_text_sources = [];
    }
    if (typeof retrieval.load_excluded !== "boolean") {
      retrieval.load_excluded = false;
    }
    cleaned.retrieval = retrieval;

    const screening = (cleaned.screening && typeof cleaned.screening === "object")
      ? cleaned.screening
      : {};
    for (const stage of ["abstract", "fulltext"]) {
      const stageData = (screening[stage] && typeof screening[stage] === "object")
        ? screening[stage]
        : {};
      if (typeof stageData.confidence_reporting !== "boolean") {
        stageData.confidence_reporting = true;
      }
      screening[stage] = stageData;
    }
    cleaned.screening = screening;

    const parsing = (cleaned.parsing && typeof cleaned.parsing === "object")
      ? cleaned.parsing
      : {};
    if (!String(parsing.coordinate_model || "").trim()) {
      delete parsing.coordinate_model;
    }
    cleaned.parsing = parsing;

    const annotation = (cleaned.annotation && typeof cleaned.annotation === "object")
      ? cleaned.annotation
      : {};
    if (parsing.parse_coordinates === false) {
      annotation.enabled = false;
    }
    const metadataFields = Array.isArray(annotation.metadata_fields)
      ? annotation.metadata_fields.map((item) => String(item || "").trim()).filter(Boolean)
      : [];
    annotation.metadata_fields = metadataFields.length
      ? metadataFields
      : [...DEFAULT_ANNOTATION_METADATA_FIELDS];
    cleaned.annotation = annotation;

    const output = (cleaned.output && typeof cleaned.output === "object")
      ? cleaned.output
      : {};
    delete output.directory;
    const formats = Array.isArray(output.formats)
      ? output.formats.map((item) => String(item || "").trim()).filter(Boolean)
      : [];
    output.formats = formats.length ? formats : ["csv", "json"];
    if (typeof output.nimads !== "boolean") {
      output.nimads = true;
    }
    if (typeof output.export_excluded_studies !== "boolean") {
      output.export_excluded_studies = true;
    }
    cleaned.output = output;

    return cleaned;
  }

  async function createProject() {
    const name = createName.trim();
    if (!name) {
      setStatusMsg({ type: "error", text: "Project name is required." });
      return;
    }
    try {
      setCreateBusy(true);
      const project = await api("/api/projects", {
        method: "POST",
        body: JSON.stringify({
          name,
          description: createDescription.trim() || null,
        }),
      });
      if (globalPreferredModel) {
        await api(`/api/projects/${project.id}/spec`, {
          method: "PUT",
          body: JSON.stringify({
            form: {
              defaults: {
                model: globalPreferredModel,
              },
            },
          }),
        });
      }
      await refreshProjects();
      enterEditor(project.id, "build");
      setCreateModalOpen(false);
      setCreateName("");
      setCreateDescription("");
      setStatusMsg({ type: "ok", text: "Project created." });
    } catch (err) {
      setStatusMsg({ type: "error", text: err.message });
    } finally {
      setCreateBusy(false);
    }
  }

  async function importProject() {
    const configPath = importConfigPath.trim();
    if (!configPath) return;
    try {
      setImportBusy(true);
      const project = await api("/api/projects", {
        method: "POST",
        body: JSON.stringify({
          config_path: configPath,
          name: importName.trim() || null,
          description: importDescription.trim() || null,
        }),
      });
      await refreshProjects();
      enterEditor(project.id, "build");
      setImportModalOpen(false);
      setImportConfigPath("");
      setImportName("");
      setImportDescription("");
      setStatusMsg({ type: "ok", text: "Project imported." });
    } catch (err) {
      setStatusMsg({ type: "error", text: err.message });
    } finally {
      setImportBusy(false);
    }
  }

  function openEditProjectModal(project) {
    if (!project?.id) return;
    setEditProjectId(project.id);
    setEditName(project.name || "");
    setEditDescription(project.description || "");
    setEditModalOpen(true);
  }

  function openCloneProjectModal(project) {
    if (!project?.id) return;
    setCloneProjectId(project.id);
    setCloneName(`${project.name || "Project"} copy`);
    setCloneDescription(project.description || "");
    setCloneMode("schema_only");
    setCloneModalOpen(true);
  }

  async function saveProjectDetails() {
    const name = editName.trim();
    if (!name) {
      setStatusMsg({ type: "error", text: "Project name is required." });
      return;
    }
    if (!editProjectId) return;
    try {
      setEditBusy(true);
      await api(`/api/projects/${editProjectId}`, {
        method: "PUT",
        body: JSON.stringify({
          name,
          description: editDescription.trim(),
        }),
      });
      await refreshProjects();
      setEditModalOpen(false);
      setStatusMsg({ type: "ok", text: "Project details updated." });
    } catch (err) {
      setStatusMsg({ type: "error", text: err.message });
    } finally {
      setEditBusy(false);
    }
  }

  async function cloneProject() {
    const name = cloneName.trim();
    if (!name) {
      setStatusMsg({ type: "error", text: "Clone name is required." });
      return;
    }
    if (!cloneProjectId) return;
    try {
      setCloneBusy(true);
      const clonePayload = {
        mode: cloneMode,
        name,
        description: cloneDescription.trim(),
      };
      let cloned;
      try {
        cloned = await api(`/api/projects/${cloneProjectId}/clone`, {
          method: "POST",
          body: JSON.stringify(clonePayload),
        });
      } catch (err) {
        if (!String(err?.message || "").toLowerCase().includes("method not allowed")) {
          throw err;
        }
        cloned = await api(`/api/projects/${cloneProjectId}/clone`, {
          method: "PUT",
          body: JSON.stringify(clonePayload),
        });
      }
      await refreshProjects();
      setCloneModalOpen(false);
      setCloneProjectId("");
      setCloneName("");
      setCloneDescription("");
      setCloneMode("schema_only");
      enterEditor(cloned.id, "build");
      const report = cloned?.clone_report || {};
      if (cloneMode === "schema_and_cached_results") {
        setStatusMsg({
          type: "ok",
          text: `Project cloned with schema + cached results (${report.cloned_runs_count || 0} runs copied).`,
        });
      } else {
        setStatusMsg({ type: "ok", text: "Project cloned (schema only)." });
      }
    } catch (err) {
      setStatusMsg({ type: "error", text: err.message });
    } finally {
      setCloneBusy(false);
    }
  }

  function closeDeleteDialog() {
    setDeletePreview(null);
    setDeleteMode("metadata_only");
  }

  async function openDeleteDialog(projectId) {
    if (!projectId) return;
    try {
      const preview = await api(`/api/projects/${projectId}/delete-preview`);
      setDeletePreview(preview);
      setDeleteMode(preview.has_outputs ? "metadata_config_and_outputs" : "metadata_only");
    } catch (err) {
      setStatusMsg({ type: "error", text: err.message });
    }
  }

  function buildDeleteResultMessage(report) {
    const runCount = report?.removed?.run_metadata_files?.length || 0;
    const outputsCount = report?.removed?.output_folders?.length || 0;
    const configDeleted = Boolean(report?.removed?.config_path);
    const skippedConfig = report?.skipped?.config;
    const skippedOutputsCount = report?.skipped?.output_folders?.length || 0;

    const parts = [
      "Removed project metadata",
      `removed ${runCount} run metadata file${runCount === 1 ? "" : "s"}`,
      configDeleted ? "deleted config" : "kept config",
      `deleted ${outputsCount} output folder${outputsCount === 1 ? "" : "s"}`,
    ];
    if (skippedConfig) {
      parts.push(`config skipped (${skippedConfig})`);
    }
    if (skippedOutputsCount) {
      parts.push(`skipped ${skippedOutputsCount} output folder${skippedOutputsCount === 1 ? "" : "s"}`);
    }
    return parts.join(", ") + ".";
  }

  async function executeDeleteProject() {
    if (!deletePreview?.project_id) return;
    try {
      setDeleteBusy(true);
      const report = await api(`/api/projects/${deletePreview.project_id}/delete`, {
        method: "POST",
        body: JSON.stringify({ mode: deleteMode }),
      });
      closeDeleteDialog();
      if (selectedProjectId === deletePreview.project_id) {
        setSelectedProjectId(null);
        setSpecForm({});
        setYamlText("");
        setSpecPath("");
        setView("projects");
        setEditorTab("build");
        setSettingsOpen(false);
      }
      setSelectedRunId(null);
      setSelectedRun(null);
      setLogs([]);
      logOffsetRef.current = 0;
      setLogOffset(0);
      await refreshProjects();
      await refreshRuns();
      setStatusMsg({ type: "ok", text: buildDeleteResultMessage(report) });
    } catch (err) {
      setStatusMsg({ type: "error", text: err.message });
    } finally {
      setDeleteBusy(false);
    }
  }

  async function saveSpec() {
    if (!selectedProjectId) return;
    try {
      const payload = yamlMode
        ? { yaml_text: yamlText }
        : { form: sanitizeWizardForm(specForm) };
      const result = await api(`/api/projects/${selectedProjectId}/spec`, {
        method: "PUT",
        body: JSON.stringify(payload),
      });
      setYamlText(result.yaml_text || "");
      setSpecForm(result.form || {});
      setStatusMsg({ type: "ok", text: "Specification saved." });
    } catch (err) {
      setStatusMsg({ type: "error", text: err.message });
    }
  }

  async function autosaveSpecForNavigation() {
    if (!selectedProjectId) return false;
    try {
      const payload = yamlMode
        ? { yaml_text: yamlText }
        : { form: sanitizeWizardForm(specForm) };
      const result = await api(`/api/projects/${selectedProjectId}/spec`, {
        method: "PUT",
        body: JSON.stringify(payload),
      });
      setYamlText(result.yaml_text || "");
      setSpecForm(result.form || {});
      return true;
    } catch (err) {
      setStatusMsg({ type: "error", text: `Autosave failed: ${err.message}` });
      return false;
    }
  }

  async function navigateBuildStep(stepId) {
    if (!stepId || stepId === buildStep) return;
    const saved = await autosaveSpecForNavigation();
    if (!saved) return;
    setEditorTab("build");
    setBuildStep(stepId);
  }

  async function navigateEditorTab(tabName) {
    if (tabName === editorTab) return;
    const saved = await autosaveSpecForNavigation();
    if (!saved) return;
    setEditorTab(tabName);
    if (tabName === "runs") {
      setRunsSubTab("screening");
    }
  }

  async function goBuildStepWithAutosave(direction) {
    const nextIndex = Math.min(
      Math.max(buildStepIndex + direction, 0),
      BUILD_STEPS.length - 1
    );
    await navigateBuildStep(BUILD_STEPS[nextIndex][0]);
  }

  async function validateSpec() {
    if (!selectedProjectId) return;
    try {
      if (yamlMode) {
        await saveSpec();
      }
      const result = await api(`/api/projects/${selectedProjectId}/validate`, {
        method: "POST",
      });
      if (result.ok) {
        setStatusMsg({ type: "ok", text: result.message || "Config valid." });
      } else {
        setStatusMsg({ type: "error", text: result.message || "Config invalid." });
      }
    } catch (err) {
      setStatusMsg({ type: "error", text: err.message });
    }
  }

  async function startRun() {
    if (!selectedProjectId) return;
    try {
      const payload = {
        ...runForm,
        output_folder: runForm.output_folder || null,
      };
      const run = await api(`/api/projects/${selectedProjectId}/runs`, {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setSelectedRunId(run.id);
      setSelectedRun(run);
      setLogs([]);
      logOffsetRef.current = 0;
      setLogOffset(0);
      await refreshRuns();
      setStatusMsg({ type: "ok", text: "Run started." });
    } catch (err) {
      setStatusMsg({ type: "error", text: err.message });
    }
  }

  async function startMetaRun() {
    if (!selectedProjectId) return;
    try {
      const payload = {
        ...metaForm,
        include_ids: metaForm.include_ids || null,
      };
      const run = await api(`/api/projects/${selectedProjectId}/meta-runs`, {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setSelectedRunId(run.id);
      setSelectedRun(run);
      setLogs([]);
      logOffsetRef.current = 0;
      setLogOffset(0);
      await refreshRuns();
      setStatusMsg({ type: "ok", text: "Meta run started." });
    } catch (err) {
      setStatusMsg({ type: "error", text: err.message });
    }
  }

  async function cancelRun() {
    const runIdToCancel = activeRunForActiveTab?.id || currentExecutionRun?.id || selectedRunId;
    if (!runIdToCancel) return;
    try {
      await api(`/api/runs/${runIdToCancel}/cancel`, { method: "POST" });
      setStatusMsg({ type: "ok", text: "Cancel requested." });
      await refreshRuns();
    } catch (err) {
      setStatusMsg({ type: "error", text: err.message });
    }
  }

  async function saveSecrets() {
    try {
      await api("/api/settings/secrets", {
        method: "PUT",
        body: JSON.stringify(secrets),
      });
      await refreshSecrets();
      setStatusMsg({ type: "ok", text: "Secrets saved to ~/.autonima.env." });
    } catch (err) {
      setStatusMsg({ type: "error", text: err.message });
    }
  }

  async function savePreferences() {
    try {
      const nextModels = parseLines(preferredModelsText);
      const nextDefaultModel = pickPreferredDefaultModel(nextModels, globalPreferredModel);
      const payload = {
        preferred_models: nextModels,
        default_model: nextDefaultModel,
      };
      const data = await api("/api/settings/preferences", {
        method: "PUT",
        body: JSON.stringify(payload),
      });
      const models = Array.isArray(data?.preferred_models) ? data.preferred_models : [];
      const preferredDefault = typeof data?.default_model === "string" ? data.default_model : "";
      const resolvedDefault = pickPreferredDefaultModel(models, preferredDefault);
      setPreferredModels(models);
      setPreferredModelsText(stringifyLines(models));
      setGlobalPreferredModel(resolvedDefault);
      setYamlModelSelection((prev) => (models.includes(prev) ? prev : (resolvedDefault || "")));
      setStatusMsg({ type: "ok", text: "Preferred models saved." });
    } catch (err) {
      setStatusMsg({ type: "error", text: err.message });
    }
  }

  return (
    <div className="app-shell">
      <datalist id="preferred-model-options">
        {preferredModels.map((modelName) => (
          <option key={modelName} value={modelName} />
        ))}
      </datalist>
      <div className="topbar">
        <div className="brand-lockup">
          <img className="brand-logo" src="/static/synth.png" alt="Autonima logo" />
          <div className="brand-text">
            <div className="brand">autonima</div>
            <div className="brand-subtitle">LLM-assisted neuroimaging review workflows</div>
          </div>
        </div>
        <div className="topbar-actions">
          <label className="topbar-model-picker">
            <span>Default model</span>
            <select
              value={globalPreferredModel}
              onChange={(e) => saveGlobalModelPreference(e.target.value)}
              disabled={!preferredModels.length}
              title={preferredModels.length ? "Global default model for new projects and model fields" : "Add preferred models in global settings first"}
            >
              {!preferredModels.length ? (
                <option value="">No preferred models</option>
              ) : (
                <option value="">None</option>
              )}
              {preferredModels.map((modelName) => (
                <option key={modelName} value={modelName}>
                  {modelName}
                </option>
              ))}
            </select>
          </label>
          <button
            className="settings-button"
            title="Global Settings"
            aria-label="Open global settings"
            onClick={() => setSettingsOpen(true)}
          >
            ⚙
          </button>
        </div>
      </div>

      {statusMsg ? (
        <div className={`status-msg ${statusMsg.type}`}>{statusMsg.text}</div>
      ) : null}

      {deletePreview ? (
        <div className="modal-backdrop" onClick={closeDeleteDialog}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <h3>Delete Project</h3>
            <p>
              <strong>{deletePreview.project_name}</strong>
            </p>
            <div className="kv mono" style={{ marginBottom: 12 }}>
              <div>Config Path</div>
              <div>{deletePreview.config_path}</div>
              <div>Run Metadata</div>
              <div>{deletePreview.run_metadata_count}</div>
              <div>Detected Outputs</div>
              <div>{deletePreview.output_folders_detected?.length || 0}</div>
            </div>

            {deletePreview.has_active_runs ? (
              <div className="status-msg error">
                Project has active runs ({(deletePreview.active_run_ids || []).join(", ")}). Stop runs before deleting.
              </div>
            ) : null}

            {!deletePreview.config_deletable ? (
              <div className="status-msg error">
                Config file is outside workspace boundary and will not be deleted.
              </div>
            ) : null}

            {(deletePreview.output_folders_detected || []).length > 0 ? (
              <div style={{ marginBottom: 10 }}>
                <label>Output Folders</label>
                <div className="mono list-box">
                  {(deletePreview.output_folders_detected || []).map((path) => {
                    const inWorkspace = (deletePreview.output_folders_deletable || []).includes(path);
                    return (
                      <div key={path}>
                        {path}
                        {!inWorkspace ? " (outside workspace, cannot delete)" : ""}
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : null}

            <div style={{ marginBottom: 12 }}>
              <label>Delete Mode</label>
              <select value={deleteMode} onChange={(e) => setDeleteMode(e.target.value)}>
                <option value="metadata_only">Metadata only</option>
                <option value="metadata_and_config">Metadata + config</option>
                {deletePreview.has_outputs ? (
                  <option value="metadata_config_and_outputs">
                    Metadata + config + workspace outputs
                  </option>
                ) : null}
              </select>
            </div>

            <div className="actions">
              <button className="secondary" onClick={closeDeleteDialog} disabled={deleteBusy}>
                Cancel
              </button>
              <button
                className="danger"
                onClick={executeDeleteProject}
                disabled={deleteBusy || deletePreview.has_active_runs}
              >
                {deleteBusy ? "Deleting..." : "Delete Project"}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {createModalOpen ? (
        <div className="modal-backdrop" onClick={() => setCreateModalOpen(false)}>
          <div className="modal-card modal-card-narrow" onClick={(e) => e.stopPropagation()}>
            <h3>Create New Project</h3>
            <div>
              <label>Project Name</label>
              <input
                value={createName}
                onChange={(e) => setCreateName(e.target.value)}
              />
            </div>
            <div style={{ marginTop: 10 }}>
              <label>Description (optional)</label>
              <textarea
                value={createDescription}
                onChange={(e) => setCreateDescription(e.target.value)}
                placeholder="Short summary of this project"
                style={{ minHeight: 90 }}
              />
            </div>
            <div className="actions" style={{ marginTop: 12 }}>
              <button className="secondary" onClick={() => setCreateModalOpen(false)} disabled={createBusy}>
                Cancel
              </button>
              <button className="primary" onClick={createProject} disabled={createBusy || !createName.trim()}>
                {createBusy ? "Creating..." : "Create Project"}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {importModalOpen ? (
        <div className="modal-backdrop" onClick={() => setImportModalOpen(false)}>
          <div className="modal-card modal-card-narrow" onClick={(e) => e.stopPropagation()}>
            <h3>Import YAML Project</h3>
            <div>
              <label>YAML Config Path</label>
              <input
                className="mono"
                value={importConfigPath}
                onChange={(e) => setImportConfigPath(e.target.value)}
                placeholder="/path/to/config.yaml"
              />
            </div>
            <div style={{ marginTop: 10 }}>
              <label>Project Name (optional)</label>
              <input
                value={importName}
                onChange={(e) => setImportName(e.target.value)}
                placeholder="Defaults to YAML file name"
              />
            </div>
            <div style={{ marginTop: 10 }}>
              <label>Description (optional)</label>
              <textarea
                value={importDescription}
                onChange={(e) => setImportDescription(e.target.value)}
                placeholder="Short summary of this project"
                style={{ minHeight: 90 }}
              />
            </div>
            <div className="actions" style={{ marginTop: 12 }}>
              <button className="secondary" onClick={() => setImportModalOpen(false)} disabled={importBusy}>
                Cancel
              </button>
              <button
                className="primary"
                onClick={importProject}
                disabled={importBusy || !importConfigPath.trim()}
              >
                {importBusy ? "Importing..." : "Import Project"}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {editModalOpen ? (
        <div className="modal-backdrop" onClick={() => setEditModalOpen(false)}>
          <div className="modal-card modal-card-narrow" onClick={(e) => e.stopPropagation()}>
            <h3>Edit Project</h3>
            <div>
              <label>Project Name</label>
              <input
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
              />
            </div>
            <div style={{ marginTop: 10 }}>
              <label>Description (optional)</label>
              <textarea
                value={editDescription}
                onChange={(e) => setEditDescription(e.target.value)}
                placeholder="Short summary of this project"
                style={{ minHeight: 90 }}
              />
            </div>
            <div className="actions" style={{ marginTop: 12 }}>
              <button className="secondary" onClick={() => setEditModalOpen(false)} disabled={editBusy}>
                Cancel
              </button>
              <button className="primary" onClick={saveProjectDetails} disabled={editBusy || !editName.trim()}>
                {editBusy ? "Saving..." : "Save Changes"}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {cloneModalOpen ? (
        <div className="modal-backdrop" onClick={() => setCloneModalOpen(false)}>
          <div className="modal-card modal-card-narrow" onClick={(e) => e.stopPropagation()}>
            <h3>Clone Project</h3>
            <div>
              <label>New Project Name</label>
              <input
                value={cloneName}
                onChange={(e) => setCloneName(e.target.value)}
              />
            </div>
            <div style={{ marginTop: 10 }}>
              <label>Description (optional)</label>
              <textarea
                value={cloneDescription}
                onChange={(e) => setCloneDescription(e.target.value)}
                placeholder="Short summary of this project"
                style={{ minHeight: 90 }}
              />
            </div>
            <div style={{ marginTop: 10 }}>
              <label>Clone Mode</label>
              <select value={cloneMode} onChange={(e) => setCloneMode(e.target.value)}>
                <option value="schema_only">Schema only</option>
                <option value="schema_and_cached_results">Schema + cached results</option>
              </select>
            </div>
            <div className="actions" style={{ marginTop: 12 }}>
              <button className="secondary" onClick={() => setCloneModalOpen(false)} disabled={cloneBusy}>
                Cancel
              </button>
              <button className="primary" onClick={cloneProject} disabled={cloneBusy || !cloneName.trim()}>
                {cloneBusy ? "Cloning..." : "Clone Project"}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {projectModelSettingsOpen ? (
        <div className="modal-backdrop" onClick={() => setProjectModelSettingsOpen(false)}>
          <div className="modal-card modal-card-narrow" onClick={(e) => e.stopPropagation()}>
            <h3>Model Settings</h3>
            <div>
              <label htmlFor="project-default-model">Global Default Model (optional)</label>
              <select
                id="project-default-model"
                value={globalPreferredModel}
                onChange={(e) => saveGlobalModelPreference(e.target.value)}
                disabled={!preferredModels.length}
                title={preferredModels.length ? "Global default model for new projects and model fields" : "Add preferred models in global settings first"}
              >
                {!preferredModels.length ? (
                  <option value="">No preferred models</option>
                ) : (
                  <option value="">None</option>
                )}
                {preferredModels.map((modelName) => (
                  <option key={modelName} value={modelName}>
                    {modelName}
                  </option>
                ))}
              </select>
            </div>
            <div className="actions" style={{ marginTop: 12 }}>
              <button className="secondary" onClick={() => setProjectModelSettingsOpen(false)}>
                Close
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {studyListModalOpen ? (
        <div className="modal-backdrop" onClick={() => setStudyListModalOpen(false)}>
          <div className="modal-card modal-card-narrow" onClick={(e) => e.stopPropagation()}>
            <h3>Add Study List</h3>
            <p>
              Select a <code>.txt</code> file with one PMID per line.
            </p>
            <div>
              <label>PMID Text File</label>
              <input
                type="file"
                accept=".txt,text/plain"
                onChange={(e) => handleStudyListFileSelected(e.target.files?.[0])}
                disabled={studyListBusy}
              />
            </div>
            {studyListFileName ? (
              <p className="mono" style={{ marginTop: 8 }}>
                Selected: {studyListFileName}
              </p>
            ) : null}
            <div className="actions" style={{ marginTop: 12 }}>
              <button className="secondary" onClick={() => setStudyListModalOpen(false)} disabled={studyListBusy}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {sourceModalOpen ? (
        <div className="modal-backdrop" onClick={() => setSourceModalOpen(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <h3>{sourceModalIndex == null ? "Add Source" : "Edit Source"}</h3>
            <div className="grid-2">
              <div>
                <label>Source Type</label>
                <select
                  value={sourceType}
                  onChange={(e) => setSourceType(e.target.value)}
                  disabled={sourceModalIndex != null}
                >
                  <option value="pubget">PubGet</option>
                  <option value="custom">Local full-text folder</option>
                </select>
              </div>
              {sourceType === "custom" ? (
                <div>
                  <label>PMID Source</label>
                  <select
                    value={sourceForm.pmid_source}
                    onChange={(e) => updateSourceForm("pmid_source", e.target.value)}
                  >
                    <option value="folder_name">folder_name</option>
                    <option value="file_name">file_name</option>
                    <option value="json">json</option>
                  </select>
                </div>
              ) : null}
            </div>

            {sourceType === "custom" ? (
              <>
                <div style={{ marginTop: 10 }}>
                  <label>Root Path</label>
                  <input
                    className="mono"
                    value={sourceForm.root_path}
                    onChange={(e) => updateSourceForm("root_path", e.target.value)}
                    placeholder="/path/to/fulltexts"
                  />
                </div>
                <div className="grid-2" style={{ marginTop: 10 }}>
                  <div>
                    <label>Text Path Templates (newline list)</label>
                    <textarea
                      className="mono"
                      value={sourceForm.text_path_templates}
                      onChange={(e) => updateSourceForm("text_path_templates", e.target.value)}
                      placeholder={"text.txt\nprocessed/pubget/text.txt"}
                    />
                  </div>
                  <div>
                    <label>Coordinate Path Templates (newline list)</label>
                    <textarea
                      className="mono"
                      value={sourceForm.coordinates_path_templates}
                      onChange={(e) => updateSourceForm("coordinates_path_templates", e.target.value)}
                      placeholder="coordinates.json"
                    />
                  </div>
                </div>
                <div className="grid-2" style={{ marginTop: 10 }}>
                  <div>
                    <label>Allowed Extensions (newline list)</label>
                    <textarea
                      className="mono"
                      value={sourceForm.allowed_extensions}
                      onChange={(e) => updateSourceForm("allowed_extensions", e.target.value)}
                      placeholder={".html\n.txt"}
                    />
                  </div>
                  <div>
                    <label>Processed Data Path</label>
                    <input
                      className="mono"
                      value={sourceForm.processed_data_path}
                      onChange={(e) => updateSourceForm("processed_data_path", e.target.value)}
                      placeholder="/path/to/processed"
                    />
                  </div>
                </div>
                <div className="grid-2" style={{ marginTop: 10 }}>
                  <div>
                    <label>JSON Filename</label>
                    <input
                      value={sourceForm.json_filename}
                      onChange={(e) => updateSourceForm("json_filename", e.target.value)}
                      placeholder="identifiers.json"
                    />
                  </div>
                  <div>
                    <label>JSON PMID Key</label>
                    <input
                      value={sourceForm.json_pmid_key}
                      onChange={(e) => updateSourceForm("json_pmid_key", e.target.value)}
                      placeholder="pmid"
                    />
                  </div>
                </div>
              </>
            ) : (
              <div className="status-msg" style={{ marginTop: 10 }}>
                PubGet uses the built-in Autonima retrieval integration and does not need additional fields.
              </div>
            )}

            <div className="actions" style={{ marginTop: 12 }}>
              <button className="secondary" onClick={() => setSourceModalOpen(false)}>
                Cancel
              </button>
              <button className="primary" onClick={saveSourceModal}>
                {sourceModalIndex == null ? "Add Source" : "Save Source"}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {settingsOpen ? (
        <div>
          <div className="card">
            <div className="actions" style={{ justifyContent: "space-between", marginBottom: 8 }}>
              <h2 style={{ marginBottom: 0 }}>Configuration & Secrets</h2>
              <button className="secondary" onClick={() => setSettingsOpen(false)}>
                Close Settings
              </button>
            </div>
            <p>Stored in <code>~/.autonima.env</code>, not in project YAML.</p>

            <h3>Preferred Models</h3>
            <p>
              Stored in <code>~/.autonima-ui.json</code>. The project model settings selector uses this list,
              and model fields default to that selected value unless overridden.
            </p>
            <div>
              <label>Preferred Model Names (one per line)</label>
              <textarea
                className="mono"
                value={preferredModelsText}
                onChange={(e) => setPreferredModelsText(e.target.value)}
                placeholder={"gpt-5-mini-2025-08-07\ngpt-4.1-mini\ngpt-4o-mini"}
              />
            </div>

            <div className="actions" style={{ marginTop: 12, marginBottom: 12 }}>
              <button className="secondary" onClick={savePreferences}>Save Preferred Models</button>
              <button className="secondary" onClick={refreshPreferences}>Reload Preferred Models</button>
            </div>

            <h3>API Keys & Defaults</h3>
            <div className="grid-2">
              {[
                "OPENAI_API_KEY",
                "OPENAI_API_GATEWAY",
                "PUBGET_API_KEY",
                "NCBI_EMAIL",
              ].map((key) => (
                <div key={key}>
                  <label>{key}</label>
                  <input
                    type={key.includes("KEY") ? "password" : "text"}
                    value={secrets[key] || ""}
                    placeholder={maskedSecrets[key] || ""}
                    onChange={(e) => setSecrets((prev) => ({ ...prev, [key]: e.target.value }))}
                  />
                </div>
              ))}
            </div>

            <div className="actions" style={{ marginTop: 12 }}>
              <button className="primary" onClick={saveSecrets}>Save Secrets</button>
              <button className="secondary" onClick={refreshSecrets}>Reload</button>
            </div>
          </div>
        </div>
      ) : view === "projects" ? (
        <div className="card projects-landing">
          <div className="projects-hero">
            <div className="projects-hero-copy">
              <div className="hero-kicker">AUTONIMA WORKSPACE</div>
              <p>
                Start by selecting a project or creating a new one, then launch full
                Autonima pipelines with live progress and safe run history.
              </p>
            </div>
          </div>
          <h2>Projects</h2>
          <div className="actions" style={{ marginBottom: 12 }}>
            <button
              className="primary cta-orange"
              onClick={() => {
                setCreateName("");
                setCreateDescription("");
                setCreateModalOpen(true);
              }}
            >
              New Project
            </button>
            <button
              className="secondary"
              onClick={() => {
                setImportConfigPath("");
                setImportName("");
                setImportDescription("");
                setImportModalOpen(true);
              }}
            >
              Import YAML
            </button>
          </div>
          <div className="project-list">
            {projects.map((project) => (
              <div
                key={project.id}
                className={`project-item project-row ${selectedProjectId === project.id ? "active" : ""}`}
              >
                <div
                  className="project-main"
                  role="button"
                  tabIndex={0}
                  onClick={() => enterEditor(project.id, "build")}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      enterEditor(project.id, "build");
                    }
                  }}
                >
                  <div className="project-title">{project.name}</div>
                  {project.description ? (
                    <div className="project-description">{project.description}</div>
                  ) : (
                    <div className="project-description project-description-empty">
                      No description
                    </div>
                  )}
                  <div style={{ marginTop: 6 }}>
                    <span className="badge queued">{project.source}</span>
                  </div>
                </div>
                <div className="project-row-actions">
                  <button
                    className="icon-button clone-icon"
                    title={`Clone ${project.name}`}
                    aria-label={`Clone ${project.name}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      openCloneProjectModal(project);
                    }}
                  >
                    ⧉
                  </button>
                  <button
                    className="icon-button edit-icon"
                    title={`Edit ${project.name}`}
                    aria-label={`Edit ${project.name}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      openEditProjectModal(project);
                    }}
                  >
                    ✎
                  </button>
                  <button
                    className="icon-button danger-icon"
                    title={`Delete ${project.name}`}
                    aria-label={`Delete ${project.name}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      openDeleteDialog(project.id);
                    }}
                  >
                    🗑
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="layout">
          <div className="nav">
            <button className="secondary nav-back-button" onClick={backToProjects}>
              ← Back to Projects
            </button>
            <button
              className={editorTab === "build" ? "active" : ""}
              onClick={() => navigateEditorTab("build")}
            >
              Create
            </button>
            {editorTab === "build" ? (
              <div className="nav-subitems">
                {BUILD_STEPS.map(([stepId, label]) => (
                  <button
                    key={stepId}
                    className={buildStep === stepId ? "active" : ""}
                    onClick={() => navigateBuildStep(stepId)}
                  >
                    {label}
                  </button>
                ))}
              </div>
            ) : null}
            <button
              className={editorTab === "runs" ? "active" : ""}
              onClick={() => navigateEditorTab("runs")}
            >
              Run
            </button>
            {editorTab === "runs" ? (
              <div className="nav-subitems">
                <button
                  className={runsSubTab === "screening" ? "active" : ""}
                  onClick={() => setRunsSubTab("screening")}
                >
                  Screening
                </button>
                <button
                  className={runsSubTab === "meta" ? "active" : ""}
                  disabled={!metaAnalysisEnabled}
                  title={
                    metaAnalysisEnabled
                      ? "Run a meta-analysis from available NIMADS output"
                      : "Complete a screening run with NIMADS output to enable Meta-Analysis"
                  }
                  onClick={() => {
                    if (metaAnalysisEnabled) {
                      setRunsSubTab("meta");
                    }
                  }}
                >
                  Meta-Analysis
                </button>
              </div>
            ) : null}
          </div>

          <div>
            <div className="card">
              <div className="actions editor-header-row" style={{ marginBottom: selectedProject ? 12 : 0 }} />
              {selectedProject ? (
                <div className="editor-project-meta">
                  <div className="editor-project-title-row">
                    <h2 className="editor-project-title">{selectedProject.name}</h2>
                    <button
                      className="icon-button"
                      title="Model settings"
                      aria-label="Open model settings"
                      onClick={() => setProjectModelSettingsOpen(true)}
                    >
                      ⚙
                    </button>
                    <button
                      className="icon-button edit-icon"
                      title={`Edit ${selectedProject.name}`}
                      aria-label={`Edit ${selectedProject.name}`}
                      onClick={() => openEditProjectModal(selectedProject)}
                    >
                      ✎
                    </button>
                  </div>
                  {selectedProject.description ? (
                    <p className="editor-project-description">{selectedProject.description}</p>
                  ) : (
                    <p className="editor-project-description editor-project-description-empty">
                      No description
                    </p>
                  )}
                </div>
              ) : null}
            </div>

            {!selectedProject ? (
              <div className="card">
                <h3>No Project Selected</h3>
                <p>Select a project from the Projects view to start editing and running.</p>
                <button className="secondary" onClick={backToProjects}>Go to Projects</button>
              </div>
            ) : null}

            {selectedProject && editorTab === "build" && (
            <>
              <div className="card">
                {buildStep === "search" ? (
                  <>
                    <h3>Find studies</h3>
                    <p>Identify studies to screen by searching repositories or importing study list.</p>
                    <div className="builder-section">
                      <h3>Search</h3>
                      <p className="section-explainer">Search literature repositories by query and publication date range.</p>

                      <div style={{ marginTop: 10 }}>
                        <label>Search Query</label>
                        <input
                          value={getNested(specForm, ["search", "query"], "")}
                          onChange={(e) => {
                            setPubmedCount(null);
                            updateField(["search", "query"], e.target.value);
                          }}
                        />
                      </div>
                      <div className="grid-2">
                        <div>
                          <label>Date From</label>
                          <input
                            type="date"
                            value={toDateInputValue(getNested(specForm, ["search", "date_from"], ""))}
                            onChange={(e) => {
                              setPubmedCount(null);
                              updateField(["search", "date_from"], fromDateInputValue(e.target.value));
                            }}
                          />
                        </div>
                        <div>
                          <label>Date To</label>
                          <input
                            type="date"
                            value={toDateInputValue(getNested(specForm, ["search", "date_to"], ""))}
                            onChange={(e) => {
                              setPubmedCount(null);
                              updateField(["search", "date_to"], fromDateInputValue(e.target.value));
                            }}
                          />
                        </div>
                      </div>
                      <div className="actions" style={{ marginTop: 10 }}>
                        <button className="secondary" type="button" onClick={openSearchInPubMed}>
                          Open in PubMed
                        </button>
                        <button className="secondary" type="button" onClick={fetchPubMedCount} disabled={pubmedCountBusy}>
                          {pubmedCountBusy ? "Counting..." : "Count Results"}
                        </button>
                        {pubmedCount != null ? (
                          <span className="pubmed-count-result">{Number(pubmedCount).toLocaleString()} studies</span>
                        ) : null}
                      </div>
                      <details className="advanced-panel">
                        <summary>Advanced</summary>
                        <div className="advanced-content">
                          <div className="grid-3" style={{ marginTop: 10 }}>
                            <div>
                              <label>Search Database</label>
                              <select
                                value={getNested(specForm, ["search", "database"], "pubmed")}
                                onChange={(e) => updateField(["search", "database"], e.target.value)}
                              >
                                <option value="pubmed">pubmed</option>
                                <option value="pmc">pmc</option>
                              </select>
                            </div>
                            <div>
                              <label>Max Results (blank = unlimited)</label>
                              <input
                                type="number"
                                value={getNested(specForm, ["search", "max_results"], "")}
                                onChange={(e) => {
                                  setSearchAdvancedTouched((prev) => ({ ...prev, maxResults: true }));
                                  const raw = e.target.value;
                                  updateField(["search", "max_results"], raw === "" ? "" : Number(raw));
                                }}
                              />
                            </div>
                            <div>
                              <label>Contact Email (NCBI)</label>
                              <input
                                value={getNested(specForm, ["search", "email"], "")}
                                onChange={(e) => {
                                  setSearchAdvancedTouched((prev) => ({ ...prev, email: true }));
                                  updateField(["search", "email"], e.target.value);
                                }}
                                placeholder="Uses global setting when omitted"
                              />
                            </div>
                          </div>
                        </div>
                      </details>
                    </div>

                    <div className="builder-section builder-section-compact">
                      <h3>Add study lists</h3>
                      <p className="section-explainer">Import known PMIDs when a curated study list already exists.</p>
                      <div className="actions" style={{ marginBottom: 10 }}>
                        <button
                          className="secondary"
                          type="button"
                          onClick={() => {
                            setStudyListFileName("");
                            setStudyListModalOpen(true);
                          }}
                        >
                          Add Study List
                        </button>
                      </div>
                    </div>

                    <div className="builder-section builder-section-strong">
                      <h3>Full Text Retrieval Source</h3>
                      <p className="section-explainer">Choose where Autonima should retrieve or load full texts for downstream screening.</p>
                      <div className="retrieval-setting-row">
                        {renderToggleControl(
                          "Load Excluded",
                          getNested(specForm, ["retrieval", "load_excluded"], false),
                          (value) => updateField(["retrieval", "load_excluded"], value)
                        )}
                      </div>
                      <div className="source-tile-grid">
                        {getRetrievalSources().map((source) => (
                          <div className="source-tile" key={`builtin-${source}`}>
                            <div>
                              <div className="source-tile-title">{source}</div>
                              <div className="source-tile-meta">Built-in retrieval source</div>
                            </div>
                            <button
                              className="icon-button danger-icon"
                              type="button"
                              title={`Remove ${source}`}
                              aria-label={`Remove ${source}`}
                              onClick={() => updateRetrievalSources(getRetrievalSources().filter((item) => item !== source))}
                            >
                              ×
                            </button>
                          </div>
                        ))}
                        {getFullTextSources().map((source, index) => (
                          <div className="source-tile" key={`custom-${index}-${source.root_path || "source"}`}>
                            <div>
                              <div className="source-tile-title">Local full-text folder</div>
                              <div className="source-tile-meta mono">{source.root_path || "No root path"}</div>
                              <div className="source-tile-meta">PMID source: {source.pmid_source || "folder_name"}</div>
                            </div>
                            <div className="source-tile-actions">
                              <button
                                className="icon-button edit-icon"
                                type="button"
                                title="Edit source"
                                aria-label="Edit source"
                                onClick={() => openEditSourceModal(index)}
                              >
                                ✎
                              </button>
                              <button
                                className="icon-button danger-icon"
                                type="button"
                                title="Remove source"
                                aria-label="Remove source"
                                onClick={() => removeFullTextSource(index)}
                              >
                                ×
                              </button>
                            </div>
                          </div>
                        ))}
                        <button className="source-tile source-tile-add" type="button" onClick={openAddSourceModal}>
                          <span className="source-add-icon">+</span>
                          <span>Add Source</span>
                        </button>
                      </div>
                    </div>

                  </>
                ) : null}

                {buildStep === "screening" ? (
                  <>
                    <h3>Screening</h3>
                    <p className="section-explainer">Screen search results in two passes: broad abstract review, then stricter full-text eligibility.</p>
                    <div className="screening-flow">
                      <div className="screening-stage abstract">
                        <div className="screening-stage-header">
                          <div>
                            <h3>1. Abstract Screening</h3>
                            <p>Broad first pass. Include studies that are plausibly relevant; avoid excluding prematurely unless clearly out of scope.</p>
                          </div>
                          <span className="criteria-summary">
                            {getCriteriaList(["screening", "abstract", "inclusion_criteria"]).length} inclusion · {getCriteriaList(["screening", "abstract", "exclusion_criteria"]).length} exclusion
                          </span>
                        </div>
                        <label>Objective</label>
                        <input
                          value={getNested(specForm, ["screening", "abstract", "objective"], "")}
                          onChange={(e) => updateField(["screening", "abstract", "objective"], e.target.value)}
                          placeholder="Identify studies that may be relevant from the abstract."
                        />
                        <div className="criteria-grid">
                          {renderCriteriaPanel(["screening", "abstract", "inclusion_criteria"], "Inclusion Criteria", "include")}
                          {renderCriteriaPanel(["screening", "abstract", "exclusion_criteria"], "Exclusion Criteria", "exclude")}
                        </div>
                        <details className="advanced-panel">
                          <summary>Advanced</summary>
                          <div className="advanced-content">
                            {renderModelSelect("Abstract Model", ["screening", "abstract", "model"], true)}
                            {renderToggleControl(
                              "Confidence Reporting",
                              getNested(specForm, ["screening", "abstract", "confidence_reporting"], true),
                              (value) => updateField(["screening", "abstract", "confidence_reporting"], value)
                            )}
                          </div>
                        </details>
                      </div>

                      <div className="screening-stage fulltext">
                        <div className="screening-stage-header">
                          <div>
                            <h3>2. Full Text Screening</h3>
                            <p>Final eligibility pass after full text is available. Use stricter criteria to decide the final study set.</p>
                          </div>
                          <span className="criteria-summary">
                            {getCriteriaList(["screening", "fulltext", "inclusion_criteria"]).length} inclusion · {getCriteriaList(["screening", "fulltext", "exclusion_criteria"]).length} exclusion
                          </span>
                        </div>
                        <label>Objective</label>
                        <input
                          value={getNested(specForm, ["screening", "fulltext", "objective"], "")}
                          onChange={(e) => updateField(["screening", "fulltext", "objective"], e.target.value)}
                          placeholder="Determine final eligibility from full text."
                        />
                        <div className="criteria-grid">
                          {renderCriteriaPanel(["screening", "fulltext", "inclusion_criteria"], "Inclusion Criteria", "include")}
                          {renderCriteriaPanel(["screening", "fulltext", "exclusion_criteria"], "Exclusion Criteria", "exclude")}
                        </div>
                        <details className="advanced-panel">
                          <summary>Advanced</summary>
                          <div className="advanced-content">
                            {renderModelSelect("Full-text Model", ["screening", "fulltext", "model"], true)}
                            {renderToggleControl(
                              "Confidence Reporting",
                              getNested(specForm, ["screening", "fulltext", "confidence_reporting"], true),
                              (value) => updateField(["screening", "fulltext", "confidence_reporting"], value)
                            )}
                          </div>
                        </details>
                      </div>
                    </div>
                  </>
                ) : null}

                {buildStep === "parsing_annotation" ? (
                  <>
                    <div className="builder-section">
                      <h3>Parsing</h3>
                      <p className="section-explainer">Parse coordinate tables into distinct analyses and contrasts.</p>
                      <div className="grid-2">
                        <div>
                          {renderToggleControl(
                            "Parse Coordinates",
                            getNested(specForm, ["parsing", "parse_coordinates"], true),
                            (value) => updateParseCoordinates(value)
                          )}
                        </div>
                      </div>
                      <details className="advanced-panel">
                        <summary>Advanced</summary>
                        <div className="advanced-content">
                          {renderModelSelect("Coordinate Model", ["parsing", "coordinate_model"], true)}
                        </div>
                      </details>
                    </div>

                    <div className="builder-section builder-section-strong">
                      <h3>Annotation</h3>
                      <p className="section-explainer">Annotate parsed analyses using shared criteria and optional named annotation rules.</p>
                      {!isParsingEnabled() ? (
                        <div className="status-msg">
                          Annotation requires parsing. Turn on Parse Coordinates to enable annotation settings.
                        </div>
                      ) : null}
                      <div className="annotation-enable-row">
                        {renderToggleControl(
                          "Annotation Enabled",
                          isAnnotationEnabled(),
                          (value) => updateField(["annotation", "enabled"], value),
                          !isParsingEnabled()
                        )}
                      </div>
                      <div className="criteria-grid annotation-criteria-grid">
                        {renderCriteriaPanel(["annotation", "inclusion_criteria"], "Common Inclusion Criteria", "include")}
                        {renderCriteriaPanel(["annotation", "exclusion_criteria"], "Common Exclusion Criteria", "exclude")}
                      </div>
                      <details className="advanced-panel">
                        <summary>Advanced</summary>
                        <div className="advanced-content">
                          <label>Prompt Type</label>
                          <select
                            value={getNested(specForm, ["annotation", "prompt_type"], "multi_analysis")}
                            onChange={(e) => updateField(["annotation", "prompt_type"], e.target.value)}
                          >
                            <option value="multi_analysis">multi_analysis</option>
                            <option value="single_analysis">single_analysis</option>
                          </select>
                          {renderToggleControl(
                            "Create All Included Annotations",
                            getNested(specForm, ["annotation", "create_all_included_annotations"], true),
                            (value) => updateField(["annotation", "create_all_included_annotations"], value)
                          )}
                          {renderModelSelect("Annotation Model", ["annotation", "model"], true)}
                          <label>Metadata Fields (newline list)</label>
                          <textarea
                            value={stringifyLines(getNested(specForm, ["annotation", "metadata_fields"], DEFAULT_ANNOTATION_METADATA_FIELDS))}
                            onChange={(e) => updateField(["annotation", "metadata_fields"], parseLines(e.target.value))}
                            placeholder={"analysis_name\nanalysis_description\ntable_caption\nstudy_title\nstudy_fulltext"}
                          />
                        </div>
                      </details>
                    </div>

                    <div className="builder-section builder-section-compact">
                      <div className="annotation-rules-header">
                        <div>
                          <h3>Annotation-Specific Rules</h3>
                          <p className="section-explainer">Add named annotation sets with their own inclusion and exclusion criteria.</p>
                        </div>
                        <button className="secondary" type="button" onClick={addAnnotationEntry}>
                          Add Annotation
                        </button>
                      </div>
                      {(getNested(specForm, ["annotation", "annotations"], []) || []).length ? (
                        <div className="project-list">
                          {(getNested(specForm, ["annotation", "annotations"], []) || []).map((annotationEntry, index) => (
                            <div key={`${annotationEntry?.name || "annotation"}-${index}`} className="project-item">
                              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                                <strong>Annotation {index + 1}</strong>
                                <button
                                  className="danger"
                                  type="button"
                                  onClick={() => removeAnnotationEntry(index)}
                                >
                                  Remove
                                </button>
                              </div>
                              <div className="grid-2">
                                <div>
                                  <label>Name</label>
                                  <input
                                    value={(annotationEntry && annotationEntry.name) || ""}
                                    onChange={(e) => updateAnnotationEntry(index, "name", e.target.value)}
                                    placeholder="annotation_name"
                                  />
                                </div>
                                <div>
                                  <label>Description</label>
                                  <input
                                    value={(annotationEntry && annotationEntry.description) || ""}
                                    onChange={(e) => updateAnnotationEntry(index, "description", e.target.value)}
                                    placeholder="All analyses"
                                  />
                                </div>
                              </div>
                              <div className="criteria-grid">
                                {renderCriteriaPanel(["annotation", "annotations", index, "inclusion_criteria"], "Inclusion Criteria", "include")}
                                {renderCriteriaPanel(["annotation", "annotations", index, "exclusion_criteria"], "Exclusion Criteria", "exclude")}
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="status-msg">
                          No annotation-specific entries yet. Add one to configure named annotations.
                        </div>
                      )}
                    </div>
                  </>
                ) : null}

                {buildStep === "review" ? (
                  <>
                    <h3>Review</h3>
                    <p className="section-explainer">Save, validate, and make final output choices before running the project.</p>
                    {!yamlMode ? (
                      <>
                        <div className="review-section-header">
                          <div>
                            <h3>Specification Review</h3>
                            <p className="section-explainer">A visual summary of the configuration that will be saved and run.</p>
                          </div>
                        </div>
                        {renderSpecificationReview()}
                      </>
                    ) : null}
                    <details className="advanced-panel">
                      <summary>Output Options</summary>
                      <div className="advanced-content grid-3">
                        <div>
                          <label>Formats (newline list)</label>
                          <textarea
                            value={stringifyLines(getNested(specForm, ["output", "formats"], ["csv", "json"]))}
                            onChange={(e) => updateField(["output", "formats"], parseLines(e.target.value))}
                          />
                        </div>
                        <div>
                          {renderToggleControl(
                            "nimads",
                            getNested(specForm, ["output", "nimads"], true),
                            (value) => updateField(["output", "nimads"], value)
                          )}
                        </div>
                        <div>
                          {renderToggleControl(
                            "Export Excluded Studies",
                            getNested(specForm, ["output", "export_excluded_studies"], true),
                            (value) => updateField(["output", "export_excluded_studies"], value)
                          )}
                        </div>
                      </div>
                    </details>

                    <div className="actions" style={{ marginBottom: 12, marginTop: 12 }}>
                      <button className="secondary" onClick={() => setYamlMode(!yamlMode)}>
                        {yamlMode ? "Switch to Review Form" : "Switch to YAML Editor"}
                      </button>
                      <button className="secondary" onClick={validateSpec}>Validate</button>
                    </div>

                    {yamlMode ? (
                      <>
                        <div className="actions" style={{ marginBottom: 10 }}>
                          <select
                            value={yamlModelSelection}
                            onChange={(e) => setYamlModelSelection(e.target.value)}
                            disabled={!preferredModels.length}
                            style={{ minWidth: 260 }}
                          >
                            {!preferredModels.length ? (
                              <option value="">No preferred models configured</option>
                            ) : null}
                            {preferredModels.map((modelName) => (
                              <option key={modelName} value={modelName}>
                                {modelName}
                              </option>
                            ))}
                          </select>
                          <button
                            className="secondary"
                            type="button"
                            onClick={insertSelectedModelValue}
                            disabled={!yamlModelSelection}
                          >
                            Insert Model Value
                          </button>
                          <button
                            className="secondary"
                            type="button"
                            onClick={insertSelectedModelLine}
                            disabled={!yamlModelSelection}
                          >
                            Insert model: line
                          </button>
                        </div>
                        <textarea
                          ref={yamlEditorRef}
                          className="mono"
                          style={{ minHeight: 420 }}
                          value={yamlText}
                          onChange={(e) => setYamlText(e.target.value)}
                        />
                      </>
                    ) : (
                      null
                    )}
                  </>
                ) : null}

                <div className="build-step-footer actions">
                  <button
                    className="secondary"
                    type="button"
                    disabled={isFirstBuildStep}
                    onClick={() => goBuildStepWithAutosave(-1)}
                  >
                    ← Back
                  </button>
                  <button
                    className="secondary"
                    type="button"
                    onClick={async () => {
                      if (isLastBuildStep) {
                        await navigateEditorTab("runs");
                      } else {
                        await goBuildStepWithAutosave(1);
                      }
                    }}
                  >
                    {isLastBuildStep ? "Run →" : "Next →"}
                  </button>
                </div>
              </div>

            </>
          )}

            {selectedProject && editorTab === "runs" && (
            <>
              {runsSubTab === "screening" ? (
                <div className="card">
                  <div className="run-card-header">
                    <div>
                      <h2>Run Screening</h2>
                      <p className="section-explainer">Start screening and monitor progress below.</p>
                    </div>
                    <button
                      className="secondary run-meta-link"
                      disabled={!metaAnalysisEnabled}
                      title={
                        metaAnalysisEnabled
                          ? "Open Meta-Analysis"
                          : "Complete a screening run with NIMADS output to enable Meta-Analysis"
                      }
                      onClick={() => {
                        if (metaAnalysisEnabled) {
                          setRunsSubTab("meta");
                        }
                      }}
                    >
                      Meta-Analysis →
                    </button>
                  </div>

                  <div className="run-primary-controls">
                    <button
                      className={`primary run-action-button ${screeningRunInProgress ? "run-action-cancel" : "run-action-start"}`}
                      onClick={screeningRunInProgress ? cancelRun : startRun}
                      disabled={screeningRunInProgress ? !currentExecutionRun : false}
                      aria-live="polite"
                    >
                      {screeningRunInProgress ? (
                        <>
                          <span className="button-icon" aria-hidden="true">■</span>
                          Cancel Screening
                        </>
                      ) : (
                        <>
                          <span className="button-icon" aria-hidden="true">▶</span>
                          Start Screening
                        </>
                      )}
                    </button>
                    <div className="run-workers-control">
                      <label>
                        Workers
                        <span
                          className="info-tooltip"
                          tabIndex="0"
                          aria-label="Start with 2-4 workers. For larger projects, 10-15 can be appropriate when rate limits and machine capacity allow."
                        >
                          i
                        </span>
                      </label>
                      <input
                        type="number"
                        min="1"
                        value={runForm.num_workers}
                        onChange={(e) => setRunForm((prev) => ({ ...prev, num_workers: Number(e.target.value || 1) }))}
                      />
                    </div>
                  </div>

                  <details className="advanced-panel">
                    <summary>Advanced</summary>
                    <div className="advanced-content">
                      <div className="grid-2">
                        <div>
                          <label>Mode</label>
                          <select
                            value={runForm.mode}
                            onChange={(e) => setRunForm((prev) => ({ ...prev, mode: e.target.value }))}
                          >
                            <option value="run">run</option>
                            <option value="run-search">run-search</option>
                            <option value="run-abstract">run-abstract</option>
                          </select>
                        </div>
                        <div>
                          <label>Output Folder (optional)</label>
                          <input
                            className="mono"
                            value={runForm.output_folder}
                            onChange={(e) => setRunForm((prev) => ({ ...prev, output_folder: e.target.value }))}
                          />
                        </div>
                      </div>
                      {selectedOutputFolder ? (
                        <div className="run-output-summary" style={{ marginTop: 10 }}>
                          <strong>Resolved output</strong>
                          <span className="mono">{selectedOutputFolder}</span>
                        </div>
                      ) : null}
                      <div className="grid-3" style={{ marginTop: 10 }}>
                        {[
                          ["verbose", "Verbose"],
                          ["dry_run", "Dry Run"],
                          ["debug", "Debug"],
                          ["force_reextract_incomplete_fulltext", "Force Reextract Incomplete"],
                          ["apply_default_email", "Apply NCBI_EMAIL Default"],
                        ].map(([key, label]) => (
                          <div key={key}>
                            {renderToggleControl(
                              label,
                              runForm[key],
                              (value) => setRunForm((prev) => ({ ...prev, [key]: value }))
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  </details>

                  {!metaAnalysisEnabled ? (
                    <div className="run-gate-note">
                      Meta-Analysis unlocks after screening produces NIMADS output.
                    </div>
                  ) : null}
                </div>
              ) : (
                <div className="card">
                  <div className="run-card-header">
                    <div>
                      <h2>Run Meta-Analysis</h2>
                      <p className="section-explainer">Run a coordinate-based meta-analysis from screening output.</p>
                    </div>
                    <button
                      className="secondary run-meta-link"
                      onClick={() => setRunsSubTab("screening")}
                    >
                      ← Screening
                    </button>
                  </div>

                  {!metaAnalysisEnabled ? (
                    <div className="status-msg">
                      Meta-Analysis is currently unavailable for this project.
                    </div>
                  ) : (
                    <>
                      <div className="run-primary-controls">
                        <button
                          className={`primary run-action-button ${metaRunInProgress ? "run-action-cancel" : "run-action-start"}`}
                          onClick={metaRunInProgress ? cancelRun : startMetaRun}
                          disabled={metaRunInProgress ? !currentExecutionRun : !metaForm.output_folder}
                          aria-live="polite"
                        >
                          {metaRunInProgress ? (
                            <>
                              <span className="button-icon" aria-hidden="true">■</span>
                              Cancel Meta-Analysis
                            </>
                          ) : (
                            <>
                              <span className="button-icon" aria-hidden="true">▶</span>
                              Start Meta-Analysis
                            </>
                          )}
                        </button>
                        <div className="run-select-control">
                          <label>Estimator</label>
                          <select
                            value={metaForm.estimator}
                            onChange={(e) => setMetaForm((prev) => ({ ...prev, estimator: e.target.value }))}
                          >
                            <option value="mkdadensity">mkdadensity</option>
                            <option value="ale">ale</option>
                            <option value="kda">kda</option>
                          </select>
                        </div>
                        <div className="run-select-control">
                          <label>Corrector</label>
                          <select
                            value={metaForm.corrector}
                            onChange={(e) => setMetaForm((prev) => ({ ...prev, corrector: e.target.value }))}
                          >
                            <option value="fdr">fdr</option>
                            <option value="montecarlo">montecarlo</option>
                            <option value="bonferroni">bonferroni</option>
                          </select>
                        </div>
                      </div>

                      <details className="advanced-panel">
                        <summary>Advanced</summary>
                        <div className="advanced-content">
                          <div className="grid-3">
                            <div>
                              <label>Output Folder</label>
                              <input
                                className="mono"
                                value={metaForm.output_folder}
                                onChange={(e) => setMetaForm((prev) => ({ ...prev, output_folder: e.target.value }))}
                              />
                            </div>
                            <div>
                              <label>Estimator Args (JSON)</label>
                              <input
                                className="mono"
                                value={metaForm.estimator_args}
                                onChange={(e) => setMetaForm((prev) => ({ ...prev, estimator_args: e.target.value }))}
                              />
                            </div>
                            <div>
                              <label>Corrector Args (JSON)</label>
                              <input
                                className="mono"
                                value={metaForm.corrector_args}
                                onChange={(e) => setMetaForm((prev) => ({ ...prev, corrector_args: e.target.value }))}
                              />
                            </div>
                          </div>
                          <div className="grid-3" style={{ marginTop: 10 }}>
                            <div>
                              <label>Include IDs File (optional)</label>
                              <input
                                className="mono"
                                value={metaForm.include_ids}
                                onChange={(e) => setMetaForm((prev) => ({ ...prev, include_ids: e.target.value }))}
                              />
                            </div>
                            <div>
                              {renderToggleControl(
                                "Run Reports",
                                metaForm.run_reports,
                                (value) => setMetaForm((prev) => ({ ...prev, run_reports: value }))
                              )}
                            </div>
                            <div>
                              {renderToggleControl(
                                "Fail Fast",
                                metaForm.fail_fast,
                                (value) => setMetaForm((prev) => ({ ...prev, fail_fast: value }))
                              )}
                            </div>
                          </div>
                          <div className="grid-3" style={{ marginTop: 10 }}>
                            <div>
                              {renderToggleControl(
                                "Debug",
                                metaForm.debug,
                                (value) => setMetaForm((prev) => ({ ...prev, debug: value }))
                              )}
                            </div>
                          </div>
                          <div className="run-output-summary" style={{ marginTop: 10 }}>
                            <strong>NIMADS source</strong>
                            <span className="mono">
                              {eligibleMetaRun.progress?.nimads_studyset_path || `${eligibleMetaRun.output_folder}/outputs/nimads_studyset.json`}
                            </span>
                          </div>
                        </div>
                      </details>
                    </>
                  )}
                </div>
              )}

              <div className="card current-execution-card">
                <div className="current-execution-header">
                  <div>
                    <h3>Current Execution</h3>
                    {viewingPreviousRun ? (
                      <p className="section-explainer">Viewing previous run. A new active run will automatically take focus.</p>
                    ) : (
                      <p className="section-explainer">Latest execution for this project specification.</p>
                    )}
                  </div>
                  {currentExecutionRun && !(runsSubTab === "meta" && currentExecutionRun.status === "completed") ? (
                    <span
                      className={`${statusClass(currentExecutionRun.status)} ${isActiveRunStatus(currentExecutionRun.status) ? "active-now" : ""}`}
                    >
                      {isActiveRunStatus(currentExecutionRun.status) ? "Active now" : currentExecutionRun.status}
                    </span>
                  ) : null}
                </div>

                {!currentExecutionRun ? (
                  <div className="status-msg">
                    {runsSubTab === "meta"
                      ? "No meta-analysis runs yet. Start meta-analysis to monitor progress here."
                      : "No runs yet. Start screening to monitor progress here."}
                  </div>
                ) : runsSubTab === "meta" ? (
                  <>
                    {resolvedMetaRun?.status !== "completed" ? (
                      <div className="meta-run-status-card">
                        <div className={`meta-status-indicator ${resolvedMetaRun?.status}`}>
                          <div className="meta-status-circle" aria-hidden="true" />
                          <span className="meta-status-label">{resolvedMetaRun?.status || "unknown"}</span>
                        </div>
                      </div>
                    ) : null}
                    <div className="meta-artifacts-panel">
                      <h4>Live Meta Map Viewer</h4>
                      {metaArtifactsLoading ? (
                        <div className="status-msg">Loading meta-analysis maps…</div>
                      ) : null}
                      {metaArtifactsError ? (
                        <div className="status-msg error">{metaArtifactsError}</div>
                      ) : null}
                      {!metaArtifactsLoading && !metaArtifactsError && !metaArtifacts.length ? (
                        <div className="status-msg">
                          No NIfTI output maps were found yet for this run.
                        </div>
                      ) : null}

                      {!metaArtifactsLoading && !metaArtifactsError && metaArtifacts.length ? (
                        <div className="meta-artifacts-shell">
                          <div className="meta-annotation-tabs">
                            {metaArtifactGroups.map((group) => (
                              <button
                                key={group.id}
                                type="button"
                                className={activeMetaArtifactGroup === group.id ? "active" : ""}
                                onClick={() => {
                                  const preferred = preferredMetaArtifact(group.files);
                                  setSelectedMetaArtifactGroup(group.id);
                                  setSelectedMetaArtifactPath(preferred?.relative_path || "");
                                }}
                              >
                                <span>{group.label}</span>
                                <strong>{group.files.length}</strong>
                              </button>
                            ))}
                          </div>
                          <div className="meta-artifacts-layout">
                            <div className="meta-artifacts-list">
                              {activeMetaArtifacts.map((file) => {
                                const isActive = selectedMetaArtifact?.relative_path === file.relative_path;
                                return (
                                  <button
                                    key={file.relative_path}
                                    type="button"
                                    className={`meta-artifact-item ${isActive ? "active" : ""}`}
                                    onClick={() => setSelectedMetaArtifactPath(file.relative_path)}
                                  >
                                    <span className="meta-artifact-name">{file.name}</span>
                                    <span className="meta-artifact-size">{formatBytes(file.size_bytes)}</span>
                                  </button>
                                );
                              })}
                            </div>
                            <div className="meta-artifacts-viewer">
                              {selectedMetaArtifactUrl ? (
                                <MetaNiftiViewer
                                  fileUrl={selectedMetaArtifactUrl}
                                  fileName={selectedMetaArtifact?.name || ""}
                                />
                              ) : (
                                <div className="status-msg">Select a map to preview.</div>
                              )}
                            </div>
                          </div>
                        </div>
                      ) : null}
                    </div>

                    <details className="advanced-panel live-logs-panel">
                      <summary>Live Logs</summary>
                      <div className="advanced-content">
                        <div className="log-box log-box-prominent">
                          {logs.length ? logs.join("\n") : "No logs yet."}
                        </div>
                      </div>
                    </details>
                  </>
                ) : (
                  <>
                    <div className="timeline" style={{ marginBottom: 12 }}>
                      {(currentExecutionRun.progress?.timeline || []).map((stage) => (
                        <div key={stage.stage} className={`stage ${stage.status}`}>
                          <div style={{ fontWeight: 700 }}>{stage.stage}</div>
                          <div>{stage.status}</div>
                        </div>
                      ))}
                    </div>

                    <h4>Results</h4>
                    {renderCounterFunnel(
                      currentExecutionRun.progress?.counters || {},
                      currentExecutionRun.progress?.live_progress || null
                    )}

                    {(currentExecutionRun.progress?.missing_fulltexts?.available) ? (
                      <div className="missing-fulltexts-panel">
                        <div className="missing-fulltexts-header">
                          <h4>Missing Full Texts</h4>
                          <span className="missing-fulltexts-count">
                            {currentExecutionRun.progress?.missing_fulltexts?.count || 0} pending
                          </span>
                        </div>
                        <p className="section-explainer">
                          Download missing IDs, fetch HTML full texts, add them as a manual retrieval source in Find studies, then rerun screening.
                        </p>
                        <div className="missing-fulltexts-actions">
                          {currentExecutionRun.progress?.missing_fulltexts?.txt_path ? (
                            <a
                              className="secondary"
                              href={buildMissingFulltextsUrl(currentExecutionRun.id, "txt")}
                            >
                              Download PMID list (.txt)
                            </a>
                          ) : null}
                          {currentExecutionRun.progress?.missing_fulltexts?.csv_path ? (
                            <a
                              className="secondary"
                              href={buildMissingFulltextsUrl(currentExecutionRun.id, "csv")}
                            >
                              Download details (.csv)
                            </a>
                          ) : null}
                        </div>

                        {(currentExecutionRun.progress?.missing_fulltexts?.preview_pmids || []).length ? (
                          <details className="missing-fulltexts-preview">
                            <summary>
                              Preview PMIDs ({currentExecutionRun.progress?.missing_fulltexts?.count || 0} missing)
                            </summary>
                            <div className="mono">
                              {(currentExecutionRun.progress?.missing_fulltexts?.preview_pmids || []).join(", ")}
                            </div>
                          </details>
                        ) : null}
                      </div>
                    ) : null}

                    {(currentExecutionRun.progress?.log_issues?.error_count || currentExecutionRun.progress?.log_issues?.warning_count) ? (
                      <>
                        <h4>Errors & Warnings</h4>
                        <div className="run-issues-panel">
                          {(currentExecutionRun.progress?.log_issues?.errors || []).length ? (
                            <div className="run-issues-group run-issues-errors">
                              <strong>
                                Errors ({currentExecutionRun.progress?.log_issues?.error_count || 0})
                              </strong>
                              <ul>
                                {(currentExecutionRun.progress?.log_issues?.errors || []).map((line, idx) => (
                                  <li key={`err-${idx}`} className="mono">{line}</li>
                                ))}
                              </ul>
                            </div>
                          ) : null}

                          {(currentExecutionRun.progress?.log_issues?.warnings || []).length ? (
                            <div className="run-issues-group run-issues-warnings">
                              <strong>
                                Warnings ({currentExecutionRun.progress?.log_issues?.warning_count || 0})
                              </strong>
                              <ul>
                                {(currentExecutionRun.progress?.log_issues?.warnings || []).map((line, idx) => (
                                  <li key={`warn-${idx}`} className="mono">{line}</li>
                                ))}
                              </ul>
                            </div>
                          ) : null}
                        </div>
                      </>
                    ) : null}

                    <details className="advanced-panel live-logs-panel">
                      <summary>Live Logs</summary>
                      <div className="advanced-content">
                        <div className="log-box log-box-prominent">
                          {logs.length ? logs.join("\n") : "No logs yet."}
                        </div>
                      </div>
                    </details>
                  </>
                )}
              </div>

              <details className="advanced-panel previous-runs-panel">
                <summary>Previous Runs ({previousRunsForActiveTab.length})</summary>
                <div className="advanced-content">
                  <div className="project-list previous-runs-list">
                    {previousRunsForActiveTab.map((run) => (
                      <div
                        key={run.id}
                        className={`project-item clickable-item ${selectedRunId === run.id ? "active" : ""}`}
                        onClick={() => {
                          setSelectedRunId(run.id);
                          setSelectedRun(run);
                          setLogs([]);
                          logOffsetRef.current = 0;
                          setLogOffset(0);
                        }}
                      >
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                          <div className="run-history-title">
                            <strong>{run.kind === "meta" ? "Meta-Analysis" : "Screening"} · {run.mode}</strong>
                            {run.progress?.nimads_available ? (
                              <span className="badge completed">NIMADS</span>
                            ) : null}
                          </div>
                          <span className={statusClass(run.status)}>{run.status}</span>
                        </div>
                        <div className="mono" style={{ fontSize: 12 }}>
                          {run.output_folder || "-"}
                        </div>
                      </div>
                    ))}
                    {!previousRunsForActiveTab.length ? (
                      <div className="status-msg">
                        {runsForActiveTab.length
                          ? "No older runs for this tab."
                          : (runsSubTab === "meta" ? "No meta-analysis runs yet." : "No screening runs yet.")}
                      </div>
                    ) : null}
                  </div>
                </div>
              </details>
            </>
          )}
          </div>
        </div>
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
