(() => {
  "use strict";

  const ACTIVE_JOB_KEY = "anthill.activeJob";
  const THEME_KEY = "anthill.theme";
  const form = document.getElementById("generate-form");
  const storyField = document.getElementById("story");
  const storyError = document.getElementById("story-error");
  const storyCount = document.getElementById("story-count");
  const exampleStoryButton = document.getElementById("example-story");
  const generateButton = document.getElementById("generate-button");
  const emptyOutput = document.getElementById("empty-output");
  const jobOutput = document.getElementById("job-output");
  const stateIcon = document.getElementById("job-state-icon");
  const stateTitle = document.getElementById("job-state-title");
  const progressText = document.getElementById("job-progress-text");
  const percentText = document.getElementById("job-percent");
  const progressTrack = document.getElementById("progress-track");
  const progressBar = document.getElementById("progress-bar");
  const videoResult = document.getElementById("video-result");
  const videoPlayer = document.getElementById("video-player");
  const downloadLink = document.getElementById("download-link");
  const errorPanel = document.getElementById("job-error");
  const errorText = document.getElementById("job-error-text");
  const retryButton = document.getElementById("retry-button");
  const recentList = document.getElementById("recent-list");
  const recentEmpty = document.getElementById("recent-empty");
  const recentCount = document.getElementById("recent-count");
  const themeSelect = document.getElementById("theme-select");
  const readinessPanel = document.getElementById("model-readiness");
  const readinessIcon = document.getElementById("readiness-icon");
  const readinessTitle = document.getElementById("readiness-title");
  const readinessDescription = document.getElementById("readiness-description");
  const readinessSetup = document.getElementById("readiness-setup");
  const readinessRefresh = document.getElementById("readiness-refresh");
  const modelList = document.getElementById("model-list");

  let activeJobId = null;
  let activeStory = "";
  let activeDuration = 30;
  let pollTimer = null;
  let pollFailures = 0;
  let isSubmitting = false;
  let jobInProgress = false;
  let recentJobs = [];
  let readinessStatus = "unknown";

  const exampleStory = "A man recovering from surgery felt frightened about a possible cancer diagnosis. At his usual breakfast diner, a kind waitress quietly paid for his meal. He was moved by the unexpected kindness. Two days later, his doctor told him he was cancer-free. He never forgot that she had helped him through a difficult moment.";

  function setTheme(theme) {
    const selected = ["system", "light", "dark"].includes(theme) ? theme : "system";
    document.documentElement.dataset.theme = selected;
    themeSelect.value = selected;
    try {
      localStorage.setItem(THEME_KEY, selected);
    } catch (_) {
      // Theme still applies for this visit when storage is unavailable.
    }
  }

  function showError(message) {
    errorPanel.hidden = false;
    errorText.textContent = message || "Please try again. If the problem continues, check the application logs.";
    retryButton.hidden = !activeJobId;
  }

  function showReadiness(payload) {
    const models = payload && payload.models && typeof payload.models === "object" ? payload.models : {};
    const modelEntries = Object.entries(models).filter(([, item]) => item && typeof item === "object");
    const missing = modelEntries.filter(([, item]) => item.installed !== true);
    const isReady = payload && payload.status === "ready" && modelEntries.length > 0 && missing.length === 0;

    readinessPanel.hidden = false;
    readinessPanel.classList.toggle("is-missing", !isReady && modelEntries.length > 0);
    readinessPanel.classList.toggle("is-unavailable", false);
    readinessSetup.hidden = isReady || modelEntries.length === 0;
    readinessIcon.textContent = isReady ? "✓" : "!";
    modelList.replaceChildren();
    for (const [key, model] of modelEntries) {
      const chip = document.createElement("li");
      chip.className = `model-chip${model.installed === true ? "" : " is-missing"}`;
      const displayName = typeof model.name === "string" ? model.name : key;
      chip.textContent = `${displayName} · ${model.installed === true ? "ready" : "needs setup"}`;
      modelList.append(chip);
    }

    if (isReady) {
      readinessStatus = "ready";
      readinessTitle.textContent = "Your local studio is ready";
      readinessDescription.textContent = "All generation models are installed on this device.";
      generateButton.disabled = isSubmitting || jobInProgress;
      if (generateButton.querySelector("span:nth-child(2)").textContent === "Models need setup") {
        generateButton.querySelector("span:nth-child(2)").textContent = "Generate my video";
      }
    } else {
      readinessStatus = "models-needed";
      readinessTitle.textContent = "A few local models need setup";
      readinessDescription.textContent = missing.length
        ? "Install the models listed below before generating a video."
        : "The studio has not confirmed its local models yet. Check again in a moment.";
      if (!jobInProgress && !isSubmitting) {
        generateButton.disabled = true;
        generateButton.querySelector("span:nth-child(2)").textContent = "Models need setup";
      }
    }
  }

  async function checkReadiness() {
    readinessRefresh.disabled = true;
    readinessRefresh.textContent = "Checking…";
    try {
      const response = await fetch("/api/readiness", { headers: { Accept: "application/json" } });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "The local studio is temporarily unavailable.");
      showReadiness(payload);
    } catch (_) {
      readinessStatus = "unknown";
      readinessPanel.hidden = false;
      readinessPanel.classList.remove("is-missing");
      readinessPanel.classList.add("is-unavailable");
      readinessIcon.textContent = "!";
      readinessTitle.textContent = "Could not check local models";
      readinessDescription.textContent = "Check that the studio is running, then check again. You can still submit a story.";
      readinessSetup.hidden = true;
      // A failed readiness request can be a temporary connection issue. Let the server decide on submission.
      if (generateButton.querySelector("span:nth-child(2)").textContent === "Models need setup") {
        generateButton.disabled = isSubmitting || jobInProgress;
        generateButton.querySelector("span:nth-child(2)").textContent = "Generate my video";
      }
    } finally {
      readinessRefresh.disabled = false;
      readinessRefresh.textContent = "Check again";
    }
  }

  function setProgress(percent) {
    const normalized = Number.isFinite(Number(percent)) ? Math.max(0, Math.min(100, Number(percent))) : 0;
    progressBar.style.width = `${normalized}%`;
    progressTrack.setAttribute("aria-valuenow", String(Math.round(normalized)));
    percentText.textContent = normalized > 0 && normalized < 100 ? `${Math.round(normalized)}%` : "";
  }

  function resetOutput() {
    emptyOutput.hidden = true;
    jobOutput.hidden = false;
    videoResult.hidden = true;
    errorPanel.hidden = true;
    progressTrack.hidden = false;
    progressText.textContent = "";
    stateTitle.textContent = "Getting things ready…";
    stateIcon.textContent = "◌";
    stateIcon.className = "job-state-icon";
    setProgress(0);
  }

  function cleanVideoUrl(url) {
    if (typeof url !== "string" || !url.startsWith("/videos/")) return null;
    try {
      const parsed = new URL(url, window.location.origin);
      if (parsed.origin !== window.location.origin) return null;
      return `${parsed.pathname}${parsed.search}`;
    } catch (_) {
      return null;
    }
  }

  function updateJob(job) {
    if (!job || typeof job !== "object") return;
    if (job.id && activeJobId && job.id !== activeJobId) return;

    if (job.status === "queued" || job.status === "working") {
      emptyOutput.hidden = true;
      jobOutput.hidden = false;
      progressTrack.hidden = false;
      errorPanel.hidden = true;
      videoResult.hidden = true;
      stateIcon.className = "job-state-icon";
      stateIcon.textContent = job.status === "queued" ? "···" : "◌";
      stateTitle.textContent = job.status === "queued" ? "In the queue" : "Making your video";
      progressText.textContent = typeof job.progress === "string" ? job.progress : "Your story is moving through the studio.";
      setProgress(job.percent);
      generateButton.disabled = true;
      jobInProgress = true;
      generateButton.querySelector("span:nth-child(2)").textContent = "Video in progress";
      return;
    }

    if (job.status === "done") {
      const videoUrl = cleanVideoUrl(job.video);
      if (!videoUrl) {
        updateJob({ ...job, status: "error", error: "The finished video is not available yet. Please refresh and try again." });
        return;
      }
      emptyOutput.hidden = true;
      jobOutput.hidden = false;
      errorPanel.hidden = true;
      progressTrack.hidden = true;
      videoResult.hidden = false;
      stateIcon.className = "job-state-icon is-done";
      stateIcon.textContent = "✓";
      stateTitle.textContent = "Your video is ready";
      progressText.textContent = `Your video is ready. ${Number(job.duration_seconds) || activeDuration} seconds · Portrait MP4`;
      percentText.textContent = "";
      videoPlayer.src = videoUrl;
      videoPlayer.load();
      downloadLink.href = videoUrl;
      jobInProgress = false;
      generateButton.disabled = readinessStatus === "models-needed";
      generateButton.querySelector("span:nth-child(2)").textContent = "Generate another video";
      try { localStorage.removeItem(ACTIVE_JOB_KEY); } catch (_) { /* Continue without recovery storage. */ }
      clearTimeout(pollTimer);
      loadRecent();
      return;
    }

    if (job.status === "error") {
      emptyOutput.hidden = true;
      jobOutput.hidden = false;
      progressTrack.hidden = true;
      videoResult.hidden = true;
      stateIcon.className = "job-state-icon is-error";
      stateIcon.textContent = "!";
      stateTitle.textContent = "Needs another try";
      progressText.textContent = "Your story is saved with this job.";
      percentText.textContent = "";
      jobInProgress = false;
      generateButton.disabled = readinessStatus === "models-needed";
      generateButton.querySelector("span:nth-child(2)").textContent = "Generate another video";
      showError(job.error);
      clearTimeout(pollTimer);
      loadRecent();
      return;
    }
  }

  function displayRecent(jobs) {
    recentJobs = Array.isArray(jobs) ? jobs.slice(0, 20) : [];
    recentList.replaceChildren();
    recentCount.textContent = String(recentJobs.length);
    recentEmpty.hidden = recentJobs.length > 0;

    for (const job of recentJobs) {
      if (!job || typeof job.id !== "string") continue;
      const item = document.createElement("li");
      item.className = "recent-item";
      const select = document.createElement("button");
      select.className = "recent-item-button";
      select.type = "button";
      select.setAttribute("aria-label", `Open ${job.status === "done" ? "finished" : job.status === "error" ? "failed" : "in progress"} ${Number(job.duration_seconds) || 30} second video`);
      const icon = document.createElement("span");
      icon.className = "recent-icon";
      icon.setAttribute("aria-hidden", "true");
      icon.textContent = job.status === "done" ? "▶" : job.status === "error" ? "!" : "◌";
      const copy = document.createElement("span");
      copy.className = "recent-copy";
      const name = document.createElement("span");
      name.className = "recent-name";
      name.textContent = `Story video · ${Number(job.duration_seconds) || 30} sec`;
      const detail = document.createElement("span");
      detail.className = "recent-detail";
      const date = job.updated_at || job.created_at;
      detail.textContent = date ? new Date(date).toLocaleString() : "Created on this device";
      copy.append(name, detail);
      select.append(icon, copy);
      select.addEventListener("click", () => selectRecent(job));
      const status = document.createElement("span");
      status.className = `recent-status ${job.status === "done" ? "done" : job.status === "error" ? "error" : ""}`;
      status.textContent = job.status === "done" ? "Ready" : job.status === "error" ? "Failed" : job.status === "queued" ? "Queued" : "Making";
      item.append(select, status);
      recentList.append(item);
    }
  }

  function selectRecent(job) {
    if (!job || typeof job.id !== "string") return;
    activeJobId = job.id;
    activeDuration = Number(job.duration_seconds) || 30;
    resetOutput();
    if (job.status === "done" || job.status === "error") {
      updateJob(job);
      return;
    }
    try { localStorage.setItem(ACTIVE_JOB_KEY, activeJobId); } catch (_) { /* Polling still works. */ }
    pollJob(activeJobId);
  }

  async function loadRecent() {
    try {
      const response = await fetch("/api/jobs", { headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error("Recent videos could not be loaded.");
      const payload = await response.json();
      displayRecent(payload.jobs);
    } catch (_) {
      recentEmpty.hidden = false;
      recentEmpty.textContent = "Recent videos are temporarily unavailable. Refresh to try again.";
    }
  }

  async function pollJob(jobId) {
    clearTimeout(pollTimer);
    try {
      const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, { headers: { Accept: "application/json" } });
      const job = await response.json();
      if (!response.ok) throw new Error(job.error || "This video could not be found.");
      pollFailures = 0;
      updateJob(job);
      if (job.status === "working" || job.status === "queued") {
        pollTimer = setTimeout(() => pollJob(jobId), 2200);
      }
    } catch (error) {
      pollFailures += 1;
      if (pollFailures <= 5) {
        progressText.textContent = "Connection interrupted. Checking again…";
        pollTimer = setTimeout(() => pollJob(jobId), Math.min(12000, 1600 * pollFailures));
      } else {
        generateButton.disabled = false;
        showError(error instanceof TypeError ? "We lost connection while checking on this video. Refresh to reconnect." : error.message);
      }
    }
  }

  async function createJob() {
    if (isSubmitting) return;
    const story = storyField.value.trim();
    storyError.textContent = "";
    if (story.length < 25) {
      storyError.textContent = "Add at least 25 characters so the story has enough detail.";
      storyField.focus();
      return;
    }
    if (story.length > 6000) {
      storyError.textContent = "Keep the story under 6,000 characters.";
      storyField.focus();
      return;
    }

    activeJobId = null;
    isSubmitting = true;
    jobInProgress = true;
    activeStory = story;
    activeDuration = Number(new FormData(form).get("duration")) || 30;
    generateButton.disabled = true;
    generateButton.querySelector("span:nth-child(2)").textContent = "Starting your video…";
    emptyOutput.hidden = true;
    jobOutput.hidden = false;
    resetOutput();
    progressText.textContent = "Sending your story to the local studio…";
    try {
      const response = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ story, duration_seconds: activeDuration }),
      });
      const payload = await response.json();
      if (!response.ok) {
        if (response.status === 503 && payload.models) {
          showReadiness({ status: "models-needed", models: payload.models });
          throw new Error("Install the missing local models shown above, then try again.");
        }
        throw new Error(payload.error || "We couldn't start this video. Please try again.");
      }
      if (typeof payload.id !== "string") throw new Error("The studio did not return a video job. Please try again.");
      activeJobId = payload.id;
      try { localStorage.setItem(ACTIVE_JOB_KEY, activeJobId); } catch (_) { /* Job stays on server. */ }
      pollFailures = 0;
      pollJob(activeJobId);
      loadRecent();
    } catch (error) {
      showError(error instanceof TypeError ? "We couldn't reach the local studio. Check that it is running, then try again." : error.message);
      stateTitle.textContent = "Could not start the video";
      jobInProgress = false;
      if (readinessStatus !== "models-needed") {
        generateButton.disabled = false;
        generateButton.querySelector("span:nth-child(2)").textContent = "Try again";
      } else {
        generateButton.disabled = true;
        generateButton.querySelector("span:nth-child(2)").textContent = "Models need setup";
      }
    } finally {
      isSubmitting = false;
    }
  }

  async function retryJob() {
    if (!activeJobId) return;
    retryButton.disabled = true;
    retryButton.textContent = "Restarting…";
    errorPanel.hidden = true;
    resetOutput();
    jobInProgress = true;
    stateTitle.textContent = "Starting another try…";
    try {
      const response = await fetch(`/api/jobs/${encodeURIComponent(activeJobId)}/retry`, {
        method: "POST",
        headers: { Accept: "application/json" },
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "This video cannot be restarted yet.");
      if (typeof payload.id === "string") activeJobId = payload.id;
      try { localStorage.setItem(ACTIVE_JOB_KEY, activeJobId); } catch (_) { /* Continue without recovery storage. */ }
      pollFailures = 0;
      pollJob(activeJobId);
      loadRecent();
    } catch (error) {
      jobInProgress = false;
      showError(error instanceof TypeError ? "We couldn't reach the local studio. Check that it is running, then try again." : error.message);
    } finally {
      retryButton.disabled = false;
      retryButton.textContent = "Try again";
    }
  }

  storyField.addEventListener("input", () => {
    storyCount.textContent = `${storyField.value.length.toLocaleString()} / 6,000`;
    if (storyError.textContent && storyField.value.trim().length >= 25) storyError.textContent = "";
  });
  exampleStoryButton.addEventListener("click", () => {
    storyField.value = exampleStory;
    storyCount.textContent = `${storyField.value.length.toLocaleString()} / 6,000`;
    storyError.textContent = "";
    storyField.focus();
  });
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    createJob();
  });
  retryButton.addEventListener("click", retryJob);
  themeSelect.addEventListener("change", () => setTheme(themeSelect.value));
  readinessRefresh.addEventListener("click", checkReadiness);

  setTheme((() => {
    try { return localStorage.getItem(THEME_KEY) || "system"; } catch (_) { return "system"; }
  })());
  storyCount.textContent = `${storyField.value.length.toLocaleString()} / 6,000`;
  checkReadiness();
  loadRecent().then(() => {
    let savedJobId = null;
    try { savedJobId = localStorage.getItem(ACTIVE_JOB_KEY); } catch (_) { /* Existing recent job list remains available. */ }
    if (savedJobId) {
      activeJobId = savedJobId;
      resetOutput();
      generateButton.disabled = true;
      generateButton.querySelector("span:nth-child(2)").textContent = "Video in progress";
      pollJob(activeJobId);
    }
  });
})();
