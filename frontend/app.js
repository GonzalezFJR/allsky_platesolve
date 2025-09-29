const canvas = document.getElementById("skyCanvas");
const ctx = canvas.getContext("2d");
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const browseButton = document.getElementById("browseButton");
const noImageOverlay = document.getElementById("noImageOverlay");
const viewerInfo = document.getElementById("viewerInfo");
const observationForm = document.getElementById("observationForm");
const detectionForm = document.getElementById("detectionForm");
const toggleDetections = document.getElementById("toggleDetections");
const toggleExpected = document.getElementById("toggleExpected");
const toggleLabels = document.getElementById("toggleLabels");
const matchesList = document.getElementById("matchesList");
const confirmMatch = document.getElementById("confirmMatch");
const selectedDetectionInput = document.getElementById("selectedDetection");
const starNameInput = document.getElementById("starNameInput");
const starNamesDataList = document.getElementById("starNames");
const downloadButton = document.getElementById("downloadParams");
const resetDetectionsButton = document.getElementById("resetDetections");
const clearMatchesButton = document.getElementById("clearMatches");
const toast = document.getElementById("toast");
const modelSelect = document.getElementById("modelType");
const isNadirCheckbox = document.getElementById("isNadir");

const paramZenith = document.getElementById("paramZenith");
const paramTheta = document.getElementById("paramTheta");
const paramTilt = document.getElementById("paramTilt");
const paramTiltAz = document.getElementById("paramTiltAz");
const paramCoeffs = document.getElementById("paramCoeffs");

const state = {
  imageElement: null,
  imageDataUrl: null,
  imageWidth: 0,
  imageHeight: 0,
  flipHorizontal: false,
  detections: [],
  expectedStars: [],
  matches: [],
  selectedDetection: null,
  showDetections: true,
  showExpected: true,
  showLabels: true,
  radialModel: "poly2",
  modelParams: null,
  observation: null,
  starNames: [],
};

function showToast(message, variant = "info") {
  toast.textContent = message;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 2500);
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || response.statusText);
  }
  return response.json();
}

function normalizeModel(apiModel) {
  return {
    modelType: apiModel.model_type,
    xc: apiModel.xc,
    yc: apiModel.yc,
    theta: apiModel.theta_deg,
    tiltAngle: apiModel.tilt_angle,
    tiltAzimuth: apiModel.tilt_azimuth,
    coeffs: apiModel.coeffs,
    maxRadiusPx: apiModel.max_radius_px,
  };
}

function updateModelSummary() {
  if (!state.modelParams) {
    paramZenith.textContent = "—";
    paramTheta.textContent = "—";
    paramTilt.textContent = "—";
    paramTiltAz.textContent = "—";
    paramCoeffs.textContent = "—";
    downloadButton.disabled = true;
    return;
  }

  const mp = state.modelParams;
  paramZenith.textContent = `${mp.xc.toFixed(2)}, ${mp.yc.toFixed(2)}`;
  paramTheta.textContent = `${mp.theta.toFixed(2)}`;
  paramTilt.textContent = `${mp.tiltAngle.toFixed(2)}`;
  paramTiltAz.textContent = `${mp.tiltAzimuth.toFixed(2)}`;
  paramCoeffs.textContent = mp.coeffs.map((c) => c.toFixed(5)).join(", ");
  downloadButton.disabled = false;
}

function clearCanvas() {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function drawScene() {
  if (!state.imageElement) {
    clearCanvas();
    return;
  }

  canvas.width = state.imageWidth;
  canvas.height = state.imageHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  if (state.flipHorizontal) {
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(state.imageElement, 0, 0, canvas.width, canvas.height);
  ctx.restore();

  // Overlay detections
  if (state.showExpected) {
    ctx.lineWidth = 2;
    for (const star of state.expectedStars) {
      ctx.beginPath();
      ctx.strokeStyle = "rgba(64, 255, 162, 0.85)";
      ctx.arc(star.x, star.y, 10, 0, Math.PI * 2);
      ctx.stroke();
      if (state.showLabels) {
        ctx.fillStyle = "rgba(64, 255, 162, 0.85)";
        ctx.font = "14px 'Inter', sans-serif";
        ctx.fillText(star.name, star.x + 12, star.y + 4);
      }
    }
  }

  if (state.showDetections) {
    ctx.lineWidth = 2;
    for (let i = 0; i < state.detections.length; i++) {
      const det = state.detections[i];
      ctx.beginPath();
      ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
      ctx.arc(det.x, det.y, 8, 0, Math.PI * 2);
      ctx.stroke();
      if (state.selectedDetection === i) {
        ctx.strokeStyle = "rgba(72, 149, 239, 0.95)";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(det.x, det.y, 11, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }
}

function updateViewerInfo() {
  const detections = state.detections.length;
  const expected = state.expectedStars.length;
  const matches = state.matches.length;
  viewerInfo.textContent = `Detecciones: ${detections} · Estrellas esperadas: ${expected} · Identificadas: ${matches}`;
}

function resetDetections() {
  state.detections = [];
  state.matches = [];
  state.selectedDetection = null;
  selectedDetectionInput.value = "";
  starNameInput.value = "";
  matchesList.innerHTML = "";
  confirmMatch.disabled = true;
  drawScene();
  updateViewerInfo();
}

function resetExpected() {
  state.expectedStars = [];
  drawScene();
  updateViewerInfo();
}

function updateMatchesList() {
  matchesList.innerHTML = "";
  state.matches.forEach((match, index) => {
    const li = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = `${match.name} → (${match.x.toFixed(1)}, ${match.y.toFixed(1)})`;
    const removeBtn = document.createElement("button");
    removeBtn.textContent = "Quitar";
    removeBtn.addEventListener("click", () => {
      state.matches.splice(index, 1);
      state.selectedDetection = null;
      selectedDetectionInput.value = "";
      confirmMatch.disabled = true;
      updateMatchesList();
      drawScene();
      updateViewerInfo();
      if (state.matches.length) {
        fitModel();
      } else {
        updateExpectedStars();
      }
    });
    li.appendChild(label);
    li.appendChild(removeBtn);
    matchesList.appendChild(li);
  });
  updateViewerInfo();
}

async function fetchRadialModels() {
  try {
    const models = await fetchJSON("/api/radial-models");
    const current = state.radialModel;
    modelSelect.innerHTML = "";
    models.forEach((model) => {
      const option = document.createElement("option");
      option.value = model.key;
      option.textContent = `${model.name} (${model.coefficients} coef.)`;
      modelSelect.appendChild(option);
    });
    if (current && models.some((m) => m.key === current)) {
      modelSelect.value = current;
    }
    state.radialModel = modelSelect.value;
  } catch (error) {
    showToast(`Error cargando modelos: ${error.message}`);
  }
}

async function fetchStarNames() {
  try {
    const names = await fetchJSON("/api/star-names");
    state.starNames = names;
    starNamesDataList.innerHTML = names
      .map((name) => `<option value="${name}"></option>`)
      .join("");
  } catch (error) {
    showToast(`Error obteniendo catálogo: ${error.message}`);
  }
}

async function initializeModel() {
  if (!state.imageWidth || !state.imageHeight) return;
  try {
    const payload = {
      imageWidth: state.imageWidth,
      imageHeight: state.imageHeight,
      modelType: state.radialModel,
    };
    const params = await fetchJSON("/api/initial-model", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.modelParams = normalizeModel(params);
    updateModelSummary();
  } catch (error) {
    showToast(`No se pudo inicializar el modelo: ${error.message}`);
  }
}

async function updateExpectedStars() {
  if (!state.modelParams || !state.observation) {
    return;
  }
  try {
    const payload = {
      observation: {
        latitude: state.observation.latitude,
        longitude: state.observation.longitude,
        elevation: state.observation.elevation,
        captureTime: state.observation.captureTime,
      },
      model: {
        modelType: state.modelParams.modelType,
        xc: state.modelParams.xc,
        yc: state.modelParams.yc,
        theta: state.modelParams.theta,
        tiltAngle: state.modelParams.tiltAngle,
        tiltAzimuth: state.modelParams.tiltAzimuth,
        coeffs: state.modelParams.coeffs,
        maxRadiusPx: state.modelParams.maxRadiusPx,
      },
      imageWidth: state.imageWidth,
      imageHeight: state.imageHeight,
      minAltitude: 0,
    };
    const result = await fetchJSON("/api/projected-stars", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.modelParams = normalizeModel(result.model);
    state.expectedStars = result.stars.map((star) => ({ ...star }));
    drawScene();
    updateViewerInfo();
    updateModelSummary();
  } catch (error) {
    showToast(`No se pudo proyectar estrellas: ${error.message}`);
  }
}

async function runDetection() {
  if (!state.imageDataUrl) {
    showToast("Primero carga una imagen");
    return;
  }
  const formData = new FormData(detectionForm);
  const payload = {
    imageData: state.imageDataUrl,
    flipHorizontal: state.flipHorizontal,
    fwhm: parseFloat(formData.get("fwhm")) || 4,
    threshold: parseFloat(formData.get("threshold")) || 5,
    sharplo: parseFloat(formData.get("sharplo")) || 0.2,
    sharphi: parseFloat(formData.get("sharphi")) || 1.0,
    roundlo: parseFloat(formData.get("roundlo")) || -1.0,
    roundhi: parseFloat(formData.get("roundhi")) || 1.0,
    limit: parseInt(formData.get("limit"), 10) || 50,
  };

  try {
    const result = await fetchJSON("/api/detect-stars", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.detections = result.detections.map((det) => ({
      x: det.x,
      y: det.y,
      peak: det.peak,
    }));
    state.selectedDetection = null;
    selectedDetectionInput.value = "";
    starNameInput.value = "";
    confirmMatch.disabled = true;
    drawScene();
    updateViewerInfo();
    showToast(`Detecciones completadas (${state.detections.length})`);
  } catch (error) {
    showToast(`Error en detección: ${error.message}`);
  }
}

function gatherObservationFromForm() {
  const latitude = parseFloat(document.getElementById("latitude").value);
  const longitude = parseFloat(document.getElementById("longitude").value);
  const elevation = parseFloat(document.getElementById("elevation").value || "0");
  const captureTime = document.getElementById("captureTime").value;
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude) || !captureTime) {
    return null;
  }
  return { latitude, longitude, elevation, captureTime };
}

async function fitModel() {
  if (!state.matches.length) {
    return;
  }
  if (!state.observation) {
    showToast("Completa primero los datos de observación");
    return;
  }

  try {
    const payload = {
      observation: {
        latitude: state.observation.latitude,
        longitude: state.observation.longitude,
        elevation: state.observation.elevation,
        captureTime: state.observation.captureTime,
      },
      modelType: state.radialModel,
      imageWidth: state.imageWidth,
      imageHeight: state.imageHeight,
      matches: state.matches.map((match) => ({
        name: match.name,
        x: match.x,
        y: match.y,
      })),
    };
    const result = await fetchJSON("/api/fit-model", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.modelParams = normalizeModel(result.model);
    state.expectedStars = result.projectedStars;
    updateModelSummary();
    drawScene();
    updateViewerInfo();
    showToast("Modelo ajustado con éxito", "success");
  } catch (error) {
    showToast(`No se pudo ajustar el modelo: ${error.message}`);
  }
}

function downloadParameters() {
  if (!state.modelParams || !state.observation) {
    showToast("Faltan datos para descargar parámetros");
    return;
  }
  const payload = {
    radialModel: state.radialModel,
    calibrationModel: state.modelParams,
    observation: state.observation,
    matches: state.matches,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `calibracion_allsky_${Date.now()}.json`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function handleCanvasClick(event) {
  if (!state.detections.length) return;
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = (event.clientX - rect.left) * scaleX;
  const y = (event.clientY - rect.top) * scaleY;

  let closestIndex = null;
  let minDist = Infinity;
  state.detections.forEach((det, index) => {
    const dist = Math.hypot(det.x - x, det.y - y);
    if (dist < minDist) {
      minDist = dist;
      closestIndex = index;
    }
  });

  if (closestIndex !== null && minDist <= 15) {
    state.selectedDetection = closestIndex;
    const det = state.detections[closestIndex];
    selectedDetectionInput.value = `(${det.x.toFixed(1)}, ${det.y.toFixed(1)})`;
    confirmMatch.disabled = false;
    drawScene();
  }
}

function addMatch() {
  if (state.selectedDetection === null) {
    showToast("Selecciona una detección primero");
    return;
  }
  const starName = starNameInput.value.trim();
  if (!starName) {
    showToast("Introduce el nombre de la estrella");
    return;
  }
  if (state.matches.some((match) => match.detectionIndex === state.selectedDetection)) {
    showToast("Esa detección ya está identificada");
    return;
  }
  const normalisedName = starName.toLowerCase();
  if (state.matches.some((match) => match.name.toLowerCase() === normalisedName)) {
    showToast("Esa estrella ya fue asociada");
    return;
  }
  const det = state.detections[state.selectedDetection];
  state.matches.push({
    name: starName,
    x: det.x,
    y: det.y,
    detectionIndex: state.selectedDetection,
  });
  updateMatchesList();
  fitModel();
  starNameInput.value = "";
  confirmMatch.disabled = true;
  selectedDetectionInput.value = "";
  state.selectedDetection = null;
  drawScene();
}

function clearMatches() {
  state.matches = [];
  updateMatchesList();
  updateExpectedStars();
  showToast("Identificaciones reiniciadas");
}

function handleModelChange() {
  state.radialModel = modelSelect.value;
  state.modelParams = null;
  updateModelSummary();
  initializeModel().then(() => updateExpectedStars());
  resetDetections();
  showToast("Modelo radial actualizado. Ajusta de nuevo si es necesario.");
}

function handleOrientationChange() {
  state.flipHorizontal = !isNadirCheckbox.checked;
  if (state.imageElement) {
    drawScene();
  }
  resetDetections();
  updateExpectedStars();
  showToast("Orientación actualizada. Ejecuta de nuevo la detección.");
}

function handleFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (event) => {
    const dataUrl = event.target.result;
    const img = new Image();
    img.onload = () => {
      state.imageElement = img;
      state.imageDataUrl = dataUrl;
      state.imageWidth = img.width;
      state.imageHeight = img.height;
      state.modelParams = null;
      state.expectedStars = [];
      noImageOverlay.style.display = "none";
      drawScene();
      updateViewerInfo();
      updateModelSummary();
      resetDetections();
      initializeModel().then(() => updateExpectedStars());
    };
    img.src = dataUrl;
  };
  reader.readAsDataURL(file);
}

// Event listeners
browseButton.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  handleFile(file);
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragover");
  const file = event.dataTransfer.files[0];
  handleFile(file);
});

canvas.addEventListener("click", handleCanvasClick);

toggleDetections.addEventListener("change", () => {
  state.showDetections = toggleDetections.checked;
  drawScene();
});

toggleExpected.addEventListener("change", () => {
  state.showExpected = toggleExpected.checked;
  drawScene();
});

toggleLabels.addEventListener("change", () => {
  state.showLabels = toggleLabels.checked;
  drawScene();
});

detectionForm.addEventListener("submit", (event) => {
  event.preventDefault();
  runDetection();
});

observationForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const observation = gatherObservationFromForm();
  if (!observation) {
    showToast("Completa latitud, longitud y fecha");
    return;
  }
  state.observation = observation;
  showToast("Datos de observación actualizados");
  updateExpectedStars();
});

confirmMatch.addEventListener("click", addMatch);
clearMatchesButton.addEventListener("click", clearMatches);
resetDetectionsButton.addEventListener("click", resetDetections);
modelSelect.addEventListener("change", handleModelChange);
isNadirCheckbox.addEventListener("change", handleOrientationChange);
downloadButton.addEventListener("click", downloadParameters);

window.addEventListener("load", () => {
  fetchRadialModels();
  fetchStarNames();
  state.flipHorizontal = !isNadirCheckbox.checked;
  clearCanvas();
  updateViewerInfo();
});
