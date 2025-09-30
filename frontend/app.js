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
const toggleIdentified = document.getElementById("toggleIdentified");
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

// Star display controls
const labelSizeSlider = document.getElementById("labelSize");
const labelSizeValue = document.getElementById("labelSizeValue");
const maxStarsInput = document.getElementById("maxStars");
const sortByMagnitudeCheckbox = document.getElementById("sortByMagnitude");

// Parameter editor elements
const updateParametersButton = document.getElementById("updateParameters");
const refitParametersButton = document.getElementById("refitParameters");
const resetParametersButton = document.getElementById("resetParameters");

// Parameter sliders and inputs
const parameterElements = {
  xc: {
    slider: document.getElementById("xcSlider"),
    value: document.getElementById("xcValue"),
    min: document.getElementById("xcMin"),
    max: document.getElementById("xcMax")
  },
  yc: {
    slider: document.getElementById("ycSlider"),
    value: document.getElementById("ycValue"),
    min: document.getElementById("ycMin"),
    max: document.getElementById("ycMax")
  },
  theta: {
    slider: document.getElementById("thetaSlider"),
    value: document.getElementById("thetaValue"),
    min: document.getElementById("thetaMin"),
    max: document.getElementById("thetaMax")
  },
  tiltAngle: {
    slider: document.getElementById("tiltAngleSlider"),
    value: document.getElementById("tiltAngleValue"),
    min: document.getElementById("tiltAngleMin"),
    max: document.getElementById("tiltAngleMax")
  },
  tiltAzimuth: {
    slider: document.getElementById("tiltAzimuthSlider"),
    value: document.getElementById("tiltAzimuthValue"),
    min: document.getElementById("tiltAzimuthMin"),
    max: document.getElementById("tiltAzimuthMax")
  },
  coeffA: {
    slider: document.getElementById("coeffASlider"),
    value: document.getElementById("coeffAValue"),
    min: document.getElementById("coeffAMin"),
    max: document.getElementById("coeffAMax")
  },
  coeffB: {
    slider: document.getElementById("coeffBSlider"),
    value: document.getElementById("coeffBValue"),
    min: document.getElementById("coeffBMin"),
    max: document.getElementById("coeffBMax")
  },
  coeffC: {
    slider: document.getElementById("coeffCSlider"),
    value: document.getElementById("coeffCValue"),
    min: document.getElementById("coeffCMin"),
    max: document.getElementById("coeffCMax")
  }
};

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
  showIdentified: true,
  radialModel: "poly2",
  radialModels: [],
  labelSize: 14,
  maxStars: 100,
  sortByMagnitude: true,
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

function normaliseStarName(name) {
  if (!name) return "";
  return name
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .trim()
    .toLowerCase();
}

function syncMatchesWithExpected() {
  if (!state.matches.length || !state.expectedStars.length) {
    return;
  }
  const expectedMap = new Map(
    state.expectedStars.map((star) => [normaliseStarName(star.name), star])
  );
  state.matches.forEach((match) => {
    const expected = expectedMap.get(normaliseStarName(match.name));
    if (!expected) {
      return;
    }
    match.x_expected = expected.x;
    match.y_expected = expected.y;
    match.alt = expected.alt;
    match.az = expected.az;
    if (Object.prototype.hasOwnProperty.call(expected, "ra")) {
      match.RA = expected.ra;
    }
    if (Object.prototype.hasOwnProperty.call(expected, "dec")) {
      match.DEC = expected.dec;
    }
  });
}

function normalizeModel(apiModel) {
  return {
    modelType: apiModel.modelType || apiModel.model_type || 'poly2',
    xc: apiModel.xc ?? 1520,
    yc: apiModel.yc ?? 1520,
    theta: apiModel.theta ?? apiModel.theta_deg ?? 0.0,
    tiltAngle: apiModel.tiltAngle ?? apiModel.tilt_angle ?? 0.0,
    tiltAzimuth: apiModel.tiltAzimuth ?? apiModel.tilt_azimuth ?? 0.0,
    coeffs: apiModel.coeffs || [0.2, 0.8],
    maxRadiusPx: apiModel.maxRadiusPx ?? apiModel.max_radius_px ?? 2149.6,
    normFactor: apiModel.normFactor ?? apiModel.norm_factor ?? 1520,
    R: apiModel.R ?? 1520,
  };
}

function updateParameterEditor() {
  if (!state.modelParams) {
    downloadButton.disabled = true;
    return;
  }

  const mp = state.modelParams;
  
  // Update slider values and displays
  updateSliderValue('xc', mp.xc, 0);
  updateSliderValue('yc', mp.yc, 0);
  updateSliderValue('theta', mp.theta, 1);
  updateSliderValue('tiltAngle', mp.tiltAngle, 1);
  updateSliderValue('tiltAzimuth', mp.tiltAzimuth, 1);
  updateSliderValue('coeffA', mp.coeffs[0] || 0.2, 3);
  updateSliderValue('coeffB', mp.coeffs[1] || 0.8, 3);
  
  // Handle coefficient C visibility based on model type
  const coeffCRow = document.getElementById('coeffCRow');
  const selectedModel = state.radialModels.find(m => m.key === state.radialModel);
  const hasThreeCoeffs = selectedModel && selectedModel.coefficients >= 3;
  
  coeffCRow.style.display = hasThreeCoeffs ? 'block' : 'none';
  if (hasThreeCoeffs) {
    updateSliderValue('coeffC', mp.coeffs[2] || 1.0, 3);
  }
  
  downloadButton.disabled = false;
}

function updateSliderValue(paramName, value, decimals) {
  const elements = parameterElements[paramName];
  if (!elements) return;
  
  elements.slider.value = value;
  elements.value.textContent = value.toFixed(decimals);
  
  // Update slider bounds if limit inputs have values
  const minVal = parseFloat(elements.min.value);
  const maxVal = parseFloat(elements.max.value);
  
  if (!isNaN(minVal)) {
    elements.slider.min = minVal;
  }
  if (!isNaN(maxVal)) {
    elements.slider.max = maxVal;
  }
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
        ctx.font = `${state.labelSize}px 'Inter', sans-serif`;
        ctx.fillText(star.name, star.x + 12, star.y + 4);
      }
    }
  }

  if (state.showDetections) {
    const matchedDetections = new Set(
      state.matches.map((match) => match.detectionIndex)
    );
    for (let i = 0; i < state.detections.length; i++) {
      const det = state.detections[i];
      const isMatched = matchedDetections.has(i);
      if (isMatched && !state.showIdentified) {
        continue;
      }
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.strokeStyle = isMatched
        ? "rgba(135, 206, 250, 0.9)"
        : "rgba(255, 255, 255, 0.85)";
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
    state.radialModels = models;
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
    updateParameterEditor();
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
        theta: state.modelParams.theta, // a.k.a. theta_deg
        tiltAngle: state.modelParams.tiltAngle, // a.k.a. tilt_angle
        tiltAzimuth: state.modelParams.tiltAzimuth, // a.k.a. tilt_azimuth
        coeffs: state.modelParams.coeffs,
        maxRadiusPx: state.modelParams.maxRadiusPx, // a.k.a. max_radius_px
        normFactor: state.modelParams.normFactor, // a.k.a. norm_factor
        R: state.modelParams.R,
      },
      imageWidth: state.imageWidth,
      imageHeight: state.imageHeight,
      minAltitude: 0,
      maxStars: state.maxStars,
      sortByMagnitude: state.sortByMagnitude,
    };
    const result = await fetchJSON("/api/projected-stars", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.modelParams = normalizeModel(result.model);
    state.expectedStars = result.stars.map((star) => ({ ...star }));
    
    // Solo mostrar las 5 estrellas específicas del script
    const targetStars = ['deneb', 'altair', 'vega', 'antares', 'mizar'];
    const filteredStars = state.expectedStars.filter(star => 
      targetStars.some(target => star.name.toLowerCase().includes(target.toLowerCase()))
    );
    
    if (filteredStars.length) {
      console.groupCollapsed(`Estrellas principales (${filteredStars.length})`);
      filteredStars.forEach((star) => {
        console.log(
          `🌟 ${star.name}: x=${star.x.toFixed(2)}, y=${star.y.toFixed(2)}, alt=${star.alt.toFixed(1)}°, az=${star.az.toFixed(1)}°`
        );
      });
      console.groupEnd();
    } else {
      console.info("No se encontraron las estrellas principales en la proyección.");
    }
    syncMatchesWithExpected();
    drawScene();
    updateViewerInfo();
    updateParameterEditor();
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
  const latitudeField = document.getElementById("latitude");
  const longitudeField = document.getElementById("longitude");
  const elevationField = document.getElementById("elevation");
  const latitude = parseFloat(latitudeField.value);
  const longitude = parseFloat(longitudeField.value);
  const elevation = parseFloat(elevationField.value || "0");
  const captureTime = document.getElementById("captureTime").value;
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude) || !captureTime) {
    return null;
  }
  const roundedLatitude = Number(latitude.toFixed(6));
  const roundedLongitude = Number(longitude.toFixed(6));
  latitudeField.value = roundedLatitude.toFixed(6);
  longitudeField.value = roundedLongitude.toFixed(6);
  return { latitude: roundedLatitude, longitude: roundedLongitude, elevation, captureTime };
}

async function fitModel() {
  const selectedModel = state.radialModels.find(m => m.key === state.radialModel);
  const nCoeffs = selectedModel ? selectedModel.coefficients : 2; // Default to 2 if not found
  const requiredStars = nCoeffs + 2;

  if (state.matches.length < requiredStars) {
    const remaining = requiredStars - state.matches.length;
    showToast(`Añade ${remaining} estrella${remaining > 1 ? 's' : ''} más para poder ajustar el modelo`);
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
    state.expectedStars = result.projectedStars.map((star) => ({ ...star }));
    
    // Solo mostrar las 5 estrellas específicas del script
    const targetStars = ['deneb', 'altair', 'vega', 'antares', 'mizar'];
    const filteredStars = state.expectedStars.filter(star => 
      targetStars.some(target => star.name.toLowerCase().includes(target.toLowerCase()))
    );
    
    if (filteredStars.length) {
      console.groupCollapsed(`Estrellas principales (${filteredStars.length})`);
      filteredStars.forEach((star) => {
        console.log(
          `🌟 ${star.name}: x=${star.x.toFixed(2)}, y=${star.y.toFixed(2)}, alt=${star.alt.toFixed(1)}°, az=${star.az.toFixed(1)}°`
        );
      });
      console.groupEnd();
    } else {
      console.info("No se encontraron las estrellas principales en la proyección.");
    }
    syncMatchesWithExpected();
    updateParameterEditor();
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
  const normalisedName = normaliseStarName(starName);
  if (state.matches.some((match) => normaliseStarName(match.name) === normalisedName)) {
    showToast("Esa estrella ya fue asociada");
    return;
  }
  const det = state.detections[state.selectedDetection];
  const expected = state.expectedStars.find(
    (star) => normaliseStarName(star.name) === normalisedName
  );
  if (!expected) {
    console.warn(
      `No se encontró posición esperada para la estrella identificada: ${starName}`
    );
  }
  state.matches.push({
    name: starName,
    x: det.x,
    y: det.y,
    detectionIndex: state.selectedDetection,
    x_expected: expected ? expected.x : null,
    y_expected: expected ? expected.y : null,
    alt: expected ? expected.alt : null,
    az: expected ? expected.az : null,
    RA:
      expected && Object.prototype.hasOwnProperty.call(expected, "ra")
        ? expected.ra
        : null,
    DEC:
      expected && Object.prototype.hasOwnProperty.call(expected, "dec")
        ? expected.dec
        : null,
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
  drawScene();
  updateExpectedStars();
  showToast("Identificaciones reiniciadas");
}

function handleModelChange() {
  state.radialModel = modelSelect.value;
  state.modelParams = null;
  updateParameterEditor();
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

function setupStarDisplayEventListeners() {
  // Label size slider
  labelSizeSlider.addEventListener('input', (e) => {
    state.labelSize = parseInt(e.target.value);
    labelSizeValue.textContent = `${state.labelSize}px`;
    drawScene();
  });
  
  // Max stars input
  maxStarsInput.addEventListener('change', (e) => {
    state.maxStars = parseInt(e.target.value) || 100;
    updateExpectedStars();
  });
  
  // Sort by magnitude checkbox
  sortByMagnitudeCheckbox.addEventListener('change', (e) => {
    state.sortByMagnitude = e.target.checked;
    updateExpectedStars();
  });
}

function setupParameterEventListeners() {
  // Add event listeners for all parameter sliders
  Object.entries(parameterElements).forEach(([paramName, elements]) => {
    // Slider input event
    elements.slider.addEventListener('input', (e) => {
      const value = parseFloat(e.target.value);
      const decimals = paramName.startsWith('coeff') ? 3 : (paramName === 'xc' || paramName === 'yc' ? 0 : 1);
      elements.value.textContent = value.toFixed(decimals);
    });
    
    // Limit input change events
    elements.min.addEventListener('change', (e) => {
      const minVal = parseFloat(e.target.value);
      if (!isNaN(minVal)) {
        elements.slider.min = minVal;
        if (parseFloat(elements.slider.value) < minVal) {
          elements.slider.value = minVal;
          const decimals = paramName.startsWith('coeff') ? 3 : (paramName === 'xc' || paramName === 'yc' ? 0 : 1);
          elements.value.textContent = minVal.toFixed(decimals);
        }
      }
    });
    
    elements.max.addEventListener('change', (e) => {
      const maxVal = parseFloat(e.target.value);
      if (!isNaN(maxVal)) {
        elements.slider.max = maxVal;
        if (parseFloat(elements.slider.value) > maxVal) {
          elements.slider.value = maxVal;
          const decimals = paramName.startsWith('coeff') ? 3 : (paramName === 'xc' || paramName === 'yc' ? 0 : 1);
          elements.value.textContent = maxVal.toFixed(decimals);
        }
      }
    });
  });
}

function getParametersFromSliders() {
  const selectedModel = state.radialModels.find(m => m.key === state.radialModel);
  const hasThreeCoeffs = selectedModel && selectedModel.coefficients >= 3;
  
  const coeffs = [
    parseFloat(parameterElements.coeffA.slider.value),
    parseFloat(parameterElements.coeffB.slider.value)
  ];
  
  if (hasThreeCoeffs) {
    coeffs.push(parseFloat(parameterElements.coeffC.slider.value));
  }
  
  return {
    modelType: state.radialModel,
    xc: parseFloat(parameterElements.xc.slider.value),
    yc: parseFloat(parameterElements.yc.slider.value),
    theta: parseFloat(parameterElements.theta.slider.value),
    tiltAngle: parseFloat(parameterElements.tiltAngle.slider.value),
    tiltAzimuth: parseFloat(parameterElements.tiltAzimuth.slider.value),
    coeffs: coeffs,
    maxRadiusPx: state.modelParams?.maxRadiusPx ?? 2149.6,
    normFactor: state.modelParams?.normFactor ?? 1520,
    R: state.modelParams?.R ?? 1520
  };
}

async function updateParametersFromSliders() {
  if (!state.modelParams || !state.observation) {
    showToast("Completa primero los datos de observación");
    return;
  }
  
  try {
    // Update model parameters from slider values
    state.modelParams = getParametersFromSliders();
    
    // Recalculate expected star positions
    await updateExpectedStars();
    
    showToast("Parámetros actualizados", "success");
  } catch (error) {
    showToast(`Error actualizando parámetros: ${error.message}`);
  }
}

function resetParametersToDefaults() {
  if (!state.modelParams) {
    showToast("No hay parámetros para restaurar");
    return;
  }
  
  // Reset to initial model values
  initializeModel().then(() => {
    updateExpectedStars();
    showToast("Parámetros restaurados", "success");
  });
}

function getParameterBounds() {
  const bounds = {};
  
  Object.entries(parameterElements).forEach(([paramName, elements]) => {
    const minVal = parseFloat(elements.min.value);
    const maxVal = parseFloat(elements.max.value);
    
    if (!isNaN(minVal) && !isNaN(maxVal)) {
      bounds[paramName] = [minVal, maxVal];
    }
  });
  
  return bounds;
}

async function refitWithCustomBounds() {
  if (!state.matches || state.matches.length === 0) {
    showToast("Necesitas identificar al menos una estrella para reajustar");
    return;
  }
  
  if (!state.observation) {
    showToast("Completa primero los datos de observación");
    return;
  }
  
  try {
    const bounds = getParameterBounds();
    
    const payload = {
      observation: state.observation,
      matches: state.matches.map(match => ({
        name: match.name,
        x: match.x,
        y: match.y
      })),
      modelType: state.radialModel,
      imageWidth: state.imageWidth,
      imageHeight: state.imageHeight,
      customBounds: bounds
    };
    
    const result = await fetchJSON("/api/fit-model", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    
    state.modelParams = normalizeModel(result.model);
    updateParameterEditor();
    await updateExpectedStars();
    
    showToast("Modelo reajustado con límites personalizados", "success");
  } catch (error) {
    showToast(`Error en el reajuste: ${error.message}`);
  }
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

      dropZone.classList.add("has-image");

      noImageOverlay.style.display = "none";
      drawScene();
      updateViewerInfo();
      updateParameterEditor();
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

toggleIdentified.addEventListener("change", () => {
  state.showIdentified = toggleIdentified.checked;
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

// Parameter editor event listeners
updateParametersButton.addEventListener("click", updateParametersFromSliders);
refitParametersButton.addEventListener("click", refitWithCustomBounds);
resetParametersButton.addEventListener("click", resetParametersToDefaults);

window.addEventListener("load", () => {
  fetchRadialModels();
  fetchStarNames();
  setupParameterEventListeners();
  setupStarDisplayEventListeners();
  state.flipHorizontal = !isNadirCheckbox.checked;
  state.showIdentified = toggleIdentified.checked;
  clearCanvas();
  updateViewerInfo();
});
