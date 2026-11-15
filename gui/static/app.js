/**
 * app.js - AIEcoLightswitcher Modern Multi-Feed Dashboard Logic
 * Adheres to Modern Web Guidance:
 * - Dynamic minimum 4-card grid layout with auto-fit and placeholder management
 * - Fine-grained DOM mutation without flickering or frame stutter
 * - Responsive container queries and accessible form interactions
 */

document.addEventListener("DOMContentLoaded", () => {
  const feedGrid = document.getElementById("feed-grid-container");
  const statTotalFeeds = document.getElementById("stat-total-feeds");
  const statOnlineFeeds = document.getElementById("stat-online-feeds");
  const statActiveRelays = document.getElementById("stat-active-relays");
  const statDutyCycle = document.getElementById("stat-duty-cycle");
  
  const txtDevice = document.getElementById("txt-device");
  const txtHardware = document.getElementById("txt-hardware");
  const txtFps = document.getElementById("txt-fps");
  
  const addModal = document.getElementById("add-feed-dialog");
  const btnOpenAddModal = document.getElementById("btn-open-add-modal");
  const btnCloseAddModal = document.getElementById("btn-close-add-modal");
  const btnCancelAddModal = document.getElementById("btn-cancel-add-modal");
  const formAddFeed = document.getElementById("form-add-feed");
  
  const feedTypeSelect = document.getElementById("feed-type");
  const droidcamOptions = document.getElementById("droidcam-options");
  const usbOptions = document.getElementById("usb-options");
  const rtspOptions = document.getElementById("rtsp-options");
  
  const hwModal = document.getElementById("hardware-dialog");
  const btnOpenHwModal = document.getElementById("btn-hardware-modal");
  const btnCloseHwModal = document.getElementById("btn-close-hw-modal");
  const formHardware = document.getElementById("form-hardware");
  const btnHwResync = document.getElementById("btn-hw-resync");
  const hwPortSelect = document.getElementById("hw-port");

  let currentTelemetry = {};
  let renderedFeedIds = new Set();
  let nextFeedIndex = 1;

  // ========================================================================
  // Modal Handlers
  // ========================================================================
  btnOpenAddModal.addEventListener("click", () => {
    document.getElementById("feed-name").value = `Camera Feed ${nextFeedIndex}`;
    addModal.showModal();
  });

  btnCloseAddModal.addEventListener("click", () => addModal.close());
  btnCancelAddModal.addEventListener("click", () => addModal.close());

  feedTypeSelect.addEventListener("change", () => {
    const type = feedTypeSelect.value;
    droidcamOptions.style.display = type === "DROIDCAM" ? "block" : "none";
    usbOptions.style.display = type === "USB" ? "block" : "none";
    rtspOptions.style.display = type === "RTSP" ? "block" : "none";
  });

  btnOpenHwModal.addEventListener("click", async () => {
    try {
      const res = await fetch("/api/hardware/ports");
      const data = await res.json();
      hwPortSelect.innerHTML = '<option value="SIMULATION">Virtual Simulation (No Hardware Required)</option>';
      if (data.ports && data.ports.length > 0) {
        data.ports.forEach(p => {
          const opt = document.createElement("option");
          opt.value = p.device;
          opt.textContent = `${p.device} — ${p.description || "Serial Device"}`;
          hwPortSelect.appendChild(opt);
        });
      }
    } catch (err) {
      console.warn("Error fetching serial ports:", err);
    }
    hwModal.showModal();
  });

  btnCloseHwModal.addEventListener("click", () => hwModal.close());

  formHardware.addEventListener("submit", async (e) => {
    e.preventDefault();
    const port = hwPortSelect.value;
    const baud = parseInt(document.getElementById("hw-baud").value, 10);
    try {
      await fetch("/api/hardware/connect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ port, baud })
      });
      hwModal.close();
      pollTelemetry();
    } catch (err) {
      alert("Failed to connect hardware: " + err);
    }
  });

  btnHwResync.addEventListener("click", async () => {
    try {
      await fetch("/api/hardware/force_sync", { method: "POST" });
      alert("Resynchronized all Arduino physical GPIO pin states!");
    } catch (err) {
      alert("Error resyncing pins: " + err);
    }
  });

  formAddFeed.addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = document.getElementById("feed-name").value.trim();
    const type = feedTypeSelect.value;
    const pin = parseInt(document.getElementById("feed-pin").value, 10);
    const feed_id = `feed_${Date.now()}`;

    let source = 0;
    if (type === "DROIDCAM") {
      const ip = document.getElementById("droidcam-ip").value.trim();
      const port = document.getElementById("droidcam-port").value.trim();
      const path = document.getElementById("droidcam-path").value.trim();
      source = `http://${ip}:${port}${path.startsWith('/') ? path : '/' + path}`;
    } else if (type === "USB") {
      source = parseInt(document.getElementById("usb-index").value, 10);
    } else if (type === "RTSP") {
      source = document.getElementById("rtsp-url").value.trim();
    } else if (type === "MOCK_OCCUPIED") {
      source = "MOCK:PERSON_IN_ROI";
    } else if (type === "MOCK_EMPTY") {
      source = "MOCK:EMPTY_FRAME";
    }

    try {
      const res = await fetch("/api/feed/add", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ feed_id, name, source, pin })
      });
      const data = await res.json();
      if (data.status === "ok") {
        nextFeedIndex++;
        addModal.close();
        pollTelemetry();
      } else {
        alert("Could not add feed: " + (data.message || "Unknown error"));
      }
    } catch (err) {
      alert("Error adding feed: " + err);
    }
  });

  // ========================================================================
  // Grid Card DOM Generation & Updates
  // ========================================================================
  function createFeedCard(fid, telem) {
    const card = document.createElement("article");
    card.className = "feed-card";
    card.id = `card-${fid}`;
    card.innerHTML = `
      <header class="card-header">
        <div class="card-title-group">
          <span class="card-title" id="title-${fid}">${telem.name || fid}</span>
          <span class="pin-badge" id="pin-badge-${fid}">PIN ${telem.pin}</span>
        </div>
        <button class="btn-danger btn-sm" onclick="removeFeed('${fid}')" title="Remove Feed">✕</button>
      </header>

      <div class="card-media">
        <img class="feed-video-stream" src="/api/feed/${fid}/stream" alt="Live Feed ${fid}" loading="lazy">
        <div class="feed-hud-overlay">
          <div class="hud-top-row">
            <span class="hud-chip" id="hud-status-${fid}">
              <span class="pulse-dot"></span>
              <span id="hud-status-txt-${fid}">${telem.status || "CONNECTING"}</span>
            </span>
            <span class="hud-chip" id="hud-fps-${fid}">${telem.fps || 0} FPS</span>
          </div>
          <div class="hud-bottom-row">
            <span class="hud-chip" id="hud-cycle-${fid}">${telem.cycle_state || "EVALUATING"}</span>
            <span class="hud-chip" id="hud-latency-${fid}">${telem.latency_ms || 0} ms</span>
          </div>
        </div>
      </div>

      <div class="card-content">
        <div class="metric-item">
          <span class="metric-label">Detected</span>
          <span class="metric-value" id="val-occ-${fid}">${telem.roi_occupants || 0} Humans</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Smoothed</span>
          <span class="metric-value" id="val-smooth-${fid}">${telem.smoothed_presence ? "OCCUPIED" : "VACANT"}</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Light Relay</span>
          <span class="metric-value" id="val-light-${fid}">${telem.pin_state ? "HIGH (ON)" : "LOW (OFF)"}</span>
        </div>
      </div>

      <footer class="card-footer">
        <label class="switch-container" title="Enable / Disable Stream">
          <input type="checkbox" class="switch-input" id="switch-${fid}" ${telem.enabled ? "checked" : ""} onchange="toggleFeed('${fid}', this.checked)">
          <span>Active</span>
        </label>

        <div class="override-group">
          <button class="override-btn ${telem.override_mode === 'AUTO' ? 'active' : ''}" id="ov-auto-${fid}" onclick="setOverride('${fid}', 'AUTO')">AUTO</button>
          <button class="override-btn ${telem.override_mode === 'FORCE_ON' ? 'active' : ''}" id="ov-on-${fid}" onclick="setOverride('${fid}', 'FORCE_ON')">FORCE ON</button>
          <button class="override-btn ${telem.override_mode === 'FORCE_OFF' ? 'active-force-off' : ''}" id="ov-off-${fid}" onclick="setOverride('${fid}', 'FORCE_OFF')">FORCE OFF</button>
        </div>
      </footer>
    `;
    return card;
  }

  function updateFeedCard(fid, telem) {
    const card = document.getElementById(`card-${fid}`);
    if (!card) return;

    // Occupied / Sleeping / Offline styles
    const isOccupied = !!telem.smoothed_presence;
    const isSleeping = telem.cycle_state === "SLEEPING";
    const isOnline = telem.status === "ONLINE";

    card.classList.toggle("active-occupied", isOccupied && isOnline);
    card.classList.toggle("active-sleeping", isSleeping && isOnline);
    card.classList.toggle("feed-offline", !isOnline);

    // Pin badge
    const pinBadge = document.getElementById(`pin-badge-${fid}`);
    if (pinBadge) {
      pinBadge.textContent = `PIN ${telem.pin}`;
      pinBadge.classList.toggle("pin-high", !!telem.pin_state);
    }

    // Status text & chip color
    const hudStatusTxt = document.getElementById(`hud-status-txt-${fid}`);
    if (hudStatusTxt) hudStatusTxt.textContent = telem.status || "OFFLINE";

    // FPS & Latency
    const hudFps = document.getElementById(`hud-fps-${fid}`);
    if (hudFps) hudFps.textContent = `${telem.fps || 0} FPS`;

    const hudLatency = document.getElementById(`hud-latency-${fid}`);
    if (hudLatency) hudLatency.textContent = `${telem.latency_ms || 0} ms`;

    // Cycle
    const hudCycle = document.getElementById(`hud-cycle-${fid}`);
    if (hudCycle) {
      if (telem.cycle_state === "SLEEPING" && telem.cycle_countdown) {
        hudCycle.textContent = `SLEEP (${telem.cycle_countdown}s)`;
      } else {
        hudCycle.textContent = telem.cycle_state || "EVALUATING";
      }
    }

    // Content metrics
    const valOcc = document.getElementById(`val-occ-${fid}`);
    if (valOcc) valOcc.textContent = `${telem.roi_occupants || 0} Humans`;

    const valSmooth = document.getElementById(`val-smooth-${fid}`);
    if (valSmooth) {
      valSmooth.textContent = telem.smoothed_presence ? "OCCUPIED" : "VACANT";
      valSmooth.style.color = telem.smoothed_presence ? "var(--status-mint)" : "var(--text-primary)";
    }

    const valLight = document.getElementById(`val-light-${fid}`);
    if (valLight) {
      valLight.textContent = telem.pin_state ? "HIGH (ON)" : "LOW (OFF)";
      valLight.style.color = telem.pin_state ? "var(--status-mint)" : "var(--status-offline)";
    }

    // Switch
    const sw = document.getElementById(`switch-${fid}`);
    if (sw && sw !== document.activeElement) sw.checked = !!telem.enabled;

    // Overrides
    const mode = telem.override_mode || "AUTO";
    const ovAuto = document.getElementById(`ov-auto-${fid}`);
    const ovOn = document.getElementById(`ov-on-${fid}`);
    const ovOff = document.getElementById(`ov-off-${fid}`);
    if (ovAuto) ovAuto.className = `override-btn ${mode === "AUTO" ? "active" : ""}`;
    if (ovOn) ovOn.className = `override-btn ${mode === "FORCE_ON" ? "active" : ""}`;
    if (ovOff) ovOff.className = `override-btn ${mode === "FORCE_OFF" ? "active-force-off" : ""}`;
  }

  function createPlaceholderCard(index) {
    const el = document.createElement("div");
    el.className = "placeholder-card";
    el.id = `placeholder-slot-${index}`;
    el.innerHTML = `
      <div class="placeholder-icon">📷</div>
      <div class="placeholder-title">STANDBY SLOT #${index}</div>
      <div class="placeholder-desc">Channel unassigned. Connect Wi-Fi DroidCam, USB or RTSP stream.</div>
      <button class="btn-sm" onclick="document.getElementById('btn-open-add-modal').click()">➕ Add Camera to Slot</button>
    `;
    return el;
  }

  function updateGrid(telemetry) {
    const fids = Object.keys(telemetry);
    const totalFeeds = fids.length;

    // 1. Remove cards that were deleted
    renderedFeedIds.forEach(fid => {
      if (!telemetry[fid]) {
        const el = document.getElementById(`card-${fid}`);
        if (el) el.remove();
        renderedFeedIds.delete(fid);
      }
    });

    // 2. Add or update active feed cards
    fids.forEach(fid => {
      if (!renderedFeedIds.has(fid)) {
        const card = createFeedCard(fid, telemetry[fid]);
        feedGrid.appendChild(card);
        renderedFeedIds.add(fid);
      } else {
        updateFeedCard(fid, telemetry[fid]);
      }
    });

    // 3. Remove old placeholders
    document.querySelectorAll(".placeholder-card").forEach(el => el.remove());

    // 4. Ensure minimum 4 cards displayed at all times
    const placeholdersNeeded = Math.max(0, 4 - totalFeeds);
    for (let i = 1; i <= placeholdersNeeded; i++) {
      const ph = createPlaceholderCard(totalFeeds + i);
      feedGrid.appendChild(ph);
    }

    // 5. Update Global Stats
    statTotalFeeds.textContent = `${totalFeeds} Active (${Math.max(4, totalFeeds)} Grid Slots)`;
    
    const onlineCount = fids.filter(f => telemetry[f].status === "ONLINE").length;
    statOnlineFeeds.textContent = `${onlineCount} / ${totalFeeds} Online`;

    const activeRelays = fids.filter(f => !!telemetry[f].pin_state).length;
    statActiveRelays.textContent = `${activeRelays} Relays HIGH`;

    const sleepingCount = fids.filter(f => telemetry[f].cycle_state === "SLEEPING").length;
    statDutyCycle.textContent = sleepingCount > 0 ? `${sleepingCount} Feeds Sleeping` : "Active Evaluating";
  }

  // ========================================================================
  // Global Action Functions
  // ========================================================================
  window.removeFeed = async (feed_id) => {
    if (!confirm(`Are you sure you want to remove feed '${feed_id}'?`)) return;
    try {
      await fetch("/api/feed/remove", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ feed_id })
      });
      pollTelemetry();
    } catch (err) {
      alert("Error removing feed: " + err);
    }
  };

  window.toggleFeed = async (feed_id, enabled) => {
    try {
      await fetch("/api/feed/toggle", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ feed_id, enabled })
      });
      pollTelemetry();
    } catch (err) {
      alert("Error toggling feed: " + err);
    }
  };

  window.setOverride = async (feed_id, mode) => {
    try {
      await fetch("/api/feed/override", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ feed_id, mode })
      });
      pollTelemetry();
    } catch (err) {
      alert("Error setting override: " + err);
    }
  };

  // ========================================================================
  // Polling Loop
  // ========================================================================
  async function pollTelemetry() {
    try {
      const res = await fetch("/api/telemetry");
      const data = await res.json();
      currentTelemetry = data.feeds || {};
      
      // Update global header badges
      if (data.device) {
        const devName = Array.isArray(data.device) ? data.device[0] : String(data.device);
        txtDevice.textContent = `DEVICE: ${devName.toUpperCase()}`;
      }
      if (data.hardware) {
        const hwText = data.hardware.is_simulation 
          ? "ARDUINO: VIRTUAL SIM" 
          : `ARDUINO: CONNECTED (${data.hardware.port})`;
        txtHardware.textContent = hwText;
        const hwBadge = document.getElementById("badge-hardware");
        if (hwBadge) {
          hwBadge.className = data.hardware.is_connected ? "badge badge-mint" : "badge badge-amber";
        }
      }
      if (data.global_fps !== undefined) {
        txtFps.textContent = parseFloat(data.global_fps).toFixed(1);
      }

      updateGrid(currentTelemetry);
    } catch (err) {
      console.warn("Telemetry poll error:", err);
    }
  }

  // Initial poll & 300ms cadence
  pollTelemetry();
  setInterval(pollTelemetry, 300);
});
