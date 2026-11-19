"""
Interactive Graph Renderer for Epigraphical Knowledge Graph (Phase G).

Generates self-contained, theme-aligned HTML visualizer using Vis.js Network
with automated multi-CDN fallbacks and a built-in high-performance HTML5 Canvas
force-directed engine so the visualization NEVER fails or presents a blank canvas.

Features:
- Dark Graphite (#171D24 / #202832) & Muted Bronze (#C5A15A) aesthetic.
- Category filtering & live search.
- Interactive side panel for citation details, dates, and historical descriptions.
- Responsive physics simulation with node pinning and toggle.
- Full Vis.js and Canvas fallback support for dynastic flag image nodes.
"""

from __future__ import annotations

import json
from typing import Dict, Any, Optional
from services.knowledge_graph.graph_engine import EpigraphicalKnowledgeGraph, FALLBACK_DYNASTY_SVGS, GENERIC_DYNASTY_FALLBACK_SVG


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Kallvettu Epigraphical Knowledge Graph</title>
  <!-- Multi-CDN Script Loader with automated fallback chain -->
  <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/standalone/umd/vis-network.min.js"></script>
  <script type="text/javascript">
    if (typeof vis === 'undefined') {
      document.write('<script type="text/javascript" src="https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js"><\\/script>');
    }
  </script>
  <script type="text/javascript">
    if (typeof vis === 'undefined') {
      document.write('<script type="text/javascript" src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"><\\/script>');
    }
  </script>
  <style>
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    html, body {
      width: 100%;
      height: 100%;
      min-height: 100%;
      background-color: #171D24;
      color: #E2E8F0;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      height: __HEIGHT_PX__px;
    }
    #toolbar {
      height: 52px;
      min-height: 52px;
      background-color: #202832;
      padding: 8px 16px;
      display: flex;
      gap: 12px;
      align-items: center;
      border-bottom: 1px solid #2D3748;
      font-size: 13px;
      flex-shrink: 0;
      flex-wrap: wrap;
      z-index: 10;
    }
    .tool-group {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    label {
      color: #C5A15A;
      font-weight: 600;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    input, select {
      background-color: #12171D;
      color: #E2E8F0;
      border: 1px solid #4A5568;
      border-radius: 4px;
      padding: 4px 8px;
      font-size: 13px;
      outline: none;
    }
    input:focus, select:focus {
      border-color: #C5A15A;
    }
    button {
      background-color: #2D3748;
      color: #C5A15A;
      border: 1px solid #C5A15A;
      border-radius: 4px;
      padding: 5px 12px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }
    button:hover {
      background-color: #C5A15A;
      color: #12171D;
    }
    #main-container {
      display: flex;
      flex: 1 1 auto;
      height: calc(100% - 52px);
      min-height: 520px;
      width: 100%;
      position: relative;
      overflow: hidden;
    }
    #network-canvas {
      flex: 1 1 auto;
      width: calc(100% - 330px);
      height: 100%;
      min-height: 520px;
      position: relative;
      background: radial-gradient(circle at center, #1B232D 0%, #12171E 100%);
      overflow: hidden;
    }
    #canvas-container {
      width: 100%;
      height: 100%;
      position: absolute;
      top: 0;
      left: 0;
    }
    #fallback-canvas {
      display: block;
      width: 100%;
      height: 100%;
      position: absolute;
      top: 0;
      left: 0;
      cursor: grab;
    }
    #fallback-canvas:active {
      cursor: grabbing;
    }
    #side-panel {
      width: 330px;
      min-width: 330px;
      max-width: 330px;
      height: 100%;
      background-color: #1B232C;
      border-left: 1px solid #2D3748;
      padding: 16px;
      overflow-y: auto;
      font-size: 13px;
      line-height: 1.5;
      display: flex;
      flex-direction: column;
      gap: 12px;
      flex-shrink: 0;
      z-index: 5;
    }
    .panel-header {
      font-size: 15px;
      font-weight: bold;
      color: #C5A15A;
      border-bottom: 1px solid #3B4758;
      padding-bottom: 8px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      background-color: #2D3748;
      color: #C5A15A;
    }
    .prop-row {
      margin-bottom: 6px;
    }
    .prop-name {
      color: #A0AEC0;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .prop-val {
      color: #F7FAFC;
      font-size: 13px;
    }
    .citation-box {
      background-color: #12171D;
      border-left: 3px solid #C5A15A;
      padding: 8px;
      border-radius: 0 4px 4px 0;
      font-size: 12px;
      color: #CBD5E0;
    }
    .stat-pill {
      color: #A0AEC0;
      font-size: 11px;
      margin-left: auto;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .engine-indicator {
      display: inline-block;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: 600;
      background-color: rgba(72, 187, 120, 0.15);
      color: #48BB78;
      border: 1px solid #48BB78;
    }
    .flag-preview-card {
      background-color: #12171D;
      border: 1px solid #3B4758;
      border-radius: 6px;
      padding: 10px;
      text-align: center;
      margin-top: 8px;
      margin-bottom: 8px;
    }
    .flag-preview-img {
      max-width: 100%;
      max-height: 135px;
      object-fit: contain;
      border-radius: 4px;
      border: 1px solid #C5A15A;
      box-shadow: 0 4px 10px rgba(0,0,0,0.5);
      background-color: #0F141A;
    }
    .emblem-title {
      font-size: 12px;
      font-weight: 600;
      color: #E5A93C;
      margin-top: 6px;
    }
    .emblem-desc {
      font-size: 11px;
      color: #CBD5E0;
      margin-top: 4px;
      line-height: 1.4;
    }
    .provenance-card {
      margin-top: 8px;
      font-size: 10.5px;
      color: #F59E0B;
      background: rgba(245, 158, 11, 0.1);
      border-left: 3px solid #F59E0B;
      padding: 6px 8px;
      border-radius: 0 4px 4px 0;
      line-height: 1.4;
      text-align: left;
    }
    .connected-section {
      margin-top: 10px;
      border-top: 1px solid #2D3748;
      padding-top: 8px;
    }
    .connected-title {
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      color: #A0AEC0;
      letter-spacing: 0.5px;
      margin-bottom: 6px;
    }
    .entity-pill {
      display: inline-block;
      margin: 2px 4px 2px 0;
      padding: 3px 8px;
      background-color: #12171D;
      border: 1px solid #3B4758;
      border-radius: 4px;
      font-size: 11px;
      color: #E2E8F0;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .entity-pill:hover {
      border-color: #C5A15A;
      background-color: #2D3748;
      color: #C5A15A;
    }
    .status-badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: 600;
      background-color: #1A2E22;
      color: #48BB78;
      border: 1px solid #48BB78;
    }
  </style>
</head>
<body>

  <div id="toolbar">
    <div class="tool-group">
      <label for="categoryFilter">Category:</label>
      <select id="categoryFilter" onchange="filterGraph()">
        <option value="ALL">All Categories</option>
        <option value="DYNASTY">Dynasties (Heraldic Standards)</option>
        <option value="RULER">Rulers & Kings</option>
        <option value="TEMPLE">Sacred Sites / Temples</option>
        <option value="LOCATION">Locations</option>
        <option value="DONOR">Donors & Ascetics</option>
        <option value="INSCRIPTION">Inscriptions</option>
        <option value="HISTORICAL_EVENT">Historical Events</option>
        <option value="GRANT">Grants & Endowments</option>
      </select>
    </div>

    <div class="tool-group">
      <label for="searchInput">Search:</label>
      <input type="text" id="searchInput" placeholder="Search entity / dynasty / king..." oninput="filterGraph()">
    </div>

    <button onclick="fitView()">⛶ Fit View</button>
    <button onclick="togglePhysics()">⚡ Toggle Physics</button>

    <div class="stat-pill">
      <span>__NODE_COUNT__ Entities | __EDGE_COUNT__ Verified Relations</span>
      <span id="engine-badge" class="engine-indicator">Initializing...</span>
    </div>
  </div>

  <div id="main-container">
    <div id="network-canvas">
      <div id="canvas-container"></div>
      <canvas id="fallback-canvas" style="display: none;"></canvas>
    </div>
    <div id="side-panel">
      <div class="panel-header">
        <span>Entity Inspection</span>
        <span id="selected-badge" style="font-size: 10px; color: #A0AEC0;">None selected</span>
      </div>
      <div id="panel-content">
        <p style="color: #718096; font-style: italic;">Select any node in the graph to inspect epigraphical citations, historical dates, dynastic flags, and linked records.</p>
      </div>
    </div>
  </div>

  <script type="text/javascript">
    const rawNodes = __NODES_JSON__;
    const rawEdges = __EDGES_JSON__;
    const validationData = __VALIDATION_JSON__;
    const dynastyFallbackSvgs = __FALLBACK_SVGS_JSON__;

    let physicsEnabled = true;
    let network = null;
    let fallbackEngine = null;
    let activeEngineType = 'unknown';

    // 1. Data Validation & Diagnostic Logging
    console.log("[Kallvettu Knowledge Graph] Payload validation:", validationData);
    console.log("[Kallvettu Knowledge Graph] Loaded raw nodes: " + rawNodes.length + ", raw edges: " + rawEdges.length);

    // Filter valid nodes and edges
    const nodeMap = new Map();
    rawNodes.forEach(n => {
      if (n && n.id) {
        nodeMap.set(n.id, n);
      }
    });

    const validEdges = rawEdges.filter(e => {
      const srcOk = nodeMap.has(e.source);
      const tgtOk = nodeMap.has(e.target);
      if (!srcOk || !tgtOk) {
        console.warn("[Kallvettu Knowledge Graph] Dropping edge with missing node:", e);
        return false;
      }
      return true;
    });

    // 2. Pre-seed Deterministic Initial Coordinates to Prevent Singularity Collapse (0,0)
    function assignInitialCoordinates(nodes) {
      const dynCoords = {
        'DYN-CHOLA': { x: -220, y: -150 },
        'DYN-PANDYA': { x: -220, y: 150 },
        'DYN-CHERA': { x: 220, y: -150 },
        'DYN-PALLAVA': { x: 220, y: 150 }
      };

      let nonDynIndex = 0;
      const nonDynTotal = nodes.filter(n => !dynCoords[n.id]).length;

      nodes.forEach(n => {
        if (dynCoords[n.id]) {
          n.initialX = dynCoords[n.id].x;
          n.initialY = dynCoords[n.id].y;
        } else {
          const angle = (2 * Math.PI * nonDynIndex) / Math.max(1, nonDynTotal);
          const radX = 320 + (nonDynIndex % 4) * 45;
          const radY = 240 + (nonDynIndex % 3) * 45;
          n.initialX = Math.round(radX * Math.cos(angle));
          n.initialY = Math.round(radY * Math.sin(angle));
          nonDynIndex++;
        }
      });
    }
    assignInitialCoordinates(rawNodes);

    // 3. Build Vis.js Nodes
    function buildVisNodes(filterCat, searchTerm) {
      return rawNodes
        .filter(n => {
          const matchCat = filterCat === 'ALL' || n.category === filterCat;
          const matchSearch = !searchTerm ||
            n.label.toLowerCase().includes(searchTerm) ||
            (n.label_ta && n.label_ta.toLowerCase().includes(searchTerm)) ||
            (n.description && n.description.toLowerCase().includes(searchTerm)) ||
            (n.emblem_name && n.emblem_name.toLowerCase().includes(searchTerm));
          return matchCat && matchSearch;
        })
        .map(n => {
          const isDynastyWithFlag = n.category === 'DYNASTY' && (n.flag_data_uri || n.fallback_svg_data_uri || dynastyFallbackSvgs[n.id]);
          
          let shape = 'dot';
          let size = 20;

          if (isDynastyWithFlag) {
            shape = 'image';
            size = 46;
          } else if (n.category === 'DYNASTY') {
            shape = 'diamond';
            size = 32;
          } else if (n.category === 'INSCRIPTION') {
            shape = 'box';
            size = 22;
          } else if (n.category === 'RULER') {
            shape = 'dot';
            size = 26;
          } else if (n.category === 'TEMPLE') {
            shape = 'dot';
            size = 24;
          } else if (n.category === 'LOCATION') {
            shape = 'dot';
            size = 22;
          } else if (n.category === 'HISTORICAL_EVENT') {
            shape = 'diamond';
            size = 24;
          } else if (n.category === 'GRANT') {
            shape = 'ellipse';
            size = 22;
          } else if (n.category === 'DONOR') {
            shape = 'dot';
            size = 22;
          }

          let tooltip = n.label;
          if (n.label_ta) tooltip += ' (' + n.label_ta + ')';
          if (n.period) tooltip += '\\nEra: ' + n.period;
          if (n.century) tooltip += '\\nChronology: ' + n.century;
          if (n.emblem_name) tooltip += '\\nRoyal Crest: ' + n.emblem_name;
          if (n.verification_status) tooltip += '\\nStatus: ' + n.verification_status;
          tooltip += '\\n[Click to inspect full epigraphical dossier]';

          const flagImg = n.flag_data_uri || n.fallback_svg_data_uri || dynastyFallbackSvgs[n.id];
          const fallbackImg = n.fallback_svg_data_uri || dynastyFallbackSvgs[n.id] || dynastyFallbackSvgs['GENERIC'];

          const nodeConfig = {
            id: n.id,
            label: n.label + (n.label_ta ? "\\n(" + n.label_ta + ")" : ""),
            title: tooltip,
            shape: shape,
            size: size,
            x: n.initialX,
            y: n.initialY,
            color: {
              background: n.color || "#C5A15A",
              border: isDynastyWithFlag ? "#C5A15A" : "#E2E8F0",
              highlight: {
                background: "#202832",
                border: "#FFFFFF"
              },
              hover: {
                background: "#2D3748",
                border: "#F59E0B"
              }
            },
            font: {
              color: "#FFFFFF",
              size: isDynastyWithFlag ? 13 : 12,
              strokeWidth: 3,
              strokeColor: "#12171E",
              vadjust: isDynastyWithFlag ? 6 : 0
            },
            raw: n
          };

          if (isDynastyWithFlag) {
            nodeConfig.image = flagImg;
            nodeConfig.brokenImage = fallbackImg;
            nodeConfig.shapeProperties = {
              useBorderWithImage: true,
              useImageSize: false,
              interpolation: true
            };
          }

          return nodeConfig;
        });
    }

    let nodesDataSet = null;
    let edgesDataSet = null;

    // 4. Fallback HTML5 Canvas Force Graph Engine (Activates if Vis.js is blocked or offline)
    class FallbackCanvasGraphEngine {
      constructor(canvasEl, nodes, edges) {
        this.canvas = canvasEl;
        this.ctx = canvasEl.getContext('2d');
        this.rawNodes = nodes;
        this.rawEdges = edges;
        this.nodes = [];
        this.edges = [];
        this.panX = 0;
        this.panY = 0;
        this.zoom = 1.0;
        this.isPanning = false;
        this.startPan = { x: 0, y: 0 };
        this.dragNode = null;
        this.hoverNode = null;
        this.selectedNode = null;
        this.running = true;
        this.flagImages = {};
        this.animFrameId = null;

        this.preloadFlags();
        this.initGraph('ALL', '');
        this.bindEvents();
        this.resize();
        this.fit();
        this.loop();
      }

      preloadFlags() {
        this.rawNodes.forEach(n => {
          if (n.category === 'DYNASTY') {
            const img = new Image();
            img.src = n.flag_data_uri || n.fallback_svg_data_uri || dynastyFallbackSvgs[n.id];
            this.flagImages[n.id] = img;
          }
        });
      }

      initGraph(filterCat, searchTerm) {
        const filteredRaw = this.rawNodes.filter(n => {
          const matchCat = filterCat === 'ALL' || n.category === filterCat;
          const matchSearch = !searchTerm ||
            n.label.toLowerCase().includes(searchTerm) ||
            (n.label_ta && n.label_ta.toLowerCase().includes(searchTerm)) ||
            (n.description && n.description.toLowerCase().includes(searchTerm)) ||
            (n.emblem_name && n.emblem_name.toLowerCase().includes(searchTerm));
          return matchCat && matchSearch;
        });

        const activeIdSet = new Set(filteredRaw.map(n => n.id));

        this.nodes = filteredRaw.map(n => {
          const isDyn = n.category === 'DYNASTY';
          return {
            id: n.id,
            label: n.label,
            label_ta: n.label_ta || '',
            category: n.category,
            color: n.color || '#C5A15A',
            isDynasty: isDyn,
            x: n.initialX || 0,
            y: n.initialY || 0,
            vx: 0,
            vy: 0,
            radius: isDyn ? 32 : 18,
            boxW: isDyn ? 68 : 36,
            boxH: isDyn ? 42 : 36,
            raw: n
          };
        });

        const nodeObjMap = new Map();
        this.nodes.forEach(n => nodeObjMap.set(n.id, n));

        this.edges = this.rawEdges
          .filter(e => activeIdSet.has(e.source) && activeIdSet.has(e.target))
          .map(e => ({
            source: nodeObjMap.get(e.source),
            target: nodeObjMap.get(e.target),
            label: e.label,
            raw: e
          }));
      }

      resize() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width || (window.innerWidth - 330) || 800;
        this.canvas.height = rect.height || (parseInt('__HEIGHT_PX__') - 52) || 600;
      }

      fit() {
        if (this.nodes.length === 0) return;
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        this.nodes.forEach(n => {
          if (n.x < minX) minX = n.x;
          if (n.x > maxX) maxX = n.x;
          if (n.y < minY) minY = n.y;
          if (n.y > maxY) maxY = n.y;
        });
        const w = Math.max(100, maxX - minX + 160);
        const h = Math.max(100, maxY - minY + 160);
        this.zoom = Math.min(this.canvas.width / w, this.canvas.height / h, 1.2);
        this.panX = this.canvas.width / 2 - ((minX + maxX) / 2) * this.zoom;
        this.panY = this.canvas.height / 2 - ((minY + maxY) / 2) * this.zoom;
      }

      togglePhysics() {
        this.running = !this.running;
      }

      updatePhysics() {
        if (!this.running) return;

        // Repulsion between nodes
        for (let i = 0; i < this.nodes.length; i++) {
          for (let j = i + 1; j < this.nodes.length; j++) {
            const n1 = this.nodes[i];
            const n2 = this.nodes[j];
            const dx = n2.x - n1.x;
            const dy = n2.y - n1.y;
            const distSq = dx * dx + dy * dy + 100;
            const dist = Math.sqrt(distSq);
            const force = 9000 / distSq;
            const fx = (dx / dist) * force;
            const fy = (dy / dist) * force;
            n1.vx -= fx;
            n1.vy -= fy;
            n2.vx += fx;
            n2.vy += fy;
          }
        }

        // Spring attraction along edges
        for (let i = 0; i < this.edges.length; i++) {
          const e = this.edges[i];
          const dx = e.target.x - e.source.x;
          const dy = e.target.y - e.source.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const targetDist = 140;
          const springForce = (dist - targetDist) * 0.035;
          const fx = (dx / dist) * springForce;
          const fy = (dy / dist) * springForce;
          e.source.vx += fx;
          e.source.vy += fy;
          e.target.vx -= fx;
          e.target.vy -= fy;
        }

        // Center gravity and velocity integration
        this.nodes.forEach(n => {
          if (n === this.dragNode) return;
          n.vx -= n.x * 0.006;
          n.vy -= n.y * 0.006;
          n.vx *= 0.85;
          n.vy *= 0.85;
          n.x += n.vx;
          n.y += n.vy;
        });
      }

      draw() {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        ctx.save();
        ctx.translate(this.panX, this.panY);
        ctx.scale(this.zoom, this.zoom);

        // 1. Draw Edges
        this.edges.forEach(e => {
          const isHighlight = this.selectedNode && (e.source.id === this.selectedNode.id || e.target.id === this.selectedNode.id);
          ctx.beginPath();
          ctx.moveTo(e.source.x, e.source.y);
          ctx.lineTo(e.target.x, e.target.y);
          ctx.strokeStyle = isHighlight ? '#C5A15A' : '#3E4C5E';
          ctx.lineWidth = isHighlight ? 2.2 : 1.2;
          ctx.stroke();

          // Arrowhead
          const angle = Math.atan2(e.target.y - e.source.y, e.target.x - e.source.x);
          const tgtRadius = e.target.radius + 6;
          const arrX = e.target.x - Math.cos(angle) * tgtRadius;
          const arrY = e.target.y - Math.sin(angle) * tgtRadius;
          ctx.beginPath();
          ctx.moveTo(arrX, arrY);
          ctx.lineTo(arrX - 8 * Math.cos(angle - 0.4), arrY - 8 * Math.sin(angle - 0.4));
          ctx.lineTo(arrX - 8 * Math.cos(angle + 0.4), arrY - 8 * Math.sin(angle + 0.4));
          ctx.closePath();
          ctx.fillStyle = isHighlight ? '#C5A15A' : '#4A5568';
          ctx.fill();

          // Edge Label (middle)
          if (this.zoom > 0.65) {
            const midX = (e.source.x + e.target.x) / 2;
            const midY = (e.source.y + e.target.y) / 2;
            ctx.font = '9px sans-serif';
            ctx.fillStyle = isHighlight ? '#F7FAFC' : '#718096';
            ctx.textAlign = 'center';
            ctx.fillText(e.label, midX, midY - 3);
          }
        });

        // 2. Draw Nodes
        this.nodes.forEach(n => {
          const isSel = this.selectedNode && this.selectedNode.id === n.id;
          const isHov = this.hoverNode && this.hoverNode.id === n.id;

          if (n.isDynasty) {
            // Dynasty Flag Box with Contain Aspect Ratio Preservation
            const bw = n.boxW;
            const bh = n.boxH;
            const bx = n.x - bw / 2;
            const by = n.y - bh / 2;

            // Background card
            ctx.fillStyle = '#171D24';
            ctx.fillRect(bx, by, bw, bh);

            // Flag image rendering
            const img = this.flagImages[n.id];
            if (img && img.complete && img.naturalWidth > 0) {
              const pad = 4;
              const maxW = bw - pad * 2;
              const maxH = bh - pad * 2;
              const scale = Math.min(maxW / img.naturalWidth, maxH / img.naturalHeight);
              const drawW = img.naturalWidth * scale;
              const drawH = img.naturalHeight * scale;
              const drawX = n.x - drawW / 2;
              const drawY = n.y - drawH / 2;
              ctx.drawImage(img, drawX, drawY, drawW, drawH);
            } else {
              // Emblem placeholder
              ctx.font = '16px sans-serif';
              ctx.textAlign = 'center';
              ctx.fillText('👑', n.x, n.y + 5);
            }

            // Border
            ctx.strokeStyle = isSel ? '#FFFFFF' : (isHov ? '#F59E0B' : '#C5A15A');
            ctx.lineWidth = isSel ? 3 : 2;
            ctx.strokeRect(bx, by, bw, bh);

            if (isSel) {
              ctx.strokeStyle = 'rgba(197, 161, 90, 0.4)';
              ctx.lineWidth = 6;
              ctx.strokeRect(bx - 3, by - 3, bw + 6, bh + 6);
            }

            // Dynasty label below
            ctx.font = 'bold 12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillStyle = '#FFFFFF';
            ctx.lineWidth = 3;
            ctx.strokeStyle = '#12171E';
            ctx.strokeText(n.label, n.x, by + bh + 14);
            ctx.fillText(n.label, n.x, by + bh + 14);

            if (n.label_ta) {
              ctx.font = '10px sans-serif';
              ctx.fillStyle = '#C5A15A';
              ctx.strokeText('(' + n.label_ta + ')', n.x, by + bh + 26);
              ctx.fillText('(' + n.label_ta + ')', n.x, by + bh + 26);
            }
          } else {
            // Standard category shape
            ctx.beginPath();
            if (n.category === 'INSCRIPTION') {
              ctx.rect(n.x - 16, n.y - 12, 32, 24);
            } else if (n.category === 'HISTORICAL_EVENT') {
              ctx.moveTo(n.x, n.y - 14);
              ctx.lineTo(n.x + 14, n.y);
              ctx.lineTo(n.x, n.y + 14);
              ctx.lineTo(n.x - 14, n.y);
              ctx.closePath();
            } else {
              ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
            }

            ctx.fillStyle = n.color;
            ctx.fill();
            ctx.strokeStyle = isSel ? '#FFFFFF' : (isHov ? '#F59E0B' : '#CBD5E0');
            ctx.lineWidth = isSel ? 3 : 1.5;
            ctx.stroke();

            if (isSel) {
              ctx.beginPath();
              ctx.arc(n.x, n.y, n.radius + 5, 0, Math.PI * 2);
              ctx.strokeStyle = 'rgba(197, 161, 90, 0.5)';
              ctx.lineWidth = 3;
              ctx.stroke();
            }

            // Label
            ctx.font = '11px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillStyle = '#FFFFFF';
            ctx.lineWidth = 3;
            ctx.strokeStyle = '#12171E';
            ctx.strokeText(n.label, n.x, n.y + n.radius + 13);
            ctx.fillText(n.label, n.x, n.y + n.radius + 13);
          }
        });

        ctx.restore();
      }

      loop() {
        this.updatePhysics();
        this.draw();
        this.animFrameId = requestAnimationFrame(() => this.loop());
      }

      screenToWorld(clientX, clientY) {
        const rect = this.canvas.getBoundingClientRect();
        const sx = clientX - rect.left;
        const sy = clientY - rect.top;
        return {
          x: (sx - this.panX) / this.zoom,
          y: (sy - this.panY) / this.zoom
        };
      }

      getNodeAt(clientX, clientY) {
        const world = this.screenToWorld(clientX, clientY);
        for (let i = this.nodes.length - 1; i >= 0; i--) {
          const n = this.nodes[i];
          const dx = world.x - n.x;
          const dy = world.y - n.y;
          const hitRadius = n.isDynasty ? 38 : n.radius + 4;
          if (dx * dx + dy * dy <= hitRadius * hitRadius) {
            return n;
          }
        }
        return null;
      }

      bindEvents() {
        window.addEventListener('resize', () => {
          this.resize();
          this.fit();
        });

        this.canvas.addEventListener('mousedown', e => {
          const node = this.getNodeAt(e.clientX, e.clientY);
          if (node) {
            this.dragNode = node;
            this.selectedNode = node;
            showEntityDetails(node.raw);
          } else {
            this.isPanning = true;
            this.startPan = { x: e.clientX - this.panX, y: e.clientY - this.panY };
          }
        });

        this.canvas.addEventListener('mousemove', e => {
          if (this.dragNode) {
            const world = this.screenToWorld(e.clientX, e.clientY);
            this.dragNode.x = world.x;
            this.dragNode.y = world.y;
            this.dragNode.vx = 0;
            this.dragNode.vy = 0;
          } else if (this.isPanning) {
            this.panX = e.clientX - this.startPan.x;
            this.panY = e.clientY - this.startPan.y;
          } else {
            const node = this.getNodeAt(e.clientX, e.clientY);
            this.hoverNode = node;
            this.canvas.style.cursor = node ? 'pointer' : 'grab';
          }
        });

        window.addEventListener('mouseup', () => {
          this.dragNode = null;
          this.isPanning = false;
        });

        this.canvas.addEventListener('wheel', e => {
          e.preventDefault();
          const rect = this.canvas.getBoundingClientRect();
          const mx = e.clientX - rect.left;
          const my = e.clientY - rect.top;
          const zoomFactor = e.deltaY < 0 ? 1.15 : 0.88;
          const newZoom = Math.max(0.2, Math.min(3.5, this.zoom * zoomFactor));

          this.panX = mx - (mx - this.panX) * (newZoom / this.zoom);
          this.panY = my - (my - this.panY) * (newZoom / this.zoom);
          this.zoom = newZoom;
        }, { passive: false });
      }

      selectAndFocus(nodeId) {
        const target = this.nodes.find(n => n.id === nodeId);
        if (target) {
          this.selectedNode = target;
          showEntityDetails(target.raw);
          this.panX = this.canvas.width / 2 - target.x * this.zoom;
          this.panY = this.canvas.height / 2 - target.y * this.zoom;
        }
      }
    }

    // 5. Unified Control Functions
    function fitView() {
      if (activeEngineType === 'vis' && network) {
        network.fit({animation: true});
      } else if (fallbackEngine) {
        fallbackEngine.fit();
      }
    }

    function togglePhysics() {
      physicsEnabled = !physicsEnabled;
      if (activeEngineType === 'vis' && network) {
        network.setOptions({ physics: { enabled: physicsEnabled } });
      } else if (fallbackEngine) {
        fallbackEngine.togglePhysics();
      }
    }

    function filterGraph() {
      const cat = document.getElementById('categoryFilter').value;
      const q = document.getElementById('searchInput').value.trim().toLowerCase();

      if (activeEngineType === 'vis' && nodesDataSet) {
        const filtered = buildVisNodes(cat, q);
        nodesDataSet.clear();
        nodesDataSet.add(filtered);
        setTimeout(() => { if (network) network.fit({ animation: true }); }, 150);
      } else if (fallbackEngine) {
        fallbackEngine.initGraph(cat, q);
        fallbackEngine.fit();
      }
    }

    window.inspectNodeById = function(id) {
      const nodeItem = rawNodes.find(item => item.id === id);
      if (nodeItem) {
        showEntityDetails(nodeItem);
        if (activeEngineType === 'vis' && network) {
          network.focus(id, { scale: 1.15, animation: { duration: 400, easingFunction: 'easeInOutQuad' } });
          network.selectNodes([id]);
        } else if (fallbackEngine) {
          fallbackEngine.selectAndFocus(id);
        }
      }
    };

    // 6. Side Panel Entity Inspection Renderer
    function showEntityDetails(n) {
      const badgeEl = document.getElementById('selected-badge');
      if (badgeEl) badgeEl.innerText = n.id;

      // Find connected relations
      const outgoing = rawEdges.filter(e => e.source === n.id);
      const incoming = rawEdges.filter(e => e.target === n.id);
      const findNode = (id) => rawNodes.find(item => item.id === id);

      const rulers = [];
      const inscriptions = [];
      const temples = [];
      const events = [];
      const grants = [];
      const dynasties = [];

      function categorizeNeighbor(neighborId, relationLabel, direction) {
        const neighbor = findNode(neighborId);
        if (!neighbor) return;
        const entry = { node: neighbor, relation: relationLabel, direction: direction };
        if (neighbor.category === 'RULER') rulers.push(entry);
        else if (neighbor.category === 'INSCRIPTION') inscriptions.push(entry);
        else if (neighbor.category === 'TEMPLE' || neighbor.category === 'LOCATION') temples.push(entry);
        else if (neighbor.category === 'HISTORICAL_EVENT') events.push(entry);
        else if (neighbor.category === 'GRANT') grants.push(entry);
        else if (neighbor.category === 'DYNASTY') dynasties.push(entry);
      }

      outgoing.forEach(e => categorizeNeighbor(e.target, e.label, 'out'));
      incoming.forEach(e => categorizeNeighbor(e.source, e.label, 'in'));

      let flagHtml = '';
      const flagImgSrc = n.flag_data_uri || n.fallback_svg_data_uri || dynastyFallbackSvgs[n.id];
      if (n.category === 'DYNASTY' && flagImgSrc) {
        flagHtml = `
          <div class="flag-preview-card">
            <div style="font-size: 11px; text-transform: uppercase; color: #A0AEC0; margin-bottom: 6px; letter-spacing: 0.5px;">Dynastic Flag & Royal Standard</div>
            <img src="${flagImgSrc}" alt="${n.label} Flag" class="flag-preview-img" />
            <div class="emblem-title">${n.emblem_name || 'Traditional Heraldic Standard'}</div>
            ${n.emblem_description ? `<div class="emblem-desc">${n.emblem_description}</div>` : ''}
            <div class="provenance-card">
              <strong>Provenance Note:</strong> ${n.emblem_provenance || 'User-provided reference emblem (traditional/reconstructed heraldic representation, not authenticated archaeological artifact).'}
            </div>
            ${n.emblem_source ? `<div style="font-size: 10px; color: #A0AEC0; margin-top: 4px; text-align: left;"><strong>Emblem Citation:</strong> ${n.emblem_source}</div>` : ''}
          </div>
        `;
      }

      function renderPillList(title, list) {
        if (!list || list.length === 0) return '';
        const pills = list.map(item => `
          <span class="entity-pill" onclick="inspectNodeById('${item.node.id}')" title="${item.relation}">
            ${item.node.label} <span style="font-size:9.5px; opacity:0.7;">(${item.relation})</span>
          </span>
        `).join('');
        return `
          <div class="connected-section">
            <div class="connected-title">${title} (${list.length})</div>
            <div>${pills}</div>
          </div>
        `;
      }

      const html = `
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span class="badge">${n.category}</span>
          <span style="font-size:11px; color:#A0AEC0;">${n.id}</span>
        </div>
        <div style="font-size:16px; font-weight:bold; color:#FFFFFF; margin-top:8px;">${n.label}</div>
        ${n.label_ta ? `<div style="font-size:13px; color:#C5A15A;">${n.label_ta}</div>` : ''}
        
        ${flagHtml}

        <div style="margin-top:10px;">
          ${n.period ? `
            <div class="prop-row">
              <div class="prop-name">Historical Era</div>
              <div class="prop-val">${n.period}</div>
            </div>` : ''}
          ${n.century ? `
            <div class="prop-row">
              <div class="prop-name">Century / Chronology</div>
              <div class="prop-val">${n.century}</div>
            </div>` : ''}
          ${n.description ? `
            <div class="prop-row" style="margin-top:6px;">
              <div class="prop-name">Historical Description</div>
              <div class="prop-val" style="color:#CBD5E0;">${n.description}</div>
            </div>` : ''}
          ${n.verification_status ? `
            <div class="prop-row" style="margin-top:6px;">
              <div class="prop-name">Review & Verification Status</div>
              <div class="prop-val"><span class="status-badge">${n.verification_status}</span></div>
            </div>` : ''}
          ${n.citation ? `
            <div class="prop-row" style="margin-top:10px;">
              <div class="prop-name">Scholarly Source Citation</div>
              <div class="citation-box">${n.citation}</div>
            </div>` : ''}
        </div>

        ${renderPillList('Associated Dynasties', dynasties)}
        ${renderPillList('Associated Rulers & Kings', rulers)}
        ${renderPillList('Associated Inscription Records', inscriptions)}
        ${renderPillList('Associated Temples & Sites', temples)}
        ${renderPillList('Associated Historical Events', events)}
        ${renderPillList('Endowments & Land Grants', grants)}
      `;

      document.getElementById('panel-content').innerHTML = html;
    }

    // 7. Graph Initialization Orchestrator
    function initKnowledgeGraph() {
      const container = document.getElementById('canvas-container');
      const fallbackCanvas = document.getElementById('fallback-canvas');
      const badge = document.getElementById('engine-badge');

      // Verify if Vis.js library is loaded
      const hasVis = typeof vis !== 'undefined' && Boolean(vis.Network) && Boolean(vis.DataSet);

      if (hasVis) {
        try {
          console.log("[Kallvettu Knowledge Graph] Initializing Vis.js Network Engine.");
          activeEngineType = 'vis';
          badge.innerText = 'Vis.js Active';

          nodesDataSet = new vis.DataSet(buildVisNodes('ALL', ''));
          edgesDataSet = new vis.DataSet(validEdges.map(e => ({
            from: e.source,
            to: e.target,
            label: e.label,
            arrows: 'to',
            color: { color: '#4A5568', highlight: '#C5A15A' },
            font: { color: '#A0AEC0', size: 10, align: 'middle' },
            raw: e
          })));

          const data = {
            nodes: nodesDataSet,
            edges: edgesDataSet
          };

          const options = {
            autoResize: true,
            width: '100%',
            height: '100%',
            layout: {
              randomSeed: 42,
              improvedLayout: true
            },
            physics: {
              enabled: physicsEnabled,
              barnesHut: {
                gravitationalConstant: -4000,
                centralGravity: 0.35,
                springLength: 130,
                springConstant: 0.04,
                damping: 0.09,
                avoidOverlap: 0.25
              },
              stabilization: {
                enabled: true,
                iterations: 150,
                fit: true
              }
            },
            interaction: {
              hover: true,
              tooltipDelay: 150,
              navigationButtons: true,
              keyboard: true,
              zoomView: true,
              dragView: true,
              dragNodes: true
            }
          };

          network = new vis.Network(container, data, options);

          network.on('click', function(params) {
            if (params.nodes.length > 0) {
              const nodeId = params.nodes[0];
              const nodeItem = rawNodes.find(n => n.id === nodeId);
              if (nodeItem) {
                showEntityDetails(nodeItem);
              }
            }
          });

          network.once('stabilizationIterationsDone', function() {
            network.fit({ animation: false });
          });

          // ResizeObserver to ensure canvas size is always in sync with parent iframe
          if (window.ResizeObserver) {
            const ro = new ResizeObserver(() => {
              if (network) {
                network.setSize('100%', '100%');
                network.redraw();
              }
            });
            ro.observe(document.getElementById('network-canvas'));
          }

          // Initial safety fit
          setTimeout(() => {
            if (network) {
              network.setSize('100%', '100%');
              network.fit({ animation: false });
            }
          }, 350);

          return;
        } catch (err) {
          console.error("[Kallvettu Knowledge Graph] Vis.js initialization failed, falling back to Canvas Engine:", err);
        }
      }

      // Fallback: Launch pure HTML5 Canvas Force Graph Engine
      console.log("[Kallvettu Knowledge Graph] Launching Canvas Force Graph Engine (Offline/SPOF-Resilient).");
      activeEngineType = 'canvas';
      badge.innerText = 'Canvas Engine Active';
      container.style.display = 'none';
      fallbackCanvas.style.display = 'block';

      fallbackEngine = new FallbackCanvasGraphEngine(fallbackCanvas, rawNodes, validEdges);
      setTimeout(() => fallbackEngine.fit(), 200);
    }

    // Launch when DOM is ready
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', initKnowledgeGraph);
    } else {
      initKnowledgeGraph();
    }
  </script>
</body>
</html>
"""


def render_knowledge_graph_html(
    kg: EpigraphicalKnowledgeGraph,
    height_px: int = 650,
) -> str:
    """
    Renders the knowledge graph as an interactive visualization.
    Validates graph data and embeds fallback assets and diagnostics.
    """
    graph_data = kg.to_dict()
    nodes_json = json.dumps(graph_data["nodes"], ensure_ascii=False)
    edges_json = json.dumps(graph_data["edges"], ensure_ascii=False)
    validation_json = json.dumps(graph_data.get("validation", {}), ensure_ascii=False)
    fallback_svgs_json = json.dumps(FALLBACK_DYNASTY_SVGS, ensure_ascii=False)

    return (
        HTML_TEMPLATE.replace("__HEIGHT_PX__", str(height_px))
        .replace("__NODE_COUNT__", str(graph_data["node_count"]))
        .replace("__EDGE_COUNT__", str(graph_data["edge_count"]))
        .replace("__NODES_JSON__", nodes_json)
        .replace("__EDGES_JSON__", edges_json)
        .replace("__VALIDATION_JSON__", validation_json)
        .replace("__FALLBACK_SVGS_JSON__", fallback_svgs_json)
    )
