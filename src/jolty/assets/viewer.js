// The jolty 3D viewer: a circuit laid out on a board, its simulation played on it.
//
// The page embeds the scene as JSON: each component's schematic layout, the net of the
// whole circuit each local net is, and every analysis's points. The board draws one
// component instance at a time, its parts standing on their pins and its wires as traces.
// Each frame colors the traces by voltage, moves light along each lead at a speed that
// grows with its current, and lights each part with the power it takes.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const root = document.getElementById("jolty");
const data = JSON.parse(document.getElementById("scene").textContent);

// Formatting.

const PREFIXES = [[1e12, "T"], [1e9, "G"], [1e6, "M"], [1e3, "k"], [1, ""], [1e-3, "m"], [1e-6, "µ"], [1e-9, "n"], [1e-12, "p"], [1e-15, "f"], [1e-18, "a"]];

function si(value, unit = "") {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (value === 0) return `0 ${unit}`.trim();
  const magnitude = Math.abs(value);
  for (const [scale, prefix] of PREFIXES) {
    if (magnitude >= scale * 0.9995) return `${(value / scale).toPrecision(4).replace(/\.?0+$/, "")} ${prefix}${unit}`.trim();
  }
  return `${value.toExponential(3)} ${unit}`.trim();
}

const h = (tag, attributes = {}, ...children) => {
  const element = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (key === "class") element.className = value;
    else if (key.startsWith("on")) element.addEventListener(key.slice(2), value);
    else element.setAttribute(key, value);
  }
  for (const child of children.flat()) element.append(child instanceof Node ? child : document.createTextNode(String(child)));
  return element;
};

// Colors.

const VIRIDIS = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"];
const DIVERGING = ["#2a5fd6", "#7fa3e8", "#bdbdb8", "#f0a070", "#e0502a"];

function ramp(stops, t) {
  const clamped = Math.min(1, Math.max(0, t));
  const position = clamped * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.floor(position));
  const a = new THREE.Color(stops[index]);
  const b = new THREE.Color(stops[index + 1]);
  return a.lerp(b, position - index);
}

// The shell: bars, panels and the canvas.

root.innerHTML = "";
const canvas = h("canvas", { class: "stage" });
const title = h("h1", {}, data.title);
const crumbs = h("span", { class: "crumbs" });
const tools = h("div", { class: "tools" });
const topBar = h("div", { class: "bar top-bar" }, h("div", { class: "title" }, title, crumbs), tools);
const analysisSelect = h("select", { "aria-label": "Analysis" });
const play = h("button", { title: "Play or pause (space)" }, "▶");
const slider = h("input", { type: "range", min: "0", max: "0", value: "0", "aria-label": "Point" });
const readout = h("span", { class: "readout" });
const speed = h("select", { "aria-label": "Speed" }, ...["0.25", "0.5", "1", "2", "4"].map((value) => h("option", value === "1" ? { value, selected: "" } : { value }, `${value}×`)));
const timeline = h("div", { class: "bar timeline" }, analysisSelect, play, slider, readout, speed);
const inspector = h("div", { class: "panel inspector", role: "dialog", "aria-label": "Inspector" });
const plotCanvas = h("canvas", { class: "chart" });
const plotHeader = h("header", {}, h("span", {}, "Voltages of the named nets"), h("span", {}, ""));
const plotPanel = h("div", { class: "panel plot open" }, plotHeader, plotCanvas);
const legendRamp = h("div", { class: "ramp" });
const legendEnds = h("div", { class: "ends" }, h("span", {}, ""), h("span", {}, ""));
const legend = h("div", { class: "bar legend" }, h("span", { class: "caption" }, "Net voltage"), legendRamp, legendEnds);
const tooltip = h("div", { class: "tooltip" });
const help = h(
  "div",
  { class: "panel help", role: "dialog", "aria-label": "Help" },
  h("h2", {}, "The 3D viewer"),
  h("p", {}, "Each trace takes its net's voltage as a color, light runs along each lead as fast as its current grows, and each part glows with the power it takes."),
  h("p", {}, "Click a part or a trace to inspect it; double-click a component to open it."),
  h("table", {},
    ...[["Space", "Play or pause"], ["← →", "Step a point"], ["Home End", "First or last point"], ["F", "Fit the view"], ["T", "View from the top"], ["L", "Labels"], ["U or Esc", "Up a level, or close"], ["?", "This help"]].map(([key, what]) => h("tr", {}, h("td", {}, h("kbd", {}, key)), h("td", {}, what)))),
  h("p", {}, h("button", { onclick: () => help.classList.remove("open") }, "Close")),
);
root.append(canvas, topBar, timeline, inspector, plotPanel, legend, tooltip, help);

// The renderer.

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
const world = new THREE.Scene();
const dark = matchMedia("(prefers-color-scheme: dark)").matches;
world.background = new THREE.Color(dark ? "#111214" : "#eef0f2");
const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 2000);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI * 0.49;
world.add(new THREE.HemisphereLight("#ffffff", "#445544", 1.2));
const sun = new THREE.DirectionalLight("#ffffff", 1.6);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
world.add(sun);
world.add(sun.target);
const board = new THREE.Group();
world.add(board);

// The board's state.

let viewPath = "";
let analysisIndex = 0;
let frame = 0;
let playing = false;
let showLabels = true;
let showFlow = true;
let selection = null;
let traces = [];
let parts = [];
let leads = [];
let labels = [];
let bounds = null;
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

const analysis = () => data.analyses[analysisIndex];
const view = () => data.views[viewPath];
const pathOf = (id) => (viewPath ? `${viewPath}/${id}` : id);

function globalNet(local) {
  return view().nets[local] ?? local;
}

function voltage(global, at = frame) {
  if (global === data.ground) return 0;
  const series = analysis()?.nets[global];
  return series ? series[Math.min(at, series.length - 1)] : null;
}

function level(global, at = frame) {
  const series = analysis()?.logic?.[global];
  return series ? series[Math.min(at, series.length - 1)] : undefined;
}

function current(terminal, at = frame) {
  const series = analysis()?.currents[terminal];
  return series ? series[Math.min(at, series.length - 1)] : null;
}

// Geometry: one material per look, parts built of primitives.

const materials = {
  board: new THREE.MeshStandardMaterial({ color: "#1d4d38", roughness: 0.85 }),
  edge: new THREE.MeshStandardMaterial({ color: "#163a2a", roughness: 0.9 }),
  pad: new THREE.MeshStandardMaterial({ color: "#c9a14a", metalness: 0.8, roughness: 0.35 }),
  lead: new THREE.MeshStandardMaterial({ color: "#c8ccd0", metalness: 0.9, roughness: 0.3 }),
  black: new THREE.MeshStandardMaterial({ color: "#1b1b1d", roughness: 0.55 }),
  label: new THREE.MeshStandardMaterial({ color: "#e8e8e8", roughness: 0.6 }),
};

const BAND_COLORS = ["#000000", "#7a4a1d", "#d62f2f", "#f08a1c", "#f2d31b", "#2f9e44", "#2a5fd6", "#8a4fd6", "#8a8a8a", "#f5f5f5"];

function bodyMaterial(color) {
  return new THREE.MeshStandardMaterial({ color, roughness: 0.5, metalness: 0.05, emissive: new THREE.Color("#ff6a00"), emissiveIntensity: 0 });
}

function textTexture(text, { color = "#ffffff", background = null, size = 64 } = {}) {
  const canvasText = document.createElement("canvas");
  const context = canvasText.getContext("2d");
  context.font = `600 ${size}px ${getComputedStyle(document.body).fontFamily}`;
  const width = Math.ceil(context.measureText(text).width) + size;
  canvasText.width = width;
  canvasText.height = Math.ceil(size * 1.5);
  context.font = `600 ${size}px ${getComputedStyle(document.body).fontFamily}`;
  if (background) {
    context.fillStyle = background;
    context.fillRect(0, 0, canvasText.width, canvasText.height);
  }
  context.fillStyle = color;
  context.textBaseline = "middle";
  context.textAlign = "center";
  context.fillText(text, width / 2, canvasText.height / 2);
  const texture = new THREE.CanvasTexture(canvasText);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  return { texture, aspect: canvasText.width / canvasText.height };
}

function sprite(text, height = 0.8) {
  const { texture, aspect } = textTexture(text, { color: dark ? "#f4f4f2" : "#111111" });
  const material = new THREE.SpriteMaterial({ map: texture, depthWrite: false, transparent: true });
  const label = new THREE.Sprite(material);
  label.scale.set(height * aspect, height, 1);
  return label;
}

const toWorld = ([x, y], height = 0) => new THREE.Vector3(x, height, -y);

function cylinderBetween(a, b, radius, material, segments = 16) {
  const direction = new THREE.Vector3().subVectors(b, a);
  const length = direction.length();
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, length, segments), material);
  mesh.position.copy(a).addScaledVector(direction, 0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.clone().normalize());
  mesh.castShadow = true;
  return mesh;
}

function resistorBands(ohms) {
  if (!(ohms > 0)) return [];
  const exponent = Math.floor(Math.log10(ohms)) - 1;
  const digits = Math.round(ohms / 10 ** exponent);
  const first = Math.floor(digits / 10) % 10;
  const second = digits % 10;
  const multiplier = Math.max(0, Math.min(9, exponent));
  return [BAND_COLORS[first], BAND_COLORS[second], BAND_COLORS[multiplier], "#c9a14a"];
}

const TWO = {
  resistor: ["a", "b"],
  capacitor: ["a", "b"],
  inductor: ["a", "b"],
  diode: ["anode", "cathode"],
  voltage_source: ["positive", "negative"],
  current_source: ["positive", "negative"],
};

// A part: its body between or above its pins, its leads, and what clicking it picks.
function makePart(instance) {
  const group = new THREE.Group();
  const id = instance.id;
  const model = instance.device ?? instance.component;
  const pins = Object.fromEntries(Object.entries(instance.pins).map(([name, point]) => [name, toWorld(point)]));
  const partLeads = [];
  const bodies = [];
  const lift = 0.6;
  const addLead = (terminal, from, to) => {
    const lead = cylinderBetween(from, to, 0.045, materials.lead, 8);
    group.add(lead);
    partLeads.push({ terminal, from: from.clone(), to: to.clone() });
  };
  const pad = (point) => {
    const disc = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 0.03, 20), materials.pad);
    disc.position.copy(point).setY(0.015);
    group.add(disc);
  };
  Object.values(pins).forEach(pad);

  if (TWO[model] && pins[TWO[model][0]] && pins[TWO[model][1]]) {
    const [first, second] = TWO[model].map((terminal) => pins[terminal]);
    const axis = new THREE.Vector3().subVectors(second, first);
    const span = axis.length();
    const unit = axis.clone().normalize();
    const middle = first.clone().addScaledVector(axis, 0.5).setY(lift);
    const length = Math.min(span * 0.7, 3);
    const start = middle.clone().addScaledVector(unit, -length / 2);
    const end = middle.clone().addScaledVector(unit, length / 2);
    let radius = 0.36;
    let color = "#d9c7a1";
    if (model === "capacitor") { radius = 0.44; color = "#3d78c9"; }
    if (model === "inductor") { radius = 0.38; color = "#6b4a2b"; }
    if (model === "diode") { radius = 0.26; color = "#202022"; }
    if (model === "voltage_source") { radius = 0.55; color = "#c93d3d"; }
    if (model === "current_source") { radius = 0.55; color = "#7a4fc9"; }
    const material = bodyMaterial(color);
    const body = cylinderBetween(start, end, radius, material, 24);
    group.add(body);
    bodies.push(body);
    if (model === "resistor") {
      const bands = resistorBands(Number(instance.parameters.resistance));
      bands.forEach((band, index) => {
        const at = start.clone().addScaledVector(unit, length * (0.2 + 0.17 * index + (index === 3 ? 0.08 : 0)));
        const ring = cylinderBetween(at.clone().addScaledVector(unit, -0.05), at.clone().addScaledVector(unit, 0.05), radius + 0.012, new THREE.MeshStandardMaterial({ color: band, roughness: 0.4 }), 24);
        group.add(ring);
      });
    }
    if (model === "inductor") {
      const turns = 7;
      const points = [];
      const side = new THREE.Vector3(unit.z, 0, -unit.x);
      for (let index = 0; index <= turns * 24; index += 1) {
        const angle = (index / 24) * Math.PI * 2;
        const along = -length / 2 + (length * index) / (turns * 24);
        points.push(middle.clone().addScaledVector(unit, along).addScaledVector(side, Math.cos(angle) * (radius + 0.06)).add(new THREE.Vector3(0, Math.sin(angle) * (radius + 0.06), 0)));
      }
      const coil = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points), turns * 48, 0.05, 8), new THREE.MeshStandardMaterial({ color: "#c27a3a", metalness: 0.85, roughness: 0.3 }));
      coil.castShadow = true;
      group.add(coil);
    }
    if (model === "diode") {
      const at = start.clone().addScaledVector(unit, length * 0.82);
      group.add(cylinderBetween(at.clone().addScaledVector(unit, -0.07), at.clone().addScaledVector(unit, 0.07), radius + 0.012, materials.label, 24));
    }
    if (model === "voltage_source" || model === "current_source") {
      const cap = cylinderBetween(start.clone().addScaledVector(unit, -0.08), start, radius * 0.6, materials.lead, 24);
      group.add(cap);
    }
    addLead(TWO[model][0], first.clone().setY(0.02), first.clone().setY(lift));
    addLead(TWO[model][0], first.clone().setY(lift), start);
    addLead(TWO[model][1], second.clone().setY(0.02), second.clone().setY(lift));
    addLead(TWO[model][1], second.clone().setY(lift), end);
  } else {
    const [[left, bottom], [right, top]] = instance.body;
    const center = toWorld([(left + right) / 2, (bottom + top) / 2]);
    const chip = Boolean(instance.component) || Object.keys(pins).length > 4 || model?.startsWith("logic") || model?.includes("controlled") || model?.startsWith("behavioral");
    let width = Math.max(1.2, right - left - 0.4);
    let depth = Math.max(1.2, top - bottom - 0.4);
    let height = chip ? 0.5 : 0.8;
    let material;
    let body;
    if (!chip) {
      width = depth = 1.2;
      height = 1.1;
      material = bodyMaterial("#1e1e20");
      const geometry = new THREE.CylinderGeometry(0.65, 0.65, height, 32, 1, false, 0, Math.PI);
      body = new THREE.Mesh(geometry, material);
      body.position.copy(center).setY(height / 2 + 0.35);
    } else {
      material = bodyMaterial(instance.component ? "#202124" : "#2b2d31");
      body = new THREE.Mesh(new THREE.BoxGeometry(width, height, depth), material);
      body.position.copy(center).setY(height / 2 + 0.3);
      const name = instance.component ? `${id} · ${instance.component.split("[")[0]}` : `${id} · ${model}`;
      const { texture, aspect } = textTexture(name, { color: "#e8e8e8" });
      const face = new THREE.Mesh(new THREE.PlaneGeometry(Math.min(width * 0.9, depth * 0.6 * aspect), Math.min(width * 0.9, depth * 0.6 * aspect) / aspect), new THREE.MeshBasicMaterial({ map: texture, transparent: true }));
      face.rotation.x = -Math.PI / 2;
      face.position.copy(body.position).setY(body.position.y + height / 2 + 0.002);
      group.add(face);
    }
    body.castShadow = true;
    body.receiveShadow = true;
    group.add(body);
    bodies.push(body);
    const top3 = body.position.y - height / 2 + 0.05;
    for (const [terminal, pin] of Object.entries(pins)) {
      const toward = new THREE.Vector3(pin.x - center.x, 0, pin.z - center.z);
      const reach = chip ? Math.max(0.1, Math.min(Math.abs(toward.x) - width / 2, Math.abs(toward.z) - depth / 2)) : 0;
      const edge = chip
        ? center.clone().add(toward.clone().normalize().multiplyScalar(Math.max(0, toward.length() - reach - 0.05))).setY(top3)
        : center.clone().add(toward.clone().multiplyScalar(0.18)).setY(top3);
      addLead(terminal, pin.clone().setY(0.02), pin.clone().setY(top3));
      addLead(terminal, pin.clone().setY(top3), edge);
    }
  }
  const label = sprite(id);
  const highest = Math.max(...bodies.map((body) => body.position.y)) + 0.9;
  const at = new THREE.Box3().setFromObject(group).getCenter(new THREE.Vector3());
  label.position.set(at.x, highest, at.z);
  group.add(label);
  labels.push(label);
  for (const body of bodies) body.userData = { kind: "instance", id };
  return { id, model, instance, group, bodies, leads: partLeads };
}

function railPost(point, role, name) {
  const group = new THREE.Group();
  const position = toWorld(point);
  const color = role === "ground" ? "#2a2a2a" : "#c93d3d";
  const post = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.9, 12), materials.lead);
  post.position.copy(position).setY(0.45);
  group.add(post);
  const cap = new THREE.Mesh(role === "ground" ? new THREE.ConeGeometry(0.34, 0.36, 4) : new THREE.SphereGeometry(0.26, 16, 12), new THREE.MeshStandardMaterial({ color, roughness: 0.5 }));
  cap.position.copy(position).setY(role === "ground" ? 1.0 : 1.05);
  if (role === "ground") cap.rotation.x = Math.PI;
  group.add(cap);
  if (role !== "ground") {
    const label = sprite(name, 0.6);
    label.position.copy(position).setY(1.65);
    group.add(label);
    labels.push(label);
  }
  return group;
}

// Building the board for the view at `viewPath`.

function clearBoard() {
  board.traverse((object) => {
    object.geometry?.dispose?.();
  });
  board.clear();
  traces = [];
  parts = [];
  leads = [];
  labels = [];
}

function build() {
  clearBoard();
  const drawn = data.layouts[view().component];
  const points = [];
  for (const instance of drawn.instances) points.push(...instance.body, ...Object.values(instance.pins));
  for (const net of drawn.nets) points.push(...net.pins);
  for (const flag of drawn.flags) points.push(flag.point);
  const xs = points.map(([x]) => x);
  const ys = points.map(([, y]) => y);
  const [minX, maxX, minY, maxY] = [Math.min(...xs) - 2.5, Math.max(...xs) + 2.5, Math.min(...ys) - 2.5, Math.max(...ys) + 2.5];
  bounds = { minX, maxX, minY, maxY };
  const width = maxX - minX;
  const depth = maxY - minY;
  const plane = new THREE.Mesh(new THREE.BoxGeometry(width, 0.16, depth), [materials.edge, materials.edge, materials.board, materials.edge, materials.edge, materials.edge]);
  plane.position.set((minX + maxX) / 2, -0.08, -(minY + maxY) / 2);
  plane.receiveShadow = true;
  board.add(plane);
  const grid = new THREE.GridHelper(Math.max(width, depth), Math.round(Math.max(width, depth)), "#2c6a4f", "#245c44");
  grid.position.set((minX + maxX) / 2, 0.002, -(minY + maxY) / 2);
  grid.scale.set(width / Math.max(width, depth), 1, depth / Math.max(width, depth));
  board.add(grid);

  for (const net of drawn.nets) {
    const global = globalNet(net.name);
    const material = new THREE.MeshStandardMaterial({ color: "#c9a14a", metalness: 0.4, roughness: 0.45, emissive: "#000000" });
    const meshes = [];
    for (const [start, end] of net.segments) {
      const a = toWorld(start, 0.02);
      const b = toWorld(end, 0.02);
      const length = a.distanceTo(b);
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(length + 0.2, 0.04, 0.2), material);
      mesh.position.copy(a).lerp(b, 0.5);
      mesh.rotation.y = -Math.atan2(b.z - a.z, b.x - a.x);
      mesh.receiveShadow = true;
      mesh.userData = { kind: "net", local: net.name };
      board.add(mesh);
      meshes.push(mesh);
    }
    for (const point of net.junctions) {
      const dot = new THREE.Mesh(new THREE.CylinderGeometry(0.17, 0.17, 0.07, 20), material);
      dot.position.copy(toWorld(point, 0.035));
      dot.userData = { kind: "net", local: net.name };
      board.add(dot);
      meshes.push(dot);
    }
    if (net.role === "ground" || net.role === "supply") {
      for (const pin of net.pins) board.add(railPost(pin, net.role, net.name));
    } else if (net.labeled) {
      for (const pin of net.pins) {
        const label = sprite(net.name, 0.4);
        label.position.copy(toWorld(pin, 0.7));
        board.add(label);
        labels.push(label);
      }
    }
    traces.push({ net, global, material, meshes });
  }
  for (const flag of drawn.flags) {
    const position = toWorld(flag.point, 0.02);
    const arrow = new THREE.Mesh(new THREE.ConeGeometry(0.28, 0.6, 3), new THREE.MeshStandardMaterial({ color: "#2a78d6", roughness: 0.4 }));
    arrow.rotation.z = flag.side < 0 ? -Math.PI / 2 : Math.PI / 2;
    arrow.position.copy(position).setY(0.2);
    board.add(arrow);
    const label = sprite(flag.name, 0.45);
    label.position.copy(position).setY(0.9);
    board.add(label);
    labels.push(label);
  }
  for (const instance of drawn.instances) {
    const part = makePart(instance);
    board.add(part.group);
    parts.push(part);
    for (const lead of part.leads) {
      const path = pathOf(instance.id);
      const terminal = `${path}.${lead.terminal}`;
      const sparks = [];
      for (let index = 0; index < 3; index += 1) {
        const spark = new THREE.Mesh(new THREE.SphereGeometry(0.075, 10, 8), new THREE.MeshBasicMaterial({ color: "#fff27a" }));
        spark.userData.offset = index / 3;
        board.add(spark);
        sparks.push(spark);
      }
      leads.push({ ...lead, terminal, sparks, phase: Math.random() });
    }
  }
  for (const label of labels) label.visible = showLabels;
  updateCrumbs();
  fit();
  paint();
}

// Each frame's colors, glow and flow.

let colorRange = { low: 0, high: 1, diverging: false };

function computeRange() {
  const values = [];
  for (const series of Object.values(analysis()?.nets ?? {})) for (const value of series) if (value !== null) values.push(value);
  const low = Math.min(0, ...values);
  const high = Math.max(0, ...values);
  const diverging = low < -0.1 * high && high > 1e-12;
  const bound = Math.max(Math.abs(low), Math.abs(high)) || 1;
  const lowest = Math.min(...values);
  colorRange = diverging ? { low: -bound, high: bound, diverging } : { low: Math.min(lowest, high), high: high > lowest ? high : lowest + 1, diverging };
  const stops = diverging ? DIVERGING : VIRIDIS;
  legendRamp.style.background = `linear-gradient(90deg, ${stops.join(", ")})`;
  const unit = analysis()?.kind === "ac" ? "V/V" : "V";
  legendEnds.children[0].textContent = si(colorRange.low, unit);
  legendEnds.children[1].textContent = si(colorRange.high, unit);
  let peak = 0;
  for (const series of Object.values(analysis()?.currents ?? {})) for (const value of series) if (value !== null) peak = Math.max(peak, Math.abs(value));
  currentPeak = peak || 1;
}

let currentPeak = 1;

function colorOf(value) {
  if (value === null || value === undefined) return new THREE.Color("#777777");
  const t = (value - colorRange.low) / (colorRange.high - colorRange.low);
  return ramp(colorRange.diverging ? DIVERGING : VIRIDIS, t);
}

function power(part, at = frame) {
  const path = pathOf(part.id);
  let total = 0;
  for (const [terminal, local] of Object.entries(part.instance.terminals)) {
    const flow = current(`${path}.${terminal}`, at);
    const potential = voltage(globalNet(local), at);
    if (flow !== null && potential !== null) total += flow * potential;
  }
  return total;
}

let powerPeak = 1;

function computePowerPeak() {
  powerPeak = 0;
  const count = analysis()?.axis.length ?? 1;
  for (const part of parts) for (let at = 0; at < count; at += Math.max(1, Math.floor(count / 40))) powerPeak = Math.max(powerPeak, Math.abs(power(part, at)));
  powerPeak = powerPeak || 1;
}

function paint() {
  for (const trace of traces) {
    const logicLevel = level(trace.global);
    let color;
    if (logicLevel !== undefined) color = logicLevel === null ? new THREE.Color("#c9a14a") : new THREE.Color(logicLevel ? "#f2d31b" : "#3b528b");
    else color = colorOf(trace.net.role === "ground" ? 0 : voltage(trace.global));
    trace.material.color.copy(color);
    trace.material.emissive.copy(color).multiplyScalar(0.25);
  }
  for (const part of parts) {
    const glow = Math.min(1, Math.abs(power(part)) / powerPeak);
    for (const body of part.bodies) body.material.emissiveIntensity = glow * 0.9;
  }
  const at = analysis();
  if (at) {
    const value = at.axis[Math.min(frame, at.axis.length - 1)];
    readout.textContent = at.kind === "operating_point" ? "operating point" : `${at.label} ${si(value, at.unit)}`;
  }
  drawPlot();
  if (selection) renderInspector();
}

function flow(seconds) {
  for (const lead of leads) {
    const value = current(lead.terminal);
    const visible = showFlow && value !== null && Math.abs(value) > currentPeak * 1e-4;
    const rate = visible ? 0.15 + 0.85 * Math.max(0, 1 + Math.log10(Math.abs(value) / currentPeak) / 4) : 0;
    lead.phase = (lead.phase + seconds * rate * 0.8) % 1;
    for (const spark of lead.sparks) {
      spark.visible = visible;
      if (!visible) continue;
      let t = (lead.phase + spark.userData.offset) % 1;
      if (value < 0) t = 1 - t;
      spark.position.copy(lead.from).lerp(lead.to, t);
    }
  }
}

// The camera.

function fit() {
  if (!bounds) return;
  const centerX = (bounds.minX + bounds.maxX) / 2;
  const centerZ = -(bounds.minY + bounds.maxY) / 2;
  const span = Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY);
  const distance = span * 0.95 + 4;
  controls.target.set(centerX, 0, centerZ);
  camera.position.set(centerX + distance * 0.25, distance * 0.75, centerZ + distance * 0.75);
  sun.position.set(centerX + span * 0.4, span * 1.2 + 10, centerZ + span * 0.6);
  sun.target.position.set(centerX, 0, centerZ);
  const shadow = sun.shadow.camera;
  shadow.left = shadow.bottom = -span;
  shadow.right = shadow.top = span;
  shadow.far = span * 4 + 40;
  shadow.updateProjectionMatrix();
  controls.update();
}

function topView() {
  if (!bounds) return;
  const centerX = (bounds.minX + bounds.maxX) / 2;
  const centerZ = -(bounds.minY + bounds.maxY) / 2;
  const span = Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY);
  controls.target.set(centerX, 0, centerZ);
  camera.position.set(centerX, span * 1.4 + 6, centerZ + 0.01);
  controls.update();
}

function resize() {
  const width = root.clientWidth;
  const height = root.clientHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  drawPlot();
}

// Hierarchy.

function updateCrumbs() {
  crumbs.replaceChildren();
  const ids = viewPath ? viewPath.split("/") : [];
  crumbs.append(h("button", { class: "link", onclick: () => open("") }, data.top));
  ids.forEach((id, index) => {
    crumbs.append(" / ", h("button", { class: "link", onclick: () => open(ids.slice(0, index + 1).join("/")) }, id));
  });
}

function open(path) {
  if (!(path in data.views)) return;
  viewPath = path;
  selection = null;
  inspector.classList.remove("open");
  build();
  computePowerPeak();
}

function up() {
  if (selection) {
    select(null);
    return;
  }
  if (!viewPath) return;
  open(viewPath.split("/").slice(0, -1).join("/"));
}

// The inspector.

function select(target) {
  selection = target;
  inspector.classList.toggle("open", Boolean(target));
  if (target) renderInspector();
}

function series(name, source) {
  const values = analysis()?.[source]?.[name];
  return values ?? null;
}

function chart(canvasChart, lines, unit) {
  const ratio = Math.min(window.devicePixelRatio, 2);
  const width = canvasChart.clientWidth;
  const height = canvasChart.clientHeight;
  if (!width || !height) return;
  canvasChart.width = width * ratio;
  canvasChart.height = height * ratio;
  const context = canvasChart.getContext("2d");
  context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);
  const at = analysis();
  if (!at || at.axis.length < 2) {
    context.fillStyle = getComputedStyle(root).getPropertyValue("--muted");
    context.fillText("one point: no axis to plot along", 8, 16);
    return;
  }
  const log = at.kind === "ac";
  const xs = at.axis.map((value) => (log ? Math.log10(value) : value));
  const all = lines.flatMap((line) => line.values.filter((value) => value !== null));
  if (!all.length) return;
  let low = Math.min(...all);
  let high = Math.max(...all);
  if (high - low < 1e-15) { high += 1; low -= 1; }
  // The axis is as wide as its longest label, a sign included.
  context.font = "11px ui-monospace, monospace";
  const labels = [low, high].map((tick) => si(tick, unit));
  const widest = Math.max(...labels.map((label) => context.measureText(label).width));
  const pad = { left: Math.ceil(widest) + 8, right: 8, top: 6, bottom: 16 };
  const x = (value) => pad.left + ((value - xs[0]) / (xs[xs.length - 1] - xs[0] || 1)) * (width - pad.left - pad.right);
  const y = (value) => height - pad.bottom - ((value - low) / (high - low)) * (height - pad.top - pad.bottom);
  const style = getComputedStyle(root);
  context.strokeStyle = style.getPropertyValue("--line");
  context.lineWidth = 1;
  context.beginPath();
  for (const tick of [low, (low + high) / 2, high]) {
    context.moveTo(pad.left, y(tick));
    context.lineTo(width - pad.right, y(tick));
  }
  context.stroke();
  context.fillStyle = style.getPropertyValue("--muted");
  context.textAlign = "right";
  [low, high].forEach((tick, index) => context.fillText(labels[index], pad.left - 4, y(tick) + 4));
  const palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"];
  lines.forEach((line, index) => {
    context.strokeStyle = palette[index % palette.length];
    context.lineWidth = 1.6;
    context.beginPath();
    let started = false;
    line.values.forEach((value, point) => {
      if (value === null) { started = false; return; }
      if (!started) { context.moveTo(x(xs[point]), y(value)); started = true; } else context.lineTo(x(xs[point]), y(value));
    });
    context.stroke();
    context.textAlign = "left";
    context.fillStyle = palette[index % palette.length];
    context.fillText(line.name, pad.left + 4 + (index % 3) * 110, pad.top + 10 + Math.floor(index / 3) * 12);
  });
  const cursor = x(xs[Math.min(frame, xs.length - 1)]);
  context.strokeStyle = style.getPropertyValue("--ink");
  context.setLineDash([3, 3]);
  context.beginPath();
  context.moveTo(cursor, pad.top);
  context.lineTo(cursor, height - pad.bottom);
  context.stroke();
  context.setLineDash([]);
}

function drawPlot() {
  const at = analysis();
  if (!at) return;
  const named = Object.keys(at.nets).filter((name) => !/[/.:]/.test(name)).slice(0, 6);
  const lines = named.map((name) => ({ name, values: at.nets[name] }));
  plotHeader.children[1].textContent = at.kind === "ac" ? "magnitude per unit excitation" : "";
  chart(plotCanvas, lines, at.kind === "ac" ? "V/V" : "V");
}

function parameterRows(parameters, prefix = "") {
  const rows = [];
  for (const [key, value] of Object.entries(parameters ?? {})) {
    if (value && typeof value === "object" && !Array.isArray(value)) rows.push(...parameterRows(value, `${prefix}${key}.`));
    else rows.push(h("tr", {}, h("th", {}, `${prefix}${key}`), h("td", { class: "num" }, typeof value === "number" ? si(value) : Array.isArray(value) ? `[${value.length}]` : String(value))));
  }
  return rows;
}

function renderInspector() {
  const at = analysis();
  inspector.replaceChildren(h("button", { class: "close", title: "Close", onclick: () => select(null) }, "×"));
  if (selection.kind === "net") {
    const global = globalNet(selection.local);
    const values = series(global, "nets");
    inspector.append(
      h("h2", {}, selection.local),
      h("div", { class: "kind" }, `net ${global}`),
      h("h3", {}, "Now"),
      h("table", {}, h("tr", {}, h("th", {}, at?.kind === "ac" ? "Magnitude" : "Voltage"), h("td", { class: "num" }, si(voltage(global), at?.kind === "ac" ? "V/V" : "V")))),
    );
    if (values && values.length > 1) {
      const canvasChart = h("canvas", { class: "chart" });
      inspector.append(h("h3", {}, "Over the analysis"), canvasChart);
      requestAnimationFrame(() => chart(canvasChart, [{ name: global, values }], "V"));
    }
    return;
  }
  const part = parts.find((candidate) => candidate.id === selection.id);
  if (!part) return;
  const path = pathOf(part.id);
  const instance = part.instance;
  inspector.append(
    h("h2", {}, part.id),
    h("div", { class: "kind" }, `${instance.device ? instance.device : `component ${instance.component}`} · ${path}`),
  );
  if (instance.component) inspector.append(h("p", {}, h("button", { onclick: () => open(path) }, "Open this component")));
  const unitV = at?.kind === "ac" ? "V/V" : "V";
  const unitI = at?.kind === "ac" ? "A/V" : "A";
  const rows = Object.entries(instance.terminals).map(([terminal, local]) => {
    const name = `${path}.${terminal}`;
    const charges = series(name, "charges");
    return h("tr", {},
      h("th", {}, terminal),
      h("td", {}, local),
      h("td", { class: "num" }, si(voltage(globalNet(local)), unitV)),
      h("td", { class: "num" }, si(current(name), unitI)),
      h("td", { class: "num" }, charges ? si(charges[Math.min(frame, charges.length - 1)], "C") : "—"));
  });
  inspector.append(h("h3", {}, "Terminals"), h("table", {}, h("tr", {}, h("th", {}, "Terminal"), h("th", {}, "Net"), h("th", {}, "Voltage"), h("th", {}, "Current in"), h("th", {}, "Charge")), rows));
  if (at?.kind !== "ac") inspector.append(h("table", {}, h("tr", {}, h("th", {}, "Power taken"), h("td", { class: "num" }, si(power(part), "W")))));
  const variables = at?.variables?.[path];
  if (variables) {
    inspector.append(h("h3", {}, "Variables"), h("table", {}, Object.entries(variables).map(([name, values]) => h("tr", {}, h("th", {}, name.replaceAll("_", " ")), h("td", { class: "num" }, si(values[Math.min(frame, values.length - 1)]))))));
  }
  const terminals = Object.keys(instance.terminals);
  const matrix = terminals.map((row) => terminals.map((column) => at?.conductances?.[`${path}.${row}|${path}.${column}`]));
  if (matrix.some((row) => row.some((values) => values))) {
    inspector.append(
      h("h3", {}, "Small-signal conductances, ∂I(row)/∂V(column)"),
      h("table", { class: "matrix" },
        h("tr", {}, h("th", {}, ""), terminals.map((column) => h("th", {}, column))),
        terminals.map((row, index) => h("tr", {}, h("th", {}, row), matrix[index].map((values) => h("td", {}, values ? si(values[Math.min(frame, values.length - 1)], "S") : "0"))))),
    );
  }
  const lines = terminals.map((terminal) => ({ name: terminal, values: series(`${path}.${terminal}`, "currents") })).filter((line) => line.values);
  if (lines.length && at && at.axis.length > 1) {
    const canvasChart = h("canvas", { class: "chart" });
    inspector.append(h("h3", {}, "Terminal currents over the analysis"), canvasChart);
    requestAnimationFrame(() => chart(canvasChart, lines, unitI));
  }
  if (instance.parameters && Object.keys(instance.parameters).length) {
    inspector.append(h("h3", {}, "Parameters, in SI base units"), h("table", {}, parameterRows(instance.parameters)));
  }
}

// Picking.

function pick(event) {
  const rectangle = canvas.getBoundingClientRect();
  pointer.set(((event.clientX - rectangle.left) / rectangle.width) * 2 - 1, -((event.clientY - rectangle.top) / rectangle.height) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObjects(board.children, true).find((found) => found.object.userData?.kind);
  return hit?.object.userData ?? null;
}

let pressed = null;
canvas.addEventListener("pointerdown", (event) => { pressed = { x: event.clientX, y: event.clientY }; });
canvas.addEventListener("pointerup", (event) => {
  if (!pressed || Math.hypot(event.clientX - pressed.x, event.clientY - pressed.y) > 5) return;
  select(pick(event));
});
canvas.addEventListener("dblclick", (event) => {
  const target = pick(event);
  if (target?.kind === "instance") {
    const part = parts.find((candidate) => candidate.id === target.id);
    if (part?.instance.component) open(pathOf(target.id));
  }
});
canvas.addEventListener("pointermove", (event) => {
  const target = pick(event);
  if (!target) { tooltip.style.display = "none"; canvas.style.cursor = ""; return; }
  canvas.style.cursor = "pointer";
  const at = analysis();
  let text;
  if (target.kind === "net") text = `${target.local}  ${si(voltage(globalNet(target.local)), at?.kind === "ac" ? "V/V" : "V")}`;
  else {
    const part = parts.find((candidate) => candidate.id === target.id);
    text = `${target.id}  ${part?.instance.device ?? part?.instance.component ?? ""}`;
  }
  tooltip.textContent = text;
  tooltip.style.display = "block";
  tooltip.style.left = `${event.clientX + 14}px`;
  tooltip.style.top = `${event.clientY + 14}px`;
});
canvas.addEventListener("pointerleave", () => { tooltip.style.display = "none"; });

// The timeline.

function setAnalysis(index) {
  analysisIndex = index;
  const at = analysis();
  const count = at ? at.axis.length : 1;
  slider.max = String(count - 1);
  frame = Math.min(frame, count - 1);
  slider.value = String(frame);
  const single = count <= 1;
  slider.disabled = single;
  play.disabled = single;
  if (single) setPlaying(false);
  computeRange();
  computePowerPeak();
  paint();
}

function setFrame(index) {
  const count = analysis()?.axis.length ?? 1;
  frame = ((index % count) + count) % count;
  slider.value = String(frame);
  paint();
}

function setPlaying(on) {
  playing = on && (analysis()?.axis.length ?? 1) > 1;
  play.textContent = playing ? "❚❚" : "▶";
  play.classList.toggle("on", playing);
}

data.analyses.forEach((at, index) => {
  const kinds = { operating_point: "operating point", dc_sweep: "DC sweep", ac: "frequency response", transient: "transient" };
  analysisSelect.append(h("option", { value: String(index) }, `${at.name} · ${kinds[at.kind] ?? at.kind}`));
});
analysisSelect.addEventListener("change", () => setAnalysis(Number(analysisSelect.value)));
slider.addEventListener("input", () => { setPlaying(false); setFrame(Number(slider.value)); });
play.addEventListener("click", () => setPlaying(!playing));

// The tools.

function toggle(label, on, run, hint) {
  const button = h("button", { title: hint, class: on ? "on" : "" }, label);
  button.addEventListener("click", () => button.classList.toggle("on", run()));
  return button;
}

const screenshot = () => {
  renderer.render(world, camera);
  const link = h("a", { href: canvas.toDataURL("image/png"), download: `${data.title.replaceAll(" ", "-")}.png` });
  link.click();
};

tools.append(
  h("button", { title: "Up a level (U)", onclick: () => up() }, "Up"),
  h("button", { title: "Fit the view (F)", onclick: () => fit() }, "Fit"),
  h("button", { title: "View from the top (T)", onclick: () => topView() }, "Top"),
  toggle("Labels", true, () => { showLabels = !showLabels; for (const label of labels) label.visible = showLabels; return showLabels; }, "Labels (L)"),
  toggle("Flow", true, () => { showFlow = !showFlow; return showFlow; }, "Current flow"),
  toggle("Plot", true, () => { plotPanel.classList.toggle("open"); drawPlot(); return plotPanel.classList.contains("open"); }, "The named nets' voltages"),
  h("button", { title: "Save a PNG", onclick: screenshot }, "PNG"),
  h("button", { title: "Full screen", onclick: () => (document.fullscreenElement ? document.exitFullscreen() : root.requestFullscreen?.()) }, "⛶"),
  h("button", { title: "Help (?)", onclick: () => help.classList.toggle("open") }, "?"),
);

window.addEventListener("keydown", (event) => {
  if (event.target.closest?.("input, select, textarea")) return;
  const keys = {
    " ": () => setPlaying(!playing),
    ArrowRight: () => { setPlaying(false); setFrame(frame + 1); },
    ArrowLeft: () => { setPlaying(false); setFrame(frame - 1); },
    Home: () => setFrame(0),
    End: () => setFrame((analysis()?.axis.length ?? 1) - 1),
    f: fit,
    t: topView,
    l: () => tools.children[3].click(),
    u: up,
    Escape: up,
    "?": () => help.classList.toggle("open"),
  };
  const run = keys[event.key];
  if (run) { event.preventDefault(); run(); }
});

// The loop.

const clock = new THREE.Clock();
let elapsed = 0;
function loop() {
  const seconds = Math.min(0.1, clock.getDelta());
  if (playing) {
    elapsed += seconds * Number(speed.value);
    const count = analysis()?.axis.length ?? 1;
    const step = 6 / count;
    if (elapsed >= step) {
      setFrame(frame + Math.max(1, Math.floor(elapsed / step)));
      elapsed = 0;
    }
  }
  flow(seconds);
  controls.update();
  renderer.render(world, camera);
  requestAnimationFrame(loop);
}

// The page's address may choose the view, the analysis and the point, as in
// `#view=X1&analysis=step&point=120&paused`.
const wanted = new URLSearchParams(location.hash.slice(1));
if (data.views[wanted.get("view") ?? ""] !== undefined) viewPath = wanted.get("view") ?? "";
window.addEventListener("resize", resize);
resize();
build();
const named = data.analyses.findIndex((at) => at.name === wanted.get("analysis"));
setAnalysis(named >= 0 ? named : Math.max(0, data.analyses.findIndex((at) => at.kind === "transient")));
analysisSelect.value = String(analysisIndex);
if (wanted.has("point")) setFrame(Number(wanted.get("point")));
setPlaying(analysis()?.kind === "transient" && !wanted.has("paused"));
root.dataset.state = "ready";
loop();
