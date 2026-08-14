import Graph from "graphology";
import {
  ChevronLeft,
  ChevronRight,
  Network,
  PanelRight,
  Route,
  X,
  createElement as createLucideElement,
  createIcons,
} from "lucide";
import type { IconNode } from "lucide";
import Sigma from "sigma";

import {
  applyScopeLayout,
  applyCompactLayout,
  buildGraph,
  inMode,
  inScope,
  isModelState,
  shortestPath,
} from "./graph";
import {
  classifySink,
  colorForRegion,
  describeBehavior,
  displayLabel as presentDisplayLabel,
  groupTitle,
  modeSentence,
  pathLabel as presentPathLabel,
  reachabilitySentence as presentReachabilitySentence,
  readableInterface,
} from "./presentation";
import { sourceDescription } from "./provenance";
import type {
  EvidenceEdge,
  GraphDataset,
  ModeFilter,
  Region,
  Scope,
} from "./types";
import { activeView } from "./views";
import "./style.css";

const MODE: ModeFilter = "union";
const SCOPE: Scope = "all";
const view = activeView(window.location.search);

const app = document.querySelector<HTMLDivElement>("#app")!;

app.innerHTML = `
  <header class="topbar">
    <div class="brand">
      <span class="brand-mark"><i data-lucide="network"></i></span>
      <span>
        <strong>Evidence Paths</strong>
        <small>${view.subtitle}</small>
      </span>
    </div>
    <div class="topbar-actions">
      <nav class="view-tabs" aria-label="Monitor graph">
        <a href="?view=safe" ${view.id === "safe" ? 'aria-current="page"' : ""}>SAFE</a>
        <a href="?view=qwen" ${view.id === "qwen" ? 'aria-current="page"' : ""}>Qwen</a>
      </nav>
      <button class="icon-button mobile-inspector-button" id="show-inspector" title="Show region details" aria-label="Show region details">
        <i data-lucide="panel-right"></i>
      </button>
    </div>
  </header>

  <main class="workspace">
    <section class="graph-shell" aria-label="Interactive evidence graph">
      <div id="sigma-container"></div>
      <svg class="aggregate-edge-layer" id="aggregate-edge-layer" aria-hidden="true"></svg>
      <div class="group-label-layer" id="group-label-layer" aria-hidden="true"></div>
      <nav class="path-navigation" id="path-navigation" aria-label="Path navigation" hidden>
        <button class="path-return" id="return-to-graph" title="Return to full graph" aria-label="Return to full graph">
          <i data-lucide="network"></i>
          <span>Full graph</span>
        </button>
        <div class="path-current" aria-live="polite">
          <span id="path-position">—</span>
          <strong id="path-current-label">—</strong>
        </div>
        <button class="path-step" id="previous-path-step" title="Previous region" aria-label="Previous region">
          <i data-lucide="chevron-left"></i>
        </button>
        <button class="path-step" id="next-path-step" title="Next region" aria-label="Next region">
          <i data-lucide="chevron-right"></i>
        </button>
      </nav>
    </section>

    <aside class="inspector" id="inspector" aria-label="Region details">
      <div class="inspector-header">
        <span>Region details</span>
        <button class="mini-icon-button mobile-close-inspector" id="close-inspector" title="Close details" aria-label="Close details">
          <i data-lucide="x"></i>
        </button>
      </div>
      <div id="inspector-content"></div>
    </aside>
  </main>
`;

createIcons({
  icons: { ChevronLeft, ChevronRight, Network, PanelRight, X },
  attrs: { "stroke-width": 1.8 },
});

function element<T extends Element>(selector: string): T {
  const found = document.querySelector<T>(selector);
  if (!found) throw new Error(`missing viewer element: ${selector}`);
  return found;
}

const sigmaContainer = element<HTMLDivElement>("#sigma-container");
const aggregateEdgeLayer = element<SVGSVGElement>("#aggregate-edge-layer");
const groupLabelLayer = element<HTMLDivElement>("#group-label-layer");
const inspector = element<HTMLElement>("#inspector");
const inspectorContent = element<HTMLDivElement>("#inspector-content");
const pathNavigation = element<HTMLElement>("#path-navigation");
const pathPosition = element<HTMLElement>("#path-position");
const pathCurrentLabel = element<HTMLElement>("#path-current-label");
const previousPathButton = element<HTMLButtonElement>("#previous-path-step");
const nextPathButton = element<HTMLButtonElement>("#next-path-step");
const compactPathQuery = window.matchMedia("(max-width: 720px)");

let dataset: GraphDataset;
let graph: Graph;
let renderer: Sigma;
let selectedNode: string | null = null;
let pathFocus: Set<string> | null = null;
let pathOrder: string[] = [];
let pathIndex = 0;
let hoveredNode: string | null = null;
let regionNameCounts = new Map<string, number>();

interface RegionGroup {
  name: string;
  nodes: string[];
  stateCount: number;
  runtimeCount: number;
  aggregatedLinkCount: number;
  label: HTMLDivElement;
}

interface AggregateConnection {
  groupName: string;
  target: string;
  sources: string[];
  edges: string[];
  path: SVGPathElement;
}

let regionGroups: RegionGroup[] = [];
let aggregateConnections: AggregateConnection[] = [];
let aggregateByEdge = new Map<string, AggregateConnection>();

function regionFor(node: string): Region {
  return graph.getNodeAttribute(node, "region") as Region;
}

function sinkClass(region: Region): "both" | "monitor" | "outcome" | "neither" | "state" {
  return classifySink(region, view);
}

function nodeColor(region: Region): string {
  return colorForRegion(region, view);
}

function isVisible(region: Region): boolean {
  return inMode(region, MODE) && inScope(region, SCOPE);
}

function displayLabel(region: Region): string {
  return presentDisplayLabel(region, view, regionNameCounts.get(region.name) ?? 0);
}

function pathLabel(region: Region): string {
  return presentPathLabel(region, compactPathQuery.matches);
}

function refreshGraph(): void {
  const compactGraph = dataset.totals.regions <= 12;
  renderer.setSetting("nodeReducer", (node, data) => {
    const region = data.region as Region;
    const onPath = pathFocus?.has(node) ?? false;
    const hidden = !isVisible(region) || (pathFocus !== null && !onPath);
    const selected = node === selectedNode;
    const hovered = node === hoveredNode;
    const degree = Math.min(graph.degree(node), 12);
    const baseSize = compactGraph
      ? 6.5 + degree * 0.15
      : isModelState(region)
        ? 1.7 + degree * 0.05
        : 4.8 + degree * 0.12;
    const showLabel = compactGraph || selected || hovered || onPath;
    return {
      ...data,
      hidden,
      color: selected ? "#14282f" : nodeColor(region),
      label: showLabel ? (onPath ? pathLabel(region) : displayLabel(region)) : "",
      size: selected ? 9.5 : hovered ? 7.5 : onPath ? Math.max(baseSize, 6.3) : baseSize,
      forceLabel: showLabel,
      zIndex: selected ? 4 : hovered ? 3 : onPath ? 2 : isModelState(region) ? 0 : 1,
    };
  });

  renderer.setSetting("edgeReducer", (edgeKey, data) => {
    const edge = data.edge as EvidenceEdge;
    const source = graph.source(edgeKey);
    const target = graph.target(edgeKey);
    const sourceRegion = regionFor(source);
    const targetRegion = regionFor(target);
    const onPath = pathFocus?.has(source) && pathFocus.has(target);
    const aggregate = aggregateByEdge.get(edgeKey);
    const inspectedState = hoveredNode ?? selectedNode;
    const incidentToInspection = !inspectedState || source === inspectedState || target === inspectedState;
    const revealConstituent = Boolean(
      aggregate && inspectedState === source && isModelState(sourceRegion),
    );
    const hidden =
      !inMode(edge, MODE) ||
      !isVisible(sourceRegion) ||
      !isVisible(targetRegion) ||
      (pathFocus !== null && !onPath) ||
      (pathFocus === null && !incidentToInspection) ||
      (pathFocus === null && aggregate !== undefined && !revealConstituent);
    return {
      ...data,
      hidden,
      color: onPath ? "rgba(30, 55, 66, 0.82)" : revealConstituent ? "rgba(44, 101, 133, 0.46)" : "rgba(93, 109, 108, 0.14)",
      size: onPath ? 2 : revealConstituent ? 1.15 : 0.55,
      zIndex: onPath || revealConstituent ? 2 : 0,
    };
  });

  renderer.refresh();
}

function buildRegionGroups(): void {
  const nodesByName = new Map<string, string[]>();
  graph.forEachNode((node) => {
    const name = regionFor(node).name;
    const nodes = nodesByName.get(name) ?? [];
    nodes.push(node);
    nodesByName.set(name, nodes);
  });

  // Names come from passing, provenance-backed graph annotations. The viewer
  // groups exact matches only; it does not introduce a new semantic hierarchy.
  regionGroups = [...nodesByName]
    .filter(([, nodes]) => nodes.length > 1)
    .map(([name, nodes]) => {
      const stateCount = nodes.filter((node) => isModelState(regionFor(node))).length;
      const runtimeCount = nodes.length - stateCount;
      const aggregatedLinkCount = aggregateConnections
        .filter((connection) => connection.groupName === name)
        .reduce((total, connection) => total + connection.edges.length, 0);
      const label = document.createElement("div");
      label.className = "group-label";
      label.title = "Counts describe recorded graph structure, not measured risk";
      const nameElement = document.createElement("strong");
      nameElement.textContent = groupTitle(name);
      const count = document.createElement("span");
      count.textContent = stateCount && runtimeCount
        ? `${stateCount} state · ${runtimeCount} runtime${aggregatedLinkCount ? ` · ${aggregatedLinkCount.toLocaleString()} links` : ""}`
        : `${nodes.length} regions`;
      label.append(nameElement, count);
      groupLabelLayer.append(label);
      return { name, nodes, stateCount, runtimeCount, aggregatedLinkCount, label };
    })
    .sort((left, right) => right.nodes.length - left.nodes.length || left.name.localeCompare(right.name));
}

function buildAggregateConnections(): void {
  const candidates = new Map<string, { groupName: string; target: string; sources: Set<string>; edges: string[] }>();
  graph.forEachEdge((edgeKey) => {
    const source = graph.source(edgeKey);
    const target = graph.target(edgeKey);
    const sourceRegion = regionFor(source);
    const targetRegion = regionFor(target);
    if (!isModelState(sourceRegion) || isModelState(targetRegion) || sourceRegion.name !== targetRegion.name) return;
    const key = `${sourceRegion.name}:${target}`;
    const candidate = candidates.get(key) ?? {
      groupName: sourceRegion.name,
      target,
      sources: new Set<string>(),
      edges: [],
    };
    candidate.sources.add(source);
    candidate.edges.push(edgeKey);
    candidates.set(key, candidate);
  });

  // Eight parallel edges already exceed what can be resolved at overview scale.
  // This affects display only; every constituent edge remains in the graph.
  aggregateConnections = [...candidates.values()]
    .filter((candidate) => candidate.edges.length >= 8)
    .map((candidate) => {
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.classList.add("aggregate-edge");
      aggregateEdgeLayer.append(path);
      const connection = { ...candidate, sources: [...candidate.sources], path };
      for (const edge of connection.edges) aggregateByEdge.set(edge, connection);
      return connection;
    });
}

function updateAggregateConnections(): void {
  const show = pathFocus === null;
  aggregateEdgeLayer.style.display = show ? "block" : "none";
  if (!show) return;

  const inspectedNode = hoveredNode ?? selectedNode;
  const inspectedRegion = inspectedNode ? regionFor(inspectedNode) : null;
  for (const connection of aggregateConnections) {
    const relevantToInspection = !inspectedNode || (
      !isModelState(inspectedRegion!) && connection.target === inspectedNode
    );
    if (!relevantToInspection) {
      connection.path.style.display = "none";
      continue;
    }
    const target = renderer.graphToViewport({
      x: graph.getNodeAttribute(connection.target, "x") as number,
      y: graph.getNodeAttribute(connection.target, "y") as number,
    });
    let nearest: { x: number; y: number; distance: number } | null = null;
    for (const source of connection.sources) {
      const point = renderer.graphToViewport({
        x: graph.getNodeAttribute(source, "x") as number,
        y: graph.getNodeAttribute(source, "y") as number,
      });
      const distance = Math.hypot(point.x - target.x, point.y - target.y);
      if (!nearest || distance < nearest.distance) nearest = { ...point, distance };
    }
    if (!nearest) {
      connection.path.style.display = "none";
      continue;
    }

    const dx = target.x - nearest.x;
    const dy = target.y - nearest.y;
    const length = Math.max(Math.hypot(dx, dy), 1);
    const sourceInset = 3;
    const targetInset = 8;
    const startX = nearest.x + (dx / length) * sourceInset;
    const startY = nearest.y + (dy / length) * sourceInset;
    const endX = target.x - (dx / length) * targetInset;
    const endY = target.y - (dy / length) * targetInset;
    connection.path.style.display = "block";
    connection.path.setAttribute("d", `M ${startX} ${startY} L ${endX} ${endY}`);
    connection.path.classList.toggle("active", hoveredNode === connection.target || selectedNode === connection.target);
  }
}

function rectanglesOverlap(
  left: { left: number; right: number; top: number; bottom: number },
  right: { left: number; right: number; top: number; bottom: number },
): boolean {
  return left.left < right.right && left.right > right.left && left.top < right.bottom && left.bottom > right.top;
}

function updateRegionGroups(): void {
  const cameraRatio = renderer.getCamera().getState().ratio;
  const showGroups = pathFocus === null && cameraRatio >= 0.38;
  if (!showGroups) {
    for (const group of regionGroups) group.label.hidden = true;
    return;
  }

  const placed: Array<{ left: number; right: number; top: number; bottom: number }> = [];
  const dimensions = renderer.getDimensions();
  for (const group of regionGroups) {
    let x = 0;
    let y = 0;
    for (const node of group.nodes) {
      const position = renderer.graphToViewport({
        x: graph.getNodeAttribute(node, "x") as number,
        y: graph.getNodeAttribute(node, "y") as number,
      });
      x += position.x;
      y += position.y;
    }
    x /= group.nodes.length;
    y /= group.nodes.length;

    group.label.hidden = false;
    group.label.style.left = `${x}px`;
    group.label.style.top = `${y}px`;
    const width = group.label.offsetWidth;
    const height = group.label.offsetHeight;
    const padding = 8;
    const bounds = {
      left: x - width / 2 - padding,
      right: x + width / 2 + padding,
      top: y - height / 2 - padding,
      bottom: y + height / 2 + padding,
    };
    const outside = bounds.left < 8 || bounds.right > dimensions.width - 8 || bounds.top < 8 || bounds.bottom > dimensions.height - 8;
    const collides = placed.some((existing) => rectanglesOverlap(bounds, existing));
    group.label.hidden = outside || collides;
    if (!group.label.hidden) placed.push(bounds);
  }
}

function focusNode(node: string): void {
  if (pathFocus && !pathFocus.has(node)) return;
  selectedNode = node;
  if (pathFocus) {
    const index = pathOrder.indexOf(node);
    if (index >= 0) pathIndex = index;
    updatePathNavigation();
  }
  refreshGraph();
  renderInspector(regionFor(node));
  if (!pathFocus) {
    const display = renderer.getNodeDisplayData(node);
    if (display) {
      renderer.getCamera().animate({ x: display.x, y: display.y, ratio: 0.13 }, { duration: 350 });
    }
  }
  if (window.innerWidth < 760) inspector.classList.add("open");
}

function createIcon(icon: IconNode): SVGElement {
  return createLucideElement(icon, { width: 16, height: 16, "stroke-width": 1.8 });
}

function textSpan(value: string, className = ""): HTMLSpanElement {
  const span = document.createElement("span");
  span.className = className;
  span.textContent = value;
  return span;
}

function reachabilitySentence(region: Region): string {
  return presentReachabilitySentence(region, view);
}

function detailRow(term: string, value: string): HTMLElement {
  const row = document.createElement("div");
  row.className = "detail-row";
  row.append(textSpan(term), textSpan(value, "detail-value"));
  return row;
}

function renderInspector(region?: Region): void {
  inspectorContent.replaceChildren();

  if (!region) {
    const campaign = dataset.traceCampaign;
    const empty = document.createElement("div");
    empty.className = "empty-inspector";
    empty.innerHTML = `
      <span class="empty-mark"><i data-lucide="route"></i></span>
      <span class="eyebrow neutral">Audited graph</span>
      <h2>Follow one evidence path</h2>
      <p>${view.emptyDescription}</p>
      <div class="dataset-summary">
        <strong>${dataset.totals.regions}</strong> regions
        <span></span>
        <strong>${dataset.totals.edges}</strong> directed links
      </div>
      ${campaign ? `<p class="campaign-note"><strong>${campaign.totalQueries}</strong> audited queries · <strong>${campaign.coverage.regions.novelHoldouts}</strong> new structural regions in ${campaign.holdoutQueries} holdouts</p>` : ""}
    `;
    inspectorContent.append(empty);
    createIcons({ icons: { Route }, root: empty, attrs: { "stroke-width": 1.7 } });
    return;
  }

  const title = document.createElement("div");
  title.className = "inspector-title";
  const eyebrow = document.createElement("span");
  eyebrow.className = `eyebrow ${sinkClass(region)}`;
  eyebrow.textContent = isModelState(region)
    ? "Model state"
    : "Runtime evidence";
  const heading = document.createElement("h2");
  heading.textContent = displayLabel(region);
  const summary = document.createElement("p");
  summary.textContent = reachabilitySentence(region);
  title.append(eyebrow, heading, summary);
  inspectorContent.append(title);

  const target = dataset.regions.find((candidate) => candidate.name === view.pathTargetRegion);
  const canTrace = Boolean(target && target.id !== region.id && region.reachableSinks.includes(view.pathTargetSink));
  if (canTrace && target && pathFocus === null) {
    const action = document.createElement("button");
    action.className = "primary-action";
    action.append(createIcon(Route), textSpan(`Show path to ${view.pathTargetLabel}`));
    action.addEventListener("click", () => showPath(region, target));
    inspectorContent.append(action);
  }

  const details = document.createElement("section");
  details.className = "inspector-section details";
  const incomingAggregate = aggregateConnections.find((connection) => connection.target === region.id);
  details.append(
    detailRow("Region ID", region.id),
    detailRow("Evidence", readableInterface(region)),
    detailRow("Duration", region.lifetime === "temporal" ? "Across the rollout" : "One policy step"),
    detailRow("Runs", modeSentence(region).replace(/^Observed in /, "").replace(/\.$/, "")),
  );
  if (incomingAggregate) {
    details.append(detailRow("Connections", `${incomingAggregate.edges.length} registered-state links`));
  }
  if (region.disposition) {
    details.append(detailRow("Note", describeBehavior(region.disposition)));
  }
  inspectorContent.append(details);

  const provenance = document.createElement("section");
  provenance.className = "inspector-section";
  const provenanceHeading = document.createElement("h3");
  provenanceHeading.textContent = "Why this region is here";
  provenance.append(provenanceHeading);
  const list = document.createElement("div");
  list.className = "source-list";
  for (const basis of region.basis) {
    const source = sourceDescription(basis, region);
    const item = document.createElement("div");
    item.className = "source-item";
    item.append(textSpan(source.kind, "source-kind"), textSpan(source.text, "source-text"));
    list.append(item);
  }
  provenance.append(list);
  inspectorContent.append(provenance);
}

function showPath(source: Region, outcome: Region): void {
  const result = shortestPath(graph, source.id, outcome.id, MODE, SCOPE);
  if (!result) {
    return;
  }
  pathOrder = [...result].reverse();
  pathFocus = new Set(pathOrder);
  pathIndex = Math.max(0, pathOrder.indexOf(source.id));
  selectedNode = source.id;
  layoutPath(pathOrder);
  updatePathNavigation();
  renderInspector(source);
  refreshGraph();
  renderer.getCamera().animatedReset({ duration: 350 });
}

function layoutPath(path: readonly string[]): void {
  const xValues: number[] = [];
  const yValues: number[] = [];
  graph.forEachNode((_node, attributes) => {
    xValues.push(attributes.allX as number);
    yValues.push(attributes.allY as number);
  });
  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  const minY = Math.min(...yValues);
  const maxY = Math.max(...yValues);
  const centerX = minX + (maxX - minX) * 0.34;
  const xOffset = (maxX - minX) * 0.018;
  const top = minY + (maxY - minY) * 0.12;
  const bottom = maxY - (maxY - minY) * 0.12;
  path.forEach((node, index) => {
    const progress = path.length === 1 ? 0.5 : index / (path.length - 1);
    graph.mergeNodeAttributes(node, {
      x: centerX + (index % 2 === 0 ? -xOffset : xOffset),
      y: top + (bottom - top) * progress,
    });
  });
}

function updatePathNavigation(): void {
  const active = pathOrder.length > 0;
  pathNavigation.hidden = !active;
  if (!active) return;
  const node = pathOrder[pathIndex];
  pathPosition.textContent = `${pathIndex + 1} of ${pathOrder.length}`;
  pathCurrentLabel.textContent = displayLabel(regionFor(node));
  previousPathButton.disabled = pathIndex === 0;
  nextPathButton.disabled = pathIndex === pathOrder.length - 1;
}

function selectPathStep(index: number): void {
  if (index < 0 || index >= pathOrder.length) return;
  pathIndex = index;
  selectedNode = pathOrder[index];
  updatePathNavigation();
  renderInspector(regionFor(selectedNode));
  refreshGraph();
}

function returnToGraph(): void {
  pathFocus = null;
  pathOrder = [];
  pathIndex = 0;
  selectedNode = null;
  applyScopeLayout(graph, SCOPE);
  updatePathNavigation();
  renderInspector();
  refreshGraph();
  renderer.getCamera().animatedReset({ duration: 280 });
}

previousPathButton.addEventListener("click", () => selectPathStep(pathIndex - 1));
nextPathButton.addEventListener("click", () => selectPathStep(pathIndex + 1));
element("#return-to-graph").addEventListener("click", returnToGraph);
document.addEventListener("keydown", (event) => {
  if (!pathFocus) return;
  if (event.key === "ArrowLeft") selectPathStep(pathIndex - 1);
  if (event.key === "ArrowRight") selectPathStep(pathIndex + 1);
  if (event.key === "Escape") returnToGraph();
});
element("#show-inspector").addEventListener("click", () => inspector.classList.add("open"));
element("#close-inspector").addEventListener("click", () => inspector.classList.remove("open"));

async function start(): Promise<void> {
  document.title = `${view.label} evidence paths`;
  const response = await fetch(`${import.meta.env.BASE_URL}${view.datasetPath}`);
  const contentType = response.headers.get("content-type") ?? "";
  if (view.id === "qwen" && (!response.ok || !contentType.includes("application/json"))) {
    renderUnpublishedQwen();
    return;
  }
  if (!response.ok) throw new Error(`cannot load graph dataset: ${response.status}`);
  dataset = (await response.json()) as GraphDataset;
  if (dataset.view && dataset.view !== view.id) {
    throw new Error(`graph dataset is for ${dataset.view}, not ${view.id}`);
  }
  regionNameCounts = dataset.regions.reduce((counts, region) => {
    counts.set(region.name, (counts.get(region.name) ?? 0) + 1);
    return counts;
  }, new Map<string, number>());
  graph = buildGraph(dataset);
  if (dataset.totals.regions <= 12) applyCompactLayout(graph);
  applyScopeLayout(graph, SCOPE);
  renderer = new Sigma(graph, sigmaContainer, {
    allowInvalidContainer: false,
    defaultEdgeColor: "rgba(93, 109, 108, 0.10)",
    defaultEdgeType: "line",
    labelColor: { color: "#17272d" },
    labelDensity: 0.18,
    labelFont: "IBM Plex Sans, Inter, sans-serif",
    labelGridCellSize: 110,
    labelRenderedSizeThreshold: 5.5,
    labelSize: 12,
    renderEdgeLabels: false,
    stagePadding: 52,
    zIndex: true,
  });
  renderer.on("clickNode", ({ node }) => focusNode(node));
  renderer.on("afterRender", () => {
    updateAggregateConnections();
    updateRegionGroups();
  });
  renderer.on("enterNode", ({ node }) => {
    hoveredNode = node;
    refreshGraph();
  });
  renderer.on("leaveNode", () => {
    hoveredNode = null;
    refreshGraph();
  });
  renderer.on("clickStage", () => {
    if (pathFocus) return;
    selectedNode = null;
    refreshGraph();
    renderInspector();
  });
  compactPathQuery.addEventListener("change", () => {
    if (pathFocus) refreshGraph();
  });

  buildAggregateConnections();
  buildRegionGroups();
  renderInspector();
  refreshGraph();
  await renderer.getCamera().animatedReset({ duration: 250 });
}

function renderUnpublishedQwen(): void {
  sigmaContainer.innerHTML = `
    <div class="unpublished-state">
      <span>Qwen</span>
      <strong>Evidence graph pending</strong>
      <p>The scoring campaign must finish before its graph can be built and audited.</p>
    </div>
  `;
  inspectorContent.innerHTML = `
    <div class="empty-inspector">
      <span class="eyebrow neutral">Not yet published</span>
      <h2>Qwen evidence paths</h2>
      <p>No graph is shown until the recorded Qwen artifact passes the same provenance and structure checks as the SAFE graph.</p>
    </div>
  `;
}

start().catch((error: unknown) => {
  console.error(error);
  sigmaContainer.innerHTML = `<div class="fatal-error">Graph failed to load.</div>`;
});
