import Graph from "graphology";
import forceAtlas2 from "graphology-layout-forceatlas2";

import type {
  EvidenceEdge,
  GraphDataset,
  ModeFilter,
  Region,
  Scope,
} from "./types";

export const MODEL_STATE_INTERFACE = "registered_model_state";

export function isModelState(region: Region): boolean {
  return region.faultInterface === MODEL_STATE_INTERFACE;
}

export function inMode(item: Region | EvidenceEdge, mode: ModeFilter): boolean {
  return mode === "union" || item.modes.includes(mode);
}

export function inScope(region: Region, scope: Scope): boolean {
  return scope === "all" || !isModelState(region);
}

export function visibleRegion(
  region: Region,
  mode: ModeFilter,
  scope: Scope,
  focus: ReadonlySet<string> | null,
): boolean {
  return inMode(region, mode) && inScope(region, scope) && (!focus || focus.has(region.id));
}

function hashUnit(value: string, offset: number): number {
  let hash = 2166136261 ^ offset;
  for (const character of value) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return ((hash >>> 0) / 0xffffffff) * 2 - 1;
}

export function buildGraph(dataset: GraphDataset): Graph {
  const graph = new Graph({ multi: true, type: "directed", allowSelfLoops: true });
  for (const region of dataset.regions) {
    graph.addNode(region.id, {
      x: hashUnit(region.id, 11),
      y: hashUnit(region.id, 37),
      label: region.semanticKey,
      region,
    });
  }
  for (const edge of dataset.edges) {
    graph.addDirectedEdgeWithKey(edge.id, edge.source, edge.target, { edge });
  }

  const settings = forceAtlas2.inferSettings(graph);
  forceAtlas2.assign(graph, {
    iterations: 180,
    settings: {
      ...settings,
      adjustSizes: false,
      barnesHutOptimize: true,
      gravity: 0.8,
      scalingRatio: 5,
      slowDown: 8,
      strongGravityMode: false,
    },
  });
  graph.forEachNode((node, attributes) => {
    graph.mergeNodeAttributes(node, { allX: attributes.x, allY: attributes.y });
  });

  const runtimeGraph = new Graph({ multi: true, type: "directed", allowSelfLoops: true });
  for (const region of dataset.regions.filter((item) => !isModelState(item))) {
    runtimeGraph.addNode(region.id, {
      x: hashUnit(region.id, 11),
      y: hashUnit(region.id, 37),
    });
  }
  for (const edge of dataset.edges) {
    if (runtimeGraph.hasNode(edge.source) && runtimeGraph.hasNode(edge.target)) {
      runtimeGraph.addDirectedEdgeWithKey(edge.id, edge.source, edge.target);
    }
  }
  forceAtlas2.assign(runtimeGraph, {
    iterations: 240,
    settings: {
      ...forceAtlas2.inferSettings(runtimeGraph),
      adjustSizes: false,
      barnesHutOptimize: false,
      gravity: 0.7,
      scalingRatio: 9,
      slowDown: 6,
      strongGravityMode: false,
    },
  });
  runtimeGraph.forEachNode((node, attributes) => {
    graph.mergeNodeAttributes(node, {
      runtimeX: attributes.x,
      runtimeY: attributes.y,
      x: attributes.x,
      y: attributes.y,
    });
  });
  return graph;
}

export function applyCompactLayout(graph: Graph): void {
  const remainingIncoming = new Map<string, number>();
  graph.forEachNode((node) => remainingIncoming.set(node, graph.inDegree(node)));
  const queue = graph
    .filterNodes((node) => remainingIncoming.get(node) === 0)
    .sort();
  const order: string[] = [];
  while (queue.length > 0) {
    const node = queue.shift()!;
    order.push(node);
    const targets = graph.outNeighbors(node).sort();
    for (const target of targets) {
      const count = remainingIncoming.get(target)! - 1;
      remainingIncoming.set(target, count);
      if (count === 0) queue.push(target);
    }
    queue.sort();
  }
  if (order.length !== graph.order) return;

  order.forEach((node, index) => {
    const position = {
      x: index % 2 === 0 ? -0.035 : 0.035,
      y: (order.length - 1) / 2 - index,
    };
    graph.mergeNodeAttributes(node, {
      ...position,
      allX: position.x,
      allY: position.y,
      runtimeX: position.x,
      runtimeY: position.y,
    });
  });
}

export function applyScopeLayout(graph: Graph, scope: Scope): void {
  graph.forEachNode((node, attributes) => {
    const region = attributes.region as Region;
    const runtimeLayout = scope === "runtime" && !isModelState(region);
    graph.mergeNodeAttributes(node, {
      x: runtimeLayout ? attributes.runtimeX : attributes.allX,
      y: runtimeLayout ? attributes.runtimeY : attributes.allY,
    });
  });
}

export function directNeighborhood(
  graph: Graph,
  node: string,
  mode: ModeFilter,
  scope: Scope,
): Set<string> {
  const focus = new Set<string>([node]);
  graph.forEachNeighbor(node, (neighbor) => {
    const region = graph.getNodeAttribute(neighbor, "region") as Region;
    if (inMode(region, mode) && inScope(region, scope)) focus.add(neighbor);
  });
  return focus;
}

export function shortestPath(
  graph: Graph,
  source: string,
  target: string,
  mode: ModeFilter,
  scope: Scope,
): Set<string> | null {
  if (source === target) return new Set([source]);
  const queue = [source];
  const previous = new Map<string, string | null>([[source, null]]);

  while (queue.length > 0) {
    const current = queue.shift()!;
    for (const edgeKey of graph.outEdges(current)) {
      const edge = graph.getEdgeAttribute(edgeKey, "edge") as EvidenceEdge;
      if (!inMode(edge, mode)) continue;
      const neighbor = graph.target(edgeKey);
      const region = graph.getNodeAttribute(neighbor, "region") as Region;
      if (!inMode(region, mode) || !inScope(region, scope) || previous.has(neighbor)) continue;
      previous.set(neighbor, current);
      if (neighbor === target) {
        const path = new Set<string>();
        let cursor: string | null = target;
        while (cursor) {
          path.add(cursor);
          cursor = previous.get(cursor) ?? null;
        }
        return path;
      }
      queue.push(neighbor);
    }
  }
  return null;
}

export function edgeBelongsToFocus(
  graph: Graph,
  edgeKey: string,
  focus: ReadonlySet<string> | null,
): boolean {
  return !focus || (focus.has(graph.source(edgeKey)) && focus.has(graph.target(edgeKey)));
}
