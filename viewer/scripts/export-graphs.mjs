import { createHash } from "node:crypto";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";

const MODES = ["ordinary", "control", "stale"];
const SOURCE_LABELS = {
  ordinary: "Ordinary policy rollout",
  control: "Current-image control rollout",
  stale: "Stale-image intervention rollout",
};
const VIEWS = new Set(["safe", "qwen"]);

function parseArguments(values) {
  const parsed = {};
  for (let index = 0; index < values.length; index += 2) {
    const flag = values[index];
    const value = values[index + 1];
    if (!flag?.startsWith("--") || !value) throw new Error(`invalid argument near ${flag ?? "end"}`);
    parsed[flag.slice(2)] = value;
  }
  for (const required of [...MODES, "output"]) {
    if (!parsed[required]) throw new Error(`missing --${required}`);
  }
  parsed.view ??= "safe";
  if (!VIEWS.has(parsed.view)) throw new Error(`unsupported --view: ${parsed.view}`);
  return parsed;
}

function sha256(text) {
  return createHash("sha256").update(text).digest("hex");
}

function sorted(values) {
  return [...values].sort();
}

function canonicalRegion(region, sinkNames) {
  const value = {
    name: region.name,
    semanticKey: region.semantic_key,
    faultInterface: region.fault_interface ?? null,
    lifetime: region.lifetime ?? null,
    aggregation: region.aggregation ?? null,
    disposition: region.disposition ?? null,
    basis: sorted(region.basis ?? []),
    reachableSinks: sorted((region.reachable_sinks ?? []).map((id) => sinkNames.get(id))),
  };
  if (value.reachableSinks.some((name) => !name)) {
    throw new Error(`region ${region.region_id} refers to an unknown sink`);
  }
  return value;
}

async function loadMode(mode, graphPath) {
  const graphText = await readFile(graphPath, "utf8");
  const graph = JSON.parse(graphText);
  const auditPath = path.join(path.dirname(graphPath), "audit.json");
  const auditText = await readFile(auditPath, "utf8");
  const audit = JSON.parse(auditText);
  if (audit.passed !== true) throw new Error(`${mode} graph does not have a passing audit`);
  if (!Array.isArray(graph.regions) || !Array.isArray(graph.edges) || !Array.isArray(graph.sinks)) {
    throw new Error(`${mode} graph has an unsupported schema`);
  }

  const sinkNames = new Map(graph.sinks.map((sink) => [sink.event_id, sink.name]));
  const byRegionId = new Map();
  const regions = graph.regions.map((region) => {
    const canonical = canonicalRegion(region, sinkNames);
    const key = JSON.stringify(canonical);
    if ([...byRegionId.values()].some((item) => item.key === key)) {
      throw new Error(`${mode} graph has a duplicate canonical region: ${region.region_id}`);
    }
    byRegionId.set(region.region_id, { key, canonical });
    return {
      key,
      canonical,
      variant: { regionId: region.region_id, eventCount: region.event_count },
    };
  });
  const edges = graph.edges.map((edge) => {
    const source = byRegionId.get(edge.source);
    const target = byRegionId.get(edge.target);
    if (!source || !target) throw new Error(`${mode} edge has an unknown endpoint`);
    return { key: JSON.stringify([edge.kind, source.key, target.key]), kind: edge.kind, source: source.key, target: target.key };
  });
  return {
    mode,
    graphPath,
    graphSha256: sha256(graphText),
    auditSha256: sha256(auditText),
    auditPassed: true,
    regionCount: graph.regions.length,
    edgeCount: graph.edges.length,
    regions,
    edges,
  };
}

const args = parseArguments(process.argv.slice(2));
const inputs = await Promise.all(MODES.map((mode) => loadMode(mode, path.resolve(args[mode]))));
const regionMap = new Map();
for (const input of inputs) {
  for (const region of input.regions) {
    const existing = regionMap.get(region.key) ?? {
      id: `r-${sha256(region.key).slice(0, 16)}`,
      ...region.canonical,
      modes: [],
      variants: {},
    };
    existing.modes.push(input.mode);
    existing.variants[input.mode] = region.variant;
    regionMap.set(region.key, existing);
  }
}

const edgeMap = new Map();
for (const input of inputs) {
  for (const edge of input.edges) {
    const existing = edgeMap.get(edge.key) ?? {
      id: `e-${sha256(edge.key).slice(0, 16)}`,
      source: regionMap.get(edge.source).id,
      target: regionMap.get(edge.target).id,
      kind: edge.kind,
      modes: [],
    };
    existing.modes.push(input.mode);
    edgeMap.set(edge.key, existing);
  }
}

const regions = [...regionMap.values()].sort((left, right) => left.id.localeCompare(right.id));
const edges = [...edgeMap.values()].sort((left, right) => left.id.localeCompare(right.id));
const output = {
  schemaVersion: 1,
  view: args.view,
  trustBoundary: args.view === "qwen"
    ? "Frozen scoring fields and declared Qwen adapter boundaries; no Qwen internal operator trace, risk labels, or viewer-authored semantic groups."
    : "Canonical artifact fields only; no risk labels or viewer-authored semantic groups.",
  sources: inputs.map((input) => ({
    mode: input.mode,
    label: SOURCE_LABELS[input.mode],
    graphSha256: input.graphSha256,
    auditSha256: input.auditSha256,
    auditPassed: input.auditPassed,
    regionCount: input.regionCount,
    edgeCount: input.edgeCount,
  })),
  totals: {
    regions: regions.length,
    edges: edges.length,
    modelStateRegions: regions.filter((region) => region.faultInterface === "registered_model_state").length,
    runtimeRegions: regions.filter((region) => region.faultInterface !== "registered_model_state").length,
  },
  regions,
  edges,
};

const outputPath = path.resolve(args.output);
await mkdir(path.dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(output)}\n`, "utf8");
console.log(JSON.stringify({ output: outputPath, ...output.totals }));
