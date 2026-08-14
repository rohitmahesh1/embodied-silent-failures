import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

const COVERAGE_KINDS = ["regions", "edges", "operators", "processor_shapes"];
const MODES = ["ordinary", "control", "stale"];

function parseArguments(values) {
  const parsed = {};
  for (let index = 0; index < values.length; index += 2) {
    const flag = values[index];
    const value = values[index + 1];
    if (!flag?.startsWith("--") || !value) throw new Error(`invalid argument near ${flag ?? "end"}`);
    parsed[flag.slice(2)] = value;
  }
  for (const required of ["campaign", "seed-trace", "output"]) {
    if (!parsed[required]) throw new Error(`missing --${required}`);
  }
  return parsed;
}

function sha256(text) {
  return createHash("sha256").update(text).digest("hex");
}

async function loadJson(file) {
  const text = await readFile(file, "utf8");
  return { value: JSON.parse(text), text, sha256: sha256(text) };
}

function signatures(coverage, kind) {
  if (!Array.isArray(coverage[kind])) throw new Error(`coverage has no ${kind} records`);
  const values = new Set(coverage[kind].map((item) => item.signature));
  if (values.has(undefined) || values.size !== coverage[kind].length) {
    throw new Error(`coverage has invalid or duplicate ${kind} signatures`);
  }
  return values;
}

function addAll(target, values) {
  for (const value of values) target.add(value);
}

function differenceSize(values, baseline) {
  let count = 0;
  for (const value of values) if (!baseline.has(value)) count += 1;
  return count;
}

function equalSets(left, right) {
  return left.size === right.size && differenceSize(left, right) === 0;
}

function conditionFromStratum(stratum) {
  const condition = stratum.split("--", 1)[0];
  if (!MODES.includes(condition)) throw new Error(`unsupported trace stratum: ${stratum}`);
  return condition;
}

function canonicalRegion(region, sinkNames) {
  const reachableSinks = [...(region.reachable_sinks ?? [])].map((id) => sinkNames.get(id)).sort();
  if (reachableSinks.some((name) => !name)) {
    throw new Error(`region ${region.region_id} refers to an unknown sink`);
  }
  return {
    name: region.name,
    semanticKey: region.semantic_key,
    faultInterface: region.fault_interface ?? null,
    lifetime: region.lifetime ?? null,
    aggregation: region.aggregation ?? null,
    disposition: region.disposition ?? null,
    basis: [...(region.basis ?? [])].sort(),
    reachableSinks,
  };
}

function publicGraph(graph) {
  const sinkNames = new Map(graph.sinks.map((sink) => [sink.event_id, sink.name]));
  const byRegionId = new Map();
  const regions = graph.regions.map((region) => {
    const canonical = canonicalRegion(region, sinkNames);
    const key = JSON.stringify(canonical);
    if ([...byRegionId.values()].some((item) => item.key === key)) {
      throw new Error(`representative graph has a duplicate canonical region: ${region.region_id}`);
    }
    const record = {
      id: `r-${sha256(key).slice(0, 16)}`,
      ...canonical,
      modes: [...MODES],
      variants: {},
    };
    byRegionId.set(region.region_id, { key, record });
    return record;
  });
  const edgeKeys = new Set();
  const edges = graph.edges.map((edge) => {
    const source = byRegionId.get(edge.source);
    const target = byRegionId.get(edge.target);
    if (!source || !target) throw new Error("representative graph edge has an unknown endpoint");
    const key = JSON.stringify([edge.kind, source.key, target.key]);
    if (edgeKeys.has(key)) throw new Error("representative graph has a duplicate canonical edge");
    edgeKeys.add(key);
    return {
      id: `e-${sha256(key).slice(0, 16)}`,
      source: source.record.id,
      target: target.record.id,
      kind: edge.kind,
      modes: [...MODES],
    };
  });
  regions.sort((left, right) => left.id.localeCompare(right.id));
  edges.sort((left, right) => left.id.localeCompare(right.id));
  return { regions, edges };
}

function attemptDirectory(campaignRoot, outcome) {
  const directory = `${String(outcome.index).padStart(3, "0")}--${outcome.phase}--${outcome.stratum}`;
  const attempt = path.basename(outcome.artifact);
  if (!/^attempt-\d{3}$/.test(attempt)) throw new Error(`invalid attempt path for query ${outcome.index}`);
  return path.join(campaignRoot, "queries", directory, attempt);
}

function upperBound(zeroQueries, confidence = 0.95) {
  return 1 - (1 - confidence) ** (1 / zeroQueries);
}

const args = parseArguments(process.argv.slice(2));
const campaignRoot = path.resolve(args.campaign);
const seedRoot = path.resolve(args["seed-trace"]);
const campaignFile = await loadJson(path.join(campaignRoot, "campaign.json"));
const manifestFile = await loadJson(path.join(campaignRoot, "manifest.json"));
const seedCoverageFile = await loadJson(path.join(campaignRoot, "seed-coverage.json"));
const seedGraphFile = await loadJson(path.join(seedRoot, "graph.json"));
const seedAuditFile = await loadJson(path.join(seedRoot, "audit.json"));
const seedCompositionFile = await loadJson(path.join(seedRoot, "composition.json"));
const campaign = campaignFile.value;
const manifest = manifestFile.value;

if (campaign.status !== "complete" || campaign.completed_queries !== campaign.total_queries) {
  throw new Error("Qwen trace campaign is not complete");
}
if (campaign.failed_attempts !== 0 || campaign.outcomes.length !== campaign.total_queries) {
  throw new Error("Qwen trace campaign contains failed or missing attempts");
}
if (manifest.selection_sha256 !== campaign.selection_sha256 || manifest.total_queries !== campaign.total_queries) {
  throw new Error("Qwen campaign and frozen selection manifest disagree");
}
if (seedAuditFile.value.passed !== true) throw new Error("representative Qwen trace audit did not pass");
for (const [name, digest] of Object.entries(manifest.seed_trace.hashes)) {
  if (!["graph.json", "audit.json", "composition.json"].includes(name)) continue;
  const actual = { "graph.json": seedGraphFile, "audit.json": seedAuditFile, "composition.json": seedCompositionFile }[name].sha256;
  if (actual !== digest) throw new Error(`representative Qwen trace hash disagrees for ${name}`);
}

const records = [];
for (const outcome of campaign.outcomes) {
  if (outcome.status !== "complete") throw new Error(`Qwen trace ${outcome.index} is not complete`);
  if (outcome.index === 0) {
    records.push({ outcome, coverage: seedCoverageFile.value, mode: conditionFromStratum(outcome.stratum) });
    continue;
  }
  const directory = attemptDirectory(campaignRoot, outcome);
  const [coverageFile, auditFile] = await Promise.all([
    loadJson(path.join(directory, "coverage.json")),
    loadJson(path.join(directory, "audit.json")),
  ]);
  if (auditFile.value.passed !== true || outcome.result?.audit_passed !== true) {
    throw new Error(`Qwen trace ${outcome.index} did not pass both saved audits`);
  }
  records.push({ outcome, coverage: coverageFile.value, mode: conditionFromStratum(outcome.stratum) });
}

const discovery = Object.fromEntries(COVERAGE_KINDS.map((kind) => [kind, new Set()]));
const complete = Object.fromEntries(COVERAGE_KINDS.map((kind) => [kind, new Set()]));
for (const record of records.filter((item) => item.outcome.phase === "discovery")) {
  for (const kind of COVERAGE_KINDS) addAll(discovery[kind], signatures(record.coverage, kind));
}
const novelty = Object.fromEntries(COVERAGE_KINDS.map((kind) => [kind, 0]));
for (const record of records) {
  for (const kind of COVERAGE_KINDS) {
    const values = signatures(record.coverage, kind);
    addAll(complete[kind], values);
    if (record.outcome.phase === "holdout" && differenceSize(values, discovery[kind]) > 0) novelty[kind] += 1;
  }
}

const seedSets = Object.fromEntries(COVERAGE_KINDS.map((kind) => [kind, signatures(seedCoverageFile.value, kind)]));
for (const kind of ["regions", "edges", "operators"]) {
  if (!equalSets(seedSets[kind], complete[kind]) || novelty[kind] !== 0) {
    throw new Error(`representative graph is incomplete for Qwen ${kind} coverage`);
  }
}

// qwen_saturation.py at experiment commit cfa6016 predeclares six balanced
// discovery cells and thirty seeded holdouts. This exporter reports those
// records; it does not create semantic strata or infer risk from the viewer.
const graph = publicGraph(seedGraphFile.value);
if (graph.regions.length !== complete.regions.size || graph.edges.length !== complete.edges.size) {
  throw new Error("representative Qwen graph counts disagree with campaign coverage");
}
const holdoutQueries = records.filter((item) => item.outcome.phase === "holdout").length;
const traceCampaign = {
  basis: manifest.basis,
  selectionSha256: manifest.selection_sha256,
  campaignSha256: campaignFile.sha256,
  manifestSha256: manifestFile.sha256,
  representativeGraphSha256: seedGraphFile.sha256,
  representativeAuditSha256: seedAuditFile.sha256,
  traceRevision: seedCompositionFile.value.trace_revision,
  totalQueries: records.length,
  discoveryQueries: records.length - holdoutQueries,
  holdoutQueries,
  failedAttempts: campaign.failed_attempts,
  conditions: Object.fromEntries(MODES.map((mode) => [mode, records.filter((item) => item.mode === mode).length])),
  coverage: Object.fromEntries(COVERAGE_KINDS.map((kind) => [kind, {
    discoveryItems: discovery[kind].size,
    totalItems: complete[kind].size,
    novelHoldouts: novelty[kind],
    zeroNoveltyUpper95: novelty[kind] === 0 ? upperBound(holdoutQueries) : null,
  }])),
};
const output = {
  schemaVersion: 2,
  view: "qwen",
  trustBoundary: "PyTorch operators, registered model state, and declared evidence boundaries observed while replaying 36 pinned Qwen queries. This does not establish hidden kernel behavior, hardware placement, measured risk, or coverage of arbitrary inputs.",
  sources: [{
    label: "Representative audited Qwen internal trace",
    graphSha256: seedGraphFile.sha256,
    auditSha256: seedAuditFile.sha256,
    auditPassed: true,
    regionCount: graph.regions.length,
    edgeCount: graph.edges.length,
  }],
  traceCampaign,
  totals: {
    regions: graph.regions.length,
    edges: graph.edges.length,
    modelStateRegions: graph.regions.filter((region) => region.faultInterface === "registered_qwen_model_state").length,
    runtimeRegions: graph.regions.filter((region) => region.faultInterface !== "registered_qwen_model_state").length,
  },
  regions: graph.regions,
  edges: graph.edges,
};

const outputPath = path.resolve(args.output);
await mkdir(path.dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(output)}\n`, "utf8");
console.log(JSON.stringify({ output: outputPath, ...output.totals, holdoutQueries, novelty }));
