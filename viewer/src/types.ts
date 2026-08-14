export const MODES = ["ordinary", "control", "stale"] as const;
export type Mode = (typeof MODES)[number];
export type ModeFilter = Mode | "union";
export type Scope = "runtime" | "all";

export interface RegionVariant {
  regionId: string;
  eventCount: number;
}

export interface Region {
  id: string;
  name: string;
  semanticKey: string;
  faultInterface: string | null;
  lifetime: string | null;
  aggregation: string | null;
  disposition: string | null;
  basis: string[];
  reachableSinks: string[];
  modes: Mode[];
  variants: Partial<Record<Mode, RegionVariant>>;
}

export interface EvidenceEdge {
  id: string;
  source: string;
  target: string;
  kind: string;
  modes: Mode[];
}

export interface SourceArtifact {
  mode?: Mode;
  label: string;
  graphSha256: string;
  auditSha256: string;
  auditPassed: boolean;
  regionCount: number;
  edgeCount: number;
}

export interface CoverageSummary {
  discoveryItems: number;
  totalItems: number;
  novelHoldouts: number;
  zeroNoveltyUpper95: number | null;
}

export interface TraceCampaign {
  basis: string;
  selectionSha256: string;
  campaignSha256: string;
  manifestSha256: string;
  representativeGraphSha256: string;
  representativeAuditSha256: string;
  traceRevision: string;
  totalQueries: number;
  discoveryQueries: number;
  holdoutQueries: number;
  failedAttempts: number;
  conditions: Record<Mode, number>;
  coverage: {
    regions: CoverageSummary;
    edges: CoverageSummary;
    operators: CoverageSummary;
    processor_shapes: CoverageSummary;
  };
}

export interface GraphDataset {
  schemaVersion: number;
  view?: "safe" | "qwen";
  trustBoundary: string;
  sources: SourceArtifact[];
  traceCampaign?: TraceCampaign;
  totals: {
    regions: number;
    edges: number;
    modelStateRegions: number;
    runtimeRegions: number;
  };
  regions: Region[];
  edges: EvidenceEdge[];
}
