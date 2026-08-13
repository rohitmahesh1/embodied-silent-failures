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
  mode: Mode;
  label: string;
  graphSha256: string;
  auditSha256: string;
  auditPassed: boolean;
  regionCount: number;
  edgeCount: number;
}

export interface GraphDataset {
  schemaVersion: number;
  view?: "safe" | "qwen";
  trustBoundary: string;
  sources: SourceArtifact[];
  totals: {
    regions: number;
    edges: number;
    modelStateRegions: number;
    runtimeRegions: number;
  };
  regions: Region[];
  edges: EvidenceEdge[];
}
