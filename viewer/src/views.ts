export const VIEW_IDS = ["safe", "qwen"] as const;
export type ViewId = (typeof VIEW_IDS)[number];

export interface ViewDefinition {
  id: ViewId;
  label: string;
  subtitle: string;
  datasetPath: string;
  monitorLabel: string;
  monitorSink: string;
  outcomeSink: string;
  pathTargetRegion: string;
  pathTargetSink: string;
  pathTargetLabel: string;
  emptyHeading: string;
  emptyDescription: string;
}

const DEFINITIONS: Record<ViewId, ViewDefinition> = {
  safe: {
    id: "safe",
    label: "SAFE",
    subtitle: "OpenVLA · LIBERO-10 · SAFE",
    datasetPath: "data/openvla-safe-graph.json",
    monitorLabel: "SAFE",
    monitorSink: "rollout.monitor_timeline",
    outcomeSink: "rollout.outcome",
    pathTargetRegion: "task_outcome",
    pathTargetSink: "rollout.outcome",
    pathTargetLabel: "task outcome",
    emptyHeading: "Follow one evidence path",
    emptyDescription: "Select a region to inspect its provenance and its recorded path to the task outcome.",
  },
  qwen: {
    id: "qwen",
    label: "Qwen",
    subtitle: "OpenVLA · LIBERO-10 · Qwen",
    datasetPath: "data/openvla-qwen-graph.json",
    monitorLabel: "Qwen",
    monitorSink: "qwen.alarm",
    outcomeSink: "rollout.outcome",
    pathTargetRegion: "qwen_alarm",
    pathTargetSink: "qwen.alarm",
    pathTargetLabel: "Qwen alarm",
    emptyHeading: "Follow one evidence path",
    emptyDescription: "Select a region to inspect its provenance and recorded path to the Qwen alarm. The path is reconstructed from frozen scoring artifacts; Qwen internals remain opaque.",
  },
};

export function activeView(search: string): ViewDefinition {
  const requested = new URLSearchParams(search).get("view");
  return DEFINITIONS[requested === "qwen" ? "qwen" : "safe"];
}
