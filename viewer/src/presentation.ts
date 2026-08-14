import { isModelState } from "./graph";
import type { Region } from "./types";
import type { ViewDefinition } from "./views";

export type SinkClass = "both" | "monitor" | "outcome" | "neither" | "state";

const componentNames: Record<string, string> = {
  "attn.proj": "attention output",
  "attn.qkv": "attention QKV",
  "input_layernorm": "input normalization",
  "ls1": "layer scale 1",
  "ls2": "layer scale 2",
  "mlp.down_proj": "MLP down projection",
  "mlp.fc1": "MLP input",
  "mlp.fc2": "MLP output",
  "mlp.gate_proj": "MLP gate projection",
  "mlp.linear_fc1": "MLP input",
  "mlp.linear_fc2": "MLP output",
  "mlp.up_proj": "MLP up projection",
  "norm1": "normalization 1",
  "norm2": "normalization 2",
  "post_attention_layernorm": "post-attention normalization",
  "self_attn.k_proj": "attention key projection",
  "self_attn.k_norm": "attention key normalization",
  "self_attn.o_proj": "attention output projection",
  "self_attn.q_proj": "attention query projection",
  "self_attn.q_norm": "attention query normalization",
  "self_attn.rotary_emb": "rotary position encoding",
  "self_attn.v_proj": "attention value projection",
};

const interfaceLabels: Record<string, string> = {
  action_tokens: "Generated action tokens",
  current_image_control: "Current-image control",
  environment_observation: "Camera observation",
  executed_command: "Robot command",
  final_layer_action_features: "Final-layer action features",
  model_input: "Image and task prompt",
  observation_image: "Processed camera image",
  policy_call: "Policy execution",
  policy_image_buffer: "Selected policy image",
  prior_policy_image_buffer: "Earlier policy image",
  processed_policy_input: "Processed policy input",
  raw_action: "Decoded action",
  registered_model_state: "Model parameters and buffers",
  replayed_executed_command: "Replayed clean command",
  replayed_safe_feature: "Replayed clean SAFE feature",
  safe_feature: "SAFE input feature",
  qwen_observation_frame: "Decoded camera frame",
  qwen_observation_history: "Camera-frame history",
  qwen_internal_compute: "Observed Qwen computation",
  qwen_processor_output: "Processed Qwen input",
  qwen_private_compute: "Qwen inference",
  qwen_response_decode: "Generated Qwen response",
  qwen_response_parser: "Qwen response parser",
  registered_qwen_model_state: "Qwen parameters and buffers",
  simulator_command: "Simulator command",
  stale_image: "Stale-image intervention",
};

const behaviorDescriptions: Record<string, string> = {
  "buffered-prior-policy-image": "uses a buffered image from an earlier policy step",
  "convert-openvla-gripper-coordinate-to-libero-command": "converts OpenVLA's gripper coordinate into a LIBERO command",
  "decode-and-unnormalize-action-tokens": "decodes and unnormalizes the generated action tokens",
  "executed-command-from-clean-artifact": "replays the command from the paired clean rollout",
  "forward-command-to-simulator-and-return-next-observation": "sends the command to the simulator and returns the next observation",
  "monitor-results-attached-after-scoring": "attaches the monitor result after SAFE scoring",
  "pinned-greedy-response-from-frozen-scoring-artifact": "uses the greedy response preserved by the pinned scoring run",
  "no-intervention-applied": "records that no intervention was applied",
  "preprocess-selected-image-and-task-prompt": "prepares the selected image and task prompt for the policy",
  "record-applied-intervention": "records the intervention that was applied",
  "record-libero-terminal-outcome": "records LIBERO's terminal task result",
  "return-generated-sequences-and-hidden-states": "returns generated action sequences and hidden states",
  "return-observation-after-prior-policy-command": "returns the observation produced after the prior command",
  "return-second-to-last-vision-block-features": "returns features from the second-to-last vision block",
  "safe-feature-from-clean-artifact": "replays SAFE's feature from the paired clean rollout",
  "select-final-action-token-feature": "selects the final-layer feature for each action token",
  "selected-policy-image": "records the image selected as policy input",
  "set-pinned-initial-state-and-apply-configured-wait-steps": "pins the initial simulator state and applies the configured wait steps",
  "recorded-policy-image-then-lossy-video-encode-and-rgb-decode": "derives the Qwen frame from the recorded policy image through the rollout video's lossy encode and RGB decode",
  "all-frozen-query-alarms-in-policy-step-order": "collects every frozen Qwen alarm in policy-step order",
  current_visual_observation_not_selected_by_stale_policy_input: "the current camera observation was not selected as policy input",
  intentionally_discarded_final_vision_block: "the final vision block is intentionally not used",
  not_applicable_clean_rollout: "no intervention applies to an ordinary rollout",
  policy_inference_replaced_by_counterfactual_replay: "policy inference was replaced by paired counterfactual replay",
};

export function titleCase(value: string): string {
  return value
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function naturalizePath(path: string): string {
  const names: Record<string, string> = {
    policy: "",
    language_model: "Language model",
    model: "",
    vision_backbone: "Vision encoder",
    fused_featurizer: "Fused branch",
    featurizer: "Vision branch",
    attn_pool: "Attention pool",
    patch_embed: "Patch embedding",
    embed_tokens: "Token embedding",
    lm_head: "Action-token output",
    projector: "Multimodal projector",
    proj: "Projection",
    norm: "Normalization",
    q: "Query",
    kv: "Key/value",
    fc1: "Layer 1",
    fc2: "Layer 2",
    fc3: "Layer 3",
  };
  return path
    .split(".")
    .map((part) => names[part] ?? titleCase(part))
    .filter(Boolean)
    .join(" · ");
}

export function qwenIdentity(path: string): string {
  const languageLayer = path.match(/^qwen_model\.model\.language_model\.layers\.(\d+)(?:\.(.+))?$/);
  if (languageLayer) {
    const detail = languageLayer[2];
    return detail
      ? `Language model · Layer ${languageLayer[1]} · ${componentNames[detail] ?? naturalizePath(detail)}`
      : `Language model · Layer ${languageLayer[1]}`;
  }

  const visionBlock = path.match(/^qwen_model\.model\.visual\.blocks\.(\d+)(?:\.(.+))?$/);
  if (visionBlock) {
    const detail = visionBlock[2];
    return detail
      ? `Vision encoder · Block ${visionBlock[1]} · ${componentNames[detail] ?? naturalizePath(detail)}`
      : `Vision encoder · Block ${visionBlock[1]}`;
  }

  const deepstack = path.match(/^qwen_model\.model\.visual\.deepstack_merger_list\.(\d+)(?:\.(.+))?$/);
  if (deepstack) {
    const detail = deepstack[2];
    return detail
      ? `Vision encoder · Deepstack merger ${deepstack[1]} · ${naturalizePath(detail)}`
      : `Vision encoder · Deepstack merger ${deepstack[1]}`;
  }

  const names: Array<[string, string]> = [
    ["qwen_model.model.language_model.embed_tokens", "Language model · Token embedding"],
    ["qwen_model.model.language_model.rotary_emb", "Language model · Rotary position encoding"],
    ["qwen_model.model.language_model.norm", "Language model · Output normalization"],
    ["qwen_model.model.language_model", "Language model"],
    ["qwen_model.model.visual.patch_embed", "Vision encoder · Patch embedding"],
    ["qwen_model.model.visual.merger", "Vision encoder · Merger"],
    ["qwen_model.model.visual", "Vision encoder"],
    ["qwen_model.lm_head", "Response-token output"],
    ["qwen_model.model", "Qwen model"],
    ["qwen_model", "Qwen model root"],
  ];
  for (const [prefix, label] of names) {
    if (path === prefix) return label;
    if (path.startsWith(`${prefix}.`)) {
      return `${label} · ${naturalizePath(path.slice(prefix.length + 1))}`;
    }
  }
  return naturalizePath(path);
}

export function stateIdentity(base: string, path: string): string {
  if (path.startsWith("qwen_model.")) return qwenIdentity(path);

  const languageLayer = path.match(/^policy\.language_model\.model\.layers\.(\d+)\.(.+)$/);
  if (languageLayer) {
    return `${base} · Layer ${languageLayer[1]} · ${componentNames[languageLayer[2]] ?? naturalizePath(languageLayer[2])}`;
  }

  const visionBlock = path.match(/^policy\.vision_backbone\.(fused_featurizer|featurizer)\.blocks\.(\d+)\.(.+)$/);
  if (visionBlock) {
    const branch = visionBlock[1] === "fused_featurizer" ? "Fused branch" : "Vision branch";
    return `${base} · ${branch} block ${visionBlock[2]} · ${componentNames[visionBlock[3]] ?? naturalizePath(visionBlock[3])}`;
  }

  return `${base} · ${naturalizePath(path)}`;
}

export function classifySink(region: Region, view: ViewDefinition): SinkClass {
  if (isModelState(region)) return "state";
  const monitor = region.reachableSinks.includes(view.monitorSink);
  const outcome = region.reachableSinks.includes(view.outcomeSink);
  if (monitor && outcome) return "both";
  if (monitor) return "monitor";
  if (outcome) return "outcome";
  return "neither";
}

export function colorForRegion(region: Region, view: ViewDefinition): string {
  const colors: Record<SinkClass, string> = {
    both: "#3568b8",
    monitor: "#16846d",
    outcome: "#d2643f",
    neither: "#929c9a",
    state: "#aeb8b5",
  };
  return colors[classifySink(region, view)];
}

export function readableInterface(region: Region): string {
  return region.faultInterface
    ? interfaceLabels[region.faultInterface] ?? titleCase(region.faultInterface)
    : "Observed computation";
}

export function describeBehavior(value: string): string {
  return behaviorDescriptions[value] ?? value.replace(/[_-]+/g, " ");
}

function compactReachability(region: Region, view: ViewDefinition): string {
  const monitor = region.reachableSinks.includes(view.monitorSink);
  const outcome = region.reachableSinks.includes(view.outcomeSink);
  if (monitor && outcome) return `${view.monitorLabel} + outcome`;
  if (monitor) return `${view.monitorLabel} only`;
  if (outcome) return "outcome only";
  return "no recorded sink";
}

export function displayLabel(
  region: Region,
  view: ViewDefinition,
  duplicateNameCount: number,
): string {
  const base = titleCase(region.name);
  const statePath = region.semanticKey.split("/state/")[1];
  if (statePath) return stateIdentity(base, statePath);

  const qwenModule = region.semanticKey.match(/^qwen_model\/module\/(.+)$/);
  if (qwenModule) return qwenIdentity(qwenModule[1]);

  const detail = region.semanticKey.split("/")[1];
  if (detail) return `${base} · ${titleCase(detail)}`;
  if (duplicateNameCount === 1) return base;

  const evidence = readableInterface(region);
  const status = compactReachability(region, view);
  if (region.faultInterface) return `${base} · ${evidence} · ${status}`;
  if (region.disposition) {
    return `${base} · ${describeBehavior(region.disposition)} · ${status}`;
  }
  return `${base} · ${status}`;
}

export function pathLabel(region: Region, compact: boolean): string {
  const base = titleCase(region.name);
  const statePath = region.semanticKey.split("/state/")[1];
  if (statePath) return stateIdentity(base, statePath);
  const qwenModule = region.semanticKey.match(/^qwen_model\/module\/(.+)$/);
  if (qwenModule) return qwenIdentity(qwenModule[1]);
  const detail = region.semanticKey.split("/")[1];
  if (compact) {
    if (detail) return titleCase(detail);
    if (region.faultInterface) return readableInterface(region);
    return base;
  }
  if (detail) return `${base} · ${titleCase(detail)}`;
  if (region.faultInterface) return `${base} · ${readableInterface(region)}`;
  return base;
}

export function groupTitle(name: string): string {
  const labels: Record<string, string> = {
    qwen_model_language_model: "Language model",
    qwen_model_visual: "Vision encoder",
    qwen_model_lm_head: "Response-token output",
  };
  return labels[name] ?? titleCase(name);
}

export function reachabilitySentence(region: Region, view: ViewDefinition): string {
  const monitor = region.reachableSinks.includes(view.monitorSink);
  const outcome = region.reachableSinks.includes(view.outcomeSink);
  if (monitor && outcome) {
    return `The recorded graph connects this region to both the ${view.monitorLabel} monitor and the task outcome.`;
  }
  if (monitor) {
    return `The recorded graph connects this region to the ${view.monitorLabel} monitor only.`;
  }
  if (outcome) return "The recorded graph connects this region to the task outcome only.";
  return "The recorded graph does not connect this region to either recorded sink.";
}

export function modeSentence(region: Region): string {
  if (region.modes.length === 3) {
    return "Observed in ordinary, control, and stale-image runs.";
  }
  const names = region.modes.map((mode) => mode === "stale" ? "stale-image" : mode);
  return `Observed in ${names.join(" and ")} runs.`;
}
