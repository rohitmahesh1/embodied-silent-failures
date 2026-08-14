import {
  describeBehavior,
  qwenIdentity,
  stateIdentity,
  titleCase,
} from "./presentation";
import type { Region } from "./types";

function shortSymbol(value: string): string {
  const parts = value.split(".");
  if (parts.length < 2) return value;
  const last = parts.at(-1)!;
  const previous = parts.at(-2)!;
  return /^[A-Z]/.test(previous) || last.startsWith("__")
    ? `${previous}.${last}`
    : last;
}

export function sourceDescription(
  basis: string,
  region: Region,
): { kind: string; text: string } {
  const qwenCode = basis.match(/^code:qwen@([^:]+):([^:]+):([^:]+):file-sha256-([a-f0-9]{64})$/);
  if (qwenCode) {
    const [, revision, implementation, method] = qwenCode;
    return {
      kind: "Implementation",
      text: `Qwen3-VL-8B-Instruct revision ${revision} on Hugging Face, using ${shortSymbol(implementation)}.${method} from the exact recorded Transformers implementation.`,
    };
  }

  const code = basis.match(/^code:([^@]+)@([^:]+):([^:]+):(.+)$/);
  if (code) {
    const [, repository, commit, symbol, behavior] = code;
    const repositoryName: Record<string, string> = {
      openvla: "OpenVLA",
      libero: "LIBERO",
      safe: "SAFE",
      "embodied-silent-failures": "our experiment code",
      "huggingface-transformers": "Hugging Face Transformers",
      qwen: "Qwen",
    };
    return {
      kind: "Implementation",
      text: `${repositoryName[repository] ?? titleCase(repository)} commit ${commit}, in ${shortSymbol(symbol)}: ${describeBehavior(behavior)}.`,
    };
  }

  const localCode = basis.match(/^code:([^:]+):([^:]+):(.+)$/);
  if (localCode) {
    const [, repository, symbol, behavior] = localCode;
    const repositoryName = repository === "embodied-silent-failures"
      ? "Our experiment code"
      : titleCase(repository);
    return {
      kind: "Implementation",
      text: `${repositoryName}, in ${shortSymbol(symbol)}: ${describeBehavior(behavior)}.`,
    };
  }

  const paper = basis.match(/^paper:([^:]+):(.+)$/);
  if (paper) {
    const [, paperId, location] = paper;
    if (paperId.startsWith("openvla") && location === "fig2") {
      return { kind: "Paper", text: "The OpenVLA architecture in Figure 2." };
    }
    if (paperId.startsWith("safe") && location === "sec4.2+appendix-b.1+b.2") {
      return {
        kind: "Paper",
        text: "The SAFE feature definition in Section 4.2 and Appendices B.1–B.2.",
      };
    }
    if (paperId.startsWith("hide-and-seek") && location === "appendix-g.5") {
      return {
        kind: "Paper",
        text: "The Qwen monitor described in Hide-and-Seek, Appendix G.5.",
      };
    }
    return {
      kind: "Paper",
      text: "The corresponding architecture description in the cited paper.",
    };
  }

  const observedModule = basis.match(/^observed:torch-module:(.+)$/);
  if (observedModule) {
    const moduleNames: Record<string, string> = {
      "policy.language_model": "OpenVLA language model",
      "policy.language_model.lm_head": "OpenVLA action-token output layer",
      "policy.projector": "OpenVLA multimodal projector",
      "policy.vision_backbone": "OpenVLA vision encoder",
    };
    const qwenModule = observedModule[1].startsWith("qwen_model.");
    const moduleName = qwenModule
      ? qwenIdentity(observedModule[1])
      : moduleNames[observedModule[1]] ?? "corresponding OpenVLA module";
    const statePath = region.semanticKey.split("/state/")[1];
    if (statePath) {
      const identity = stateIdentity(titleCase(region.name), statePath)
        .replaceAll(" · ", ", ");
      return {
        kind: "Runtime",
        text: `The runtime trace recorded this as registered model state at ${identity}.`,
      };
    }
    const actionToken = region.semanticKey.match(/action_token_(\d+)$/);
    const tokenContext = actionToken
      ? ` while generating action token ${actionToken[1]}`
      : " during policy execution";
    const context = qwenModule ? " during Qwen monitor inference" : tokenContext;
    return {
      kind: "Runtime",
      text: `The runtime trace attributed this computation to ${qwenModule ? "" : "the "}${moduleName}${context}.`,
    };
  }
  if (basis === "observed:torch-dispatch-outside-module") {
    return {
      kind: "Runtime",
      text: "Observed as a PyTorch operation outside a registered model module.",
    };
  }
  if (basis === "observed:rollout-video:decoded-current-camera-frame") {
    return {
      kind: "Artifact",
      text: "The camera frame decoded from the rollout video and fixed by its recorded RGB hash.",
    };
  }

  const protocol = basis.match(/^protocol:([^:]+):(.+)$/);
  if (protocol) {
    const [, protocolId, behavior] = protocol;
    const protocolNames: Record<string, string> = {
      "counterfactual-replay-v1": "paired counterfactual protocol",
      "evidence-graph-v1": "evidence capture protocol",
      "rollout-evidence-v1": "rollout protocol",
      "stale-image-v1": "stale-image protocol",
      "qwen-observation-monitor-v1": "Qwen observation-monitor protocol",
      "qwen-rollout-evidence-v1": "Qwen rollout-evidence protocol",
    };
    return {
      kind: "Protocol",
      text: `Our ${protocolNames[protocolId] ?? titleCase(protocolId)} ${describeBehavior(behavior)}.`,
    };
  }

  return { kind: "Recorded source", text: titleCase(basis) };
}
