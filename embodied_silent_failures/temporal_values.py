from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from embodied_silent_failures.artifacts import artifact_record, write_npz_atomic
from embodied_silent_failures.temporal_fault import value_at_port, value_slice


def _copy_value(torch: Any, np: Any, value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().contiguous().cpu().clone()
    array = np.asarray(value)
    if array.dtype.kind not in "biufc":
        raise TypeError(f"temporal collection requires numeric data, got {array.dtype}")
    return array.copy()


class TemporalValueCollector:
    """Collect table-defined values reached during one policy decision."""

    def __init__(self, torch: Any, np: Any, sites: list[dict[str, Any]]) -> None:
        self._torch = torch
        self._np = np
        self._sites = {str(site["site_id"]): site for site in sites}
        if len(self._sites) != len(sites):
            raise ValueError("temporal collector site IDs must be unique")
        self._module_sites: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._boundary_sites: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for site in sites:
            identity = site["identity"]
            if identity["kind"] == "module_output":
                self._module_sites[str(identity["module_path"])].append(site)
            elif identity["kind"] == "declared_runtime_boundary":
                self._boundary_sites[str(identity["event_name"])].append(site)
            else:
                raise ValueError(f"unsupported temporal identity: {identity['kind']}")
        self._handles: list[Any] = []
        self._install_errors: dict[str, str] = {}
        self._policy_step: int | None = None
        self._module_calls: dict[str, int] = {}
        self._boundary_calls: dict[str, int] = {}
        self._values: dict[str, Any] = {}
        self._errors: dict[str, str] = {}

    @property
    def values(self) -> dict[str, Any]:
        return dict(self._values)

    @property
    def errors(self) -> dict[str, str]:
        return dict(self._errors)

    def install(self, model: Any) -> None:
        if self._handles:
            raise RuntimeError("temporal collector hooks are already installed")
        modules = dict(model.named_modules())
        for module_path in sorted(self._module_sites):
            if not module_path.startswith("policy"):
                for site in self._module_sites[module_path]:
                    self._install_errors[str(site["site_id"])] = (
                        f"module site is outside the policy root: {module_path}"
                    )
                continue
            relative = module_path.removeprefix("policy").removeprefix(".")
            if relative not in modules:
                for site in self._module_sites[module_path]:
                    self._install_errors[str(site["site_id"])] = (
                        f"policy has no traced module path {module_path}"
                    )
                continue
            self._handles.append(
                modules[relative].register_forward_hook(
                    self._module_hook(module_path)
                )
            )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._values = {}

    def begin_capture(self) -> None:
        if self._policy_step is not None:
            raise RuntimeError("cannot reset an active temporal collection")
        self._values = {}
        self._errors = dict(self._install_errors)

    @contextmanager
    def inference(self, policy_step: int) -> Iterator[None]:
        if self._policy_step is not None:
            raise RuntimeError("temporal collection contexts cannot be nested")
        self._policy_step = policy_step
        self._module_calls = {}
        self._boundary_calls = {}
        try:
            yield
        finally:
            self._policy_step = None

    def boundary(
        self, event_name: str, output: Any, *, policy_step: int | None = None
    ) -> Any:
        active_step = self._policy_step if policy_step is None else policy_step
        if active_step is None:
            raise RuntimeError("temporal collection boundary ran outside a policy step")
        call_index = self._boundary_calls.get(event_name, 0)
        self._boundary_calls[event_name] = call_index + 1
        for site in self._boundary_sites.get(event_name, []):
            identity = site["identity"]
            if call_index == int(identity["event_call_index"]):
                self._record(site, output)
        return output

    def missing_site_ids(self) -> list[str]:
        return sorted(set(self._sites) - set(self._values) - set(self._errors))

    def _module_hook(self, module_path: str) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            call_index = self._module_calls.get(module_path, 0)
            self._module_calls[module_path] = call_index + 1
            for site in self._module_sites[module_path]:
                identity = site["identity"]
                if call_index == int(identity["module_call_index"]):
                    self._record(site, output)
            return output

        return hook

    def _record(self, site: dict[str, Any], output: Any) -> None:
        site_id = str(site["site_id"])
        if site_id in self._values or site_id in self._errors:
            self._errors[site_id] = "site was reached more than once in one decision"
            self._values.pop(site_id, None)
            return
        try:
            value = value_at_port(output, str(site["identity"]["output_port"]))
            value = value_slice(
                value, str(site.get("intervention", {}).get("value_slice", "full"))
            )
            self._values[site_id] = _copy_value(self._torch, self._np, value)
        except Exception as error:
            self._errors[site_id] = f"{type(error).__name__}: {error}"


def _encoded_value(torch: Any, np: Any, value: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().contiguous().cpu()
        encoded = tensor.view(torch.uint8).numpy().copy()
        kind = "torch_tensor"
        dtype = str(tensor.dtype)
        shape = list(tensor.shape)
    else:
        array = np.ascontiguousarray(np.asarray(value))
        encoded = np.frombuffer(array.tobytes(order="C"), dtype=np.uint8).copy()
        kind = "numpy_array"
        dtype = array.dtype.str
        shape = list(array.shape)
    digest = hashlib.sha256()
    digest.update(kind.encode("ascii"))
    digest.update(b"\0")
    digest.update(dtype.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(encoded.tobytes())
    return encoded, {
        "kind": kind,
        "dtype": dtype,
        "shape": shape,
        "encoding": "contiguous raw bytes stored as uint8",
        "sha256": digest.hexdigest(),
    }


def write_temporal_value_archive(
    path: Path,
    runtime: Any,
    sites: list[dict[str, Any]],
    source_values: dict[str, Any],
    current_values: dict[str, Any],
) -> dict[str, Any]:
    arrays = {}
    entries = []
    for index, site in enumerate(sorted(sites, key=lambda value: value["site_id"])):
        site_id = str(site["site_id"])
        entry = {"site_id": site_id}
        for moment, values in (("source", source_values), ("current", current_values)):
            if site_id not in values:
                entry[moment] = None
                continue
            key = f"{moment}_{index:04d}"
            encoded, metadata = _encoded_value(
                runtime.torch, runtime.np, values[site_id]
            )
            arrays[key] = encoded
            entry[moment] = {"archive_key": key, **metadata}
        entries.append(entry)
    write_npz_atomic(path, runtime.np, arrays)
    return {
        "schema_version": 1,
        "format": "compressed NumPy archive readable with allow_pickle=False",
        "artifact": artifact_record(path),
        "entries": entries,
    }
