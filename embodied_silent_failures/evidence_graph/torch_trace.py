from contextlib import contextmanager
from typing import Any, Iterator

from embodied_silent_failures.evidence_graph.record import Recorder


@contextmanager
def capture_torch_operations(
    recorder: Recorder, roots: dict[str, Any]
) -> Iterator[None]:
    """Capture dispatched tensor operations and their active module scopes."""
    try:
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode
    except ImportError as error:
        raise RuntimeError("PyTorch is required for operator capture") from error

    module_stack: list[dict[str, Any]] = []
    module_calls: dict[str, int] = {}
    handles = []

    def enter(path: str):
        def hook(
            _module: Any, _inputs: tuple[Any, ...], _kwargs: dict[str, Any]
        ) -> None:
            call_index = module_calls.get(path, 0)
            module_calls[path] = call_index + 1
            module_stack.append({"path": path, "call_index": call_index})

        return hook

    def leave(path: str):
        def hook(
            module: Any,
            inputs: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            if not module_stack or module_stack[-1]["path"] != path:
                raise RuntimeError(f"module trace became unbalanced at {path}")
            call = module_stack[-1]
            recorder.mark(
                f"module.{path}",
                kind="module",
                inputs={
                    "args": inputs,
                    "kwargs": kwargs,
                    "observed_output": output,
                    "parameters": dict(module.named_parameters(recurse=False)),
                    "buffers": dict(module.named_buffers(recurse=False)),
                },
                outputs=output,
                details={
                    "module_path": path,
                    "module_call_index": call["call_index"],
                    "module_calls": list(module_stack),
                },
            )
            module_stack.pop()

        return hook

    registered_state: dict[int, dict[str, Any]] = {}
    for root_name, root in roots.items():
        for name, module in root.named_modules():
            path = root_name if not name else f"{root_name}.{name}"
            handles.append(
                module.register_forward_pre_hook(enter(path), with_kwargs=True)
            )
            handles.append(module.register_forward_hook(leave(path), with_kwargs=True))
        try:
            modules = root.named_modules(remove_duplicate=False)
        except TypeError:
            modules = root.named_modules()
        for name, module in modules:
            path = root_name if not name else f"{root_name}.{name}"
            for state_kind, values in (
                ("parameter", module.named_parameters(recurse=False)),
                ("buffer", module.named_buffers(recurse=False)),
            ):
                for item_name, value in values:
                    item = registered_state.setdefault(
                        id(value),
                        {
                            "value": value,
                            "root": root_name,
                            "state_kind": state_kind,
                            "registrations": [],
                        },
                    )
                    item["registrations"].append(
                        {"module_path": path, "name": item_name}
                    )

    for item in sorted(
        registered_state.values(),
        key=lambda value: [
            (entry["module_path"], entry["name"])
            for entry in value["registrations"]
        ],
    ):
        registrations = sorted(
            item["registrations"],
            key=lambda entry: (entry["module_path"], entry["name"]),
        )
        first = registrations[0]
        recorder.mark(
            f"{first['module_path']}.{first['name']}.registered_state",
            kind="state",
            outputs=item["value"],
            details={
                "root": item["root"],
                "state_kind": item["state_kind"],
                "registrations": registrations,
            },
        )

    class OperationMode(TorchDispatchMode):
        def __torch_dispatch__(
            self,
            function: Any,
            _types: Any,
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            keyword_arguments = kwargs or {}
            output = function(*args, **keyword_arguments)
            semantics = _operator_semantics(function, args, keyword_arguments)
            recorder.mark(
                str(function),
                kind="operator",
                inputs={"args": args, "kwargs": keyword_arguments},
                outputs=output,
                details={
                    "module_scope": [item["path"] for item in module_stack],
                    "module_calls": list(module_stack),
                    "operator_semantics": semantics,
                },
            )
            return output

    try:
        with OperationMode():
            yield
    finally:
        for handle in reversed(handles):
            handle.remove()


def _operator_semantics(
    function: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    schema = getattr(function, "_schema", None)
    if schema is None:
        return {"schema_status": "unavailable"}

    mutated = []
    aliases = []
    arguments = list(getattr(schema, "arguments", ()))
    returns = list(getattr(schema, "returns", ()))
    for index, argument in enumerate(arguments):
        name = str(getattr(argument, "name", index))
        present = name in kwargs or index < len(args)
        if not present:
            continue
        port = f"value.kwargs.{name}" if name in kwargs else f"value.args[{index}]"
        alias_info = getattr(argument, "alias_info", None)
        if alias_info is not None and bool(getattr(alias_info, "is_write", False)):
            mutated.append(port)

    for output_index, returned in enumerate(returns):
        output_alias = getattr(returned, "alias_info", None)
        output_sets = _alias_sets(output_alias)
        if not output_sets:
            continue
        for input_index, argument in enumerate(arguments):
            input_alias = getattr(argument, "alias_info", None)
            if not output_sets.intersection(_alias_sets(input_alias)):
                continue
            name = str(getattr(argument, "name", input_index))
            if name not in kwargs and input_index >= len(args):
                continue
            aliases.append(
                {
                    "output": f"value[{output_index}]" if len(returns) > 1 else "value",
                    "input": (
                        f"value.kwargs.{name}"
                        if name in kwargs
                        else f"value.args[{input_index}]"
                    ),
                }
            )

    return {
        "schema_status": "available",
        "schema": str(schema),
        "mutated_input_ports": sorted(mutated),
        "declared_aliases": aliases,
    }


def _alias_sets(alias_info: Any) -> set[str]:
    if alias_info is None:
        return set()
    values = set()
    for name in ("before_set", "after_set"):
        for value in getattr(alias_info, name, ()):
            values.add(str(value))
    return values


def contract_issues(
    events: list[dict[str, Any]], phases: tuple[str, ...]
) -> list[str]:
    issues = []
    for phase in phases:
        phase_events = [
            event
            for event in events
            if event.get("context", {}).get("phase") == phase
        ]
        for kind in ("module", "operator"):
            if not any(event["kind"] == kind for event in phase_events):
                issues.append(f"{phase} trace has no observed {kind} events")
    return issues
