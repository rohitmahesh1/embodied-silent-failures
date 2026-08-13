import json
import weakref
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


GRAPH_EVENT_KINDS = {"boundary", "module", "opaque", "source", "state"}
TRACE_EVENT_KINDS = GRAPH_EVENT_KINDS | {"operator"}


@dataclass(frozen=True)
class LineageValue:
    """Give a scalar or reconstructed value an explicit provenance identity."""

    key: str
    value: Any


def read_events(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"trace line {line_number} is not a JSON object")
            events.append(value)
    return events


class Recorder:
    """Write an append-only trace while retaining events for immediate reduction."""

    def __init__(self, path: Path, metadata: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.events: list[dict[str, Any]] = []
        self.annotations: list[dict[str, Any]] = []
        self._file = path.open("x", encoding="utf-8")
        self._next_event = 0
        self._next_value = 0
        self._next_storage = 0
        self._objects: dict[int, tuple[Any, str]] = {}
        self._strong_objects: dict[int, Any] = {}
        self._lineage_values: dict[str, str] = {}
        self._storage_ids: dict[
            tuple[str, int], tuple[str, dict[int, weakref.ReferenceType[Any]]]
        ] = {}
        self._contexts: list[dict[str, Any]] = []
        self._closed = False
        self._append("trace_start", "trace", details=metadata)

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close(completed=exc_type is None)

    @contextmanager
    def scope(self, **context: Any) -> Iterator[None]:
        self._contexts.append(context)
        try:
            yield
        finally:
            self._contexts.pop()

    def source(
        self,
        name: str,
        outputs: Any,
        *,
        basis: str,
        region: str,
        role: str | None = None,
        lifetime: str = "step",
        fault_interface: str | None = None,
        disposition: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.mark(
            name,
            kind="source",
            outputs=outputs,
            basis=basis,
            region=region,
            role=role,
            lifetime=lifetime,
            fault_interface=fault_interface,
            disposition=disposition,
            details=details,
        )

    def mark(
        self,
        name: str,
        *,
        kind: str = "boundary",
        inputs: Any = None,
        outputs: Any = None,
        basis: str | list[str] | None = None,
        region: str | None = None,
        role: str | None = None,
        lifetime: str = "step",
        fault_interface: str | None = None,
        disposition: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if kind not in TRACE_EVENT_KINDS:
            raise ValueError(f"unsupported graph event kind: {kind}")
        annotated = any(
            value is not None
            for value in (basis, region, role, fault_interface, disposition)
        )
        semantic = {}
        if annotated:
            semantic = {
                "region": region,
                "basis": [basis] if isinstance(basis, str) else basis,
                "role": role,
                "lifetime": lifetime,
                "fault_interface": fault_interface,
                "disposition": disposition,
            }
            semantic = {
                key: value for key, value in semantic.items() if value is not None
            }
        return self._append(
            kind,
            name,
            inputs=self._references(inputs),
            outputs=self._references(outputs),
            semantic=semantic or None,
            details=details,
        )

    def lineage(self, key: str, value: Any) -> LineageValue:
        if not key:
            raise ValueError("lineage key must be nonempty")
        return LineageValue(key, value)

    def context(self, key: str) -> Any:
        for context in reversed(self._contexts):
            if key in context:
                return context[key]
        raise KeyError(f"recorder context has no {key}")

    def close(self, completed: bool = True) -> None:
        if self._closed:
            return
        self._append("trace_end", "trace", details={"completed": completed})
        self._file.close()
        self._closed = True

    def _append(
        self,
        kind: str,
        name: str,
        *,
        inputs: list[dict[str, Any]] | None = None,
        outputs: list[dict[str, Any]] | None = None,
        semantic: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("cannot append to a closed trace")
        context: dict[str, Any] = {}
        for values in self._contexts:
            context.update(values)
        event = {
            "event_id": f"e{self._next_event:08d}",
            "kind": kind,
            "name": name,
        }
        self._next_event += 1
        if inputs:
            event["inputs"] = inputs
        if outputs:
            event["outputs"] = outputs
        if context:
            event["context"] = context
        if semantic:
            self.annotations.append({"event_id": event["event_id"], **semantic})
        if details:
            event["details"] = details
        self.events.append(event)
        self._file.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        return event

    def _references(self, value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        references = []
        for path, leaf in _leaves(value):
            lineage_key = leaf.key if isinstance(leaf, LineageValue) else None
            described = leaf.value if isinstance(leaf, LineageValue) else leaf
            value_id = self._value_id(described, lineage_key=lineage_key)
            references.append(
                {
                    "port": path,
                    "value_id": value_id,
                    **self._describe(described),
                    **({"lineage_key": lineage_key} if lineage_key else {}),
                }
            )
        return references

    def _value_id(self, value: Any, *, lineage_key: str | None = None) -> str:
        if lineage_key is not None:
            existing = self._lineage_values.get(lineage_key)
            if existing is not None:
                return existing
            value_id = f"v{self._next_value:08d}"
            self._next_value += 1
            self._lineage_values[lineage_key] = value_id
            return value_id
        # Interned scalars such as False share Python identities across unrelated
        # calls, so each scalar occurrence is a distinct provenance value.
        if isinstance(value, (bool, int, float, str)):
            value_id = f"v{self._next_value:08d}"
            self._next_value += 1
            return value_id
        object_id = id(value)
        existing = self._objects.get(object_id)
        if existing is not None:
            reference, value_id = existing
            if isinstance(reference, weakref.ReferenceType):
                if reference() is value:
                    return value_id
            elif reference is value:
                return value_id

        value_id = f"v{self._next_value:08d}"
        self._next_value += 1
        try:
            reference: Any = weakref.ref(value)
        except TypeError:
            reference = value
            self._strong_objects[object_id] = value
        self._objects[object_id] = (reference, value_id)
        return value_id

    def _describe(self, value: Any) -> dict[str, Any]:
        result = {"type": f"{type(value).__module__}.{type(value).__qualname__}"}
        shape = getattr(value, "shape", None)
        if shape is not None:
            try:
                result["shape"] = [int(size) for size in shape]
            except (TypeError, ValueError):
                result["shape"] = str(shape)
        for key in ("dtype", "device"):
            attribute = getattr(value, key, None)
            if attribute is not None:
                result[key] = str(attribute)
        storage = _storage_key(value)
        if storage is not None:
            storage_id = self._storage_id(storage, value)
            result["storage_id"] = storage_id
            byte_range = _storage_byte_range(value, storage)
            if byte_range is not None:
                result["storage_byte_start"], result["storage_byte_stop"] = byte_range
        if isinstance(value, (bool, int, float, str)):
            result["value"] = value
        return result

    def _storage_id(self, key: tuple[str, int], value: Any) -> str:
        existing = self._storage_ids.get(key)
        if existing is not None:
            storage_id, references = existing
            live = {
                object_id: reference
                for object_id, reference in references.items()
                if reference() is not None
            }
            if live:
                object_id = id(value)
                reference = live.get(object_id)
                if reference is None or reference() is not value:
                    live[object_id] = weakref.ref(value)
                self._storage_ids[key] = (storage_id, live)
                return storage_id

        storage_id = f"s{self._next_storage:08d}"
        self._next_storage += 1
        self._storage_ids[key] = (storage_id, {id(value): weakref.ref(value)})
        return storage_id


def _leaves(value: Any, path: str = "value") -> Iterator[tuple[str, Any]]:
    if isinstance(value, LineageValue):
        yield path, value
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            yield from _leaves(value[key], f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _leaves(item, f"{path}[{index}]")
    elif value is not None:
        yield path, value


def _storage_key(value: Any) -> tuple[str, int] | None:
    storage_method = getattr(value, "untyped_storage", None)
    if callable(storage_method):
        try:
            storage = storage_method()
            return str(getattr(value, "device", "unknown")), int(storage.data_ptr())
        except (RuntimeError, TypeError):
            return None
    interface = getattr(value, "__array_interface__", None)
    if isinstance(interface, dict):
        owner = value
        seen = set()
        while id(owner) not in seen:
            seen.add(id(owner))
            base = getattr(owner, "base", None)
            if base is None or not isinstance(
                getattr(base, "__array_interface__", None), dict
            ):
                break
            owner = base
        data = owner.__array_interface__.get("data")
        if isinstance(data, tuple) and data:
            return "cpu", int(data[0])
    return None


def _storage_byte_range(
    value: Any, storage_key: tuple[str, int]
) -> tuple[int, int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        sizes = [int(size) for size in shape]
    except (TypeError, ValueError):
        return None
    if any(size == 0 for size in sizes):
        return 0, 0

    element_size = getattr(value, "element_size", None)
    stride = getattr(value, "stride", None)
    offset = getattr(value, "storage_offset", None)
    if callable(element_size) and callable(stride) and callable(offset):
        try:
            itemsize = int(element_size())
            strides = [int(item) for item in stride()]
            minimum = maximum = int(offset())
        except (RuntimeError, TypeError, ValueError):
            return None
        for size, item_stride in zip(sizes, strides):
            span = (size - 1) * item_stride
            minimum += min(0, span)
            maximum += max(0, span)
        return minimum * itemsize, (maximum + 1) * itemsize

    interface = getattr(value, "__array_interface__", None)
    if not isinstance(interface, dict):
        return None
    data = interface.get("data")
    typestr = interface.get("typestr")
    if not isinstance(data, tuple) or not data or not isinstance(typestr, str):
        return None
    try:
        itemsize = int(typestr[2:])
        pointer = int(data[0])
        base_pointer = storage_key[1]
        strides = interface.get("strides")
        if strides is None:
            byte_strides = []
            running = itemsize
            for size in reversed(sizes):
                byte_strides.append(running)
                running *= size
            byte_strides.reverse()
        else:
            byte_strides = [int(item) for item in strides]
    except (TypeError, ValueError):
        return None
    minimum = maximum = pointer - base_pointer
    for size, byte_stride in zip(sizes, byte_strides):
        span = (size - 1) * byte_stride
        minimum += min(0, span)
        maximum += max(0, span)
    return minimum, maximum + itemsize
