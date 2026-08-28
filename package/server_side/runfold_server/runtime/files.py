from __future__ import annotations

import codecs
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from runfold_server.errors import ApiError

DEFAULT_CHUNK_BYTES = 8 * 1024
MAX_CHUNK_BYTES = 16 * 1024

_PATCH_HEADER = re.compile(r"^\*\*\* (Add|Update|Delete) File: (.+)$")


@dataclass(frozen=True, slots=True)
class _PatchOperation:
    action: str
    path: str
    body: tuple[str, ...]


class FileWorkspaceService:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve(strict=True)
        self._lock = threading.RLock()

    def for_user(self, user_id: str) -> AgentFileWorkspace:
        if not user_id or any(character in user_id for character in "/\\"):
            raise ValueError("Invalid workspace user")
        root = (self._root / user_id).resolve(strict=False)
        if not root.is_relative_to(self._root):
            raise ValueError("Invalid workspace user")
        root.mkdir(exist_ok=True)
        return AgentFileWorkspace(root, self._lock)


class AgentFileWorkspace:
    def __init__(self, root: Path, lock: threading.RLock) -> None:
        self._root = root.resolve(strict=True)
        self._lock = lock

    def write_file(self, path: str, content: str) -> dict[str, object]:
        target = self._file_path(path, must_exist=False)
        with self._lock:
            self._atomic_write(target, content.encode("utf-8"))
        return self._write_result(target)

    def read_file(
        self,
        path: str,
        *,
        start_line: int | None,
        end_line: int | None,
    ) -> dict[str, object]:
        target = self._file_path(path, must_exist=True)
        text = self._read_utf8(target)
        lines = text.splitlines(keepends=True)
        start = 1 if start_line is None else start_line
        end = len(lines) if end_line is None else end_line
        if start < 1 or end < start:
            raise _file_error("invalid_line_range", "File line range is invalid")
        selected = "".join(lines[start - 1 : end])
        return {
            "path": self._relative(target),
            "start_line": start,
            "end_line": min(end, len(lines)),
            "total_lines": len(lines),
            "content": selected,
        }

    def read_files(self, paths: list[str]) -> dict[str, object]:
        return {
            "files": [
                self.read_file(path, start_line=None, end_line=None) for path in paths
            ]
        }

    def list_directory(self, path: str) -> dict[str, object]:
        directory = self._directory_path(path, must_exist=True)
        entries: list[dict[str, object]] = []
        for child in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
            resolved = child.resolve(strict=True)
            self._require_inside(resolved)
            if resolved.is_dir():
                entries.append({"path": self._relative(resolved), "type": "directory"})
            elif resolved.is_file():
                entries.append(
                    {
                        "path": self._relative(resolved),
                        "type": "file",
                        "size_bytes": resolved.stat().st_size,
                    }
                )
        return {"path": self._relative(directory), "entries": entries}

    def find_files(self, pattern: str) -> dict[str, object]:
        self._validate_glob(pattern)
        matches: list[str] = []
        for candidate in self._root.glob(pattern):
            resolved = candidate.resolve(strict=True)
            self._require_inside(resolved)
            if resolved.is_file():
                matches.append(self._relative(resolved))
        return {"pattern": pattern, "paths": sorted(matches)}

    def search_files(
        self,
        *,
        query: str,
        path: str,
        pattern: str,
        case_sensitive: bool,
    ) -> dict[str, object]:
        if not query:
            raise _file_error("invalid_search_query", "Search query must not be empty")
        directory = self._directory_path(path, must_exist=True)
        self._validate_glob(pattern)
        needle = query if case_sensitive else query.casefold()
        matches: list[dict[str, object]] = []
        for candidate in directory.glob(pattern):
            if not candidate.is_file():
                continue
            resolved = candidate.resolve(strict=True)
            self._require_inside(resolved)
            try:
                text = resolved.read_text(encoding="utf-8")
            except UnicodeError:
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                haystack = line if case_sensitive else line.casefold()
                if needle in haystack:
                    matches.append(
                        {
                            "path": self._relative(resolved),
                            "line": line_number,
                            "text": line,
                        }
                    )
        return {"query": query, "matches": matches}

    def file_info(self, path: str) -> dict[str, object]:
        target = self._file_path(path, must_exist=True)
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        characters = 0
        newlines = 0
        last_character = ""
        try:
            with target.open("rb") as handle:
                while chunk := handle.read(MAX_CHUNK_BYTES):
                    text = decoder.decode(chunk)
                    characters += len(text)
                    newlines += text.count("\n")
                    if text:
                        last_character = text[-1]
                tail = decoder.decode(b"", final=True)
        except UnicodeError as error:
            raise _file_error("file_not_utf8", "File is not valid UTF-8 text") from error
        characters += len(tail)
        newlines += tail.count("\n")
        if tail:
            last_character = tail[-1]
        size = target.stat().st_size
        lines = 0 if characters == 0 else newlines + int(last_character != "\n")
        return {
            "path": self._relative(target),
            "size_bytes": size,
            "characters": characters,
            "lines": lines,
            "recommend_chunked_read": size > MAX_CHUNK_BYTES,
        }

    @staticmethod
    def count_text(text: str) -> dict[str, int]:
        return {"characters": len(text)}

    def read_file_chunk(
        self,
        path: str,
        *,
        offset_bytes: int,
        chunk_bytes: int,
    ) -> dict[str, object]:
        if offset_bytes < 0 or chunk_bytes < 1 or chunk_bytes > MAX_CHUNK_BYTES:
            raise _file_error("invalid_chunk_range", "File chunk range is invalid")
        target = self._file_path(path, must_exist=True)
        size = target.stat().st_size
        if offset_bytes > size:
            raise _file_error("invalid_chunk_range", "File chunk offset is past EOF")
        with target.open("rb") as handle:
            handle.seek(offset_bytes)
            raw = handle.read(chunk_bytes)
        consumed, text = _decode_complete_utf8_prefix(
            raw,
            offset_bytes + len(raw) == size,
        )
        next_offset = offset_bytes + consumed
        return {
            "path": self._relative(target),
            "offset_bytes": offset_bytes,
            "next_offset_bytes": next_offset,
            "eof": next_offset == size,
            "content": text,
        }

    def append_file(
        self,
        path: str,
        *,
        text: str,
        expected_size_bytes: int,
    ) -> dict[str, object]:
        if expected_size_bytes < 0:
            raise _file_error("invalid_expected_size", "Expected file size is invalid")
        target = self._file_path(path, must_exist=True)
        encoded = text.encode("utf-8")
        with self._lock:
            self._validate_utf8(target)
            current_size = target.stat().st_size
            if current_size != expected_size_bytes:
                raise _file_error(
                    "file_size_conflict",
                    "File size changed; use the latest next_offset_bytes",
                )
            try:
                with target.open("ab") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError as error:
                raise _file_error("file_write_failed", "File append failed") from error
        return {
            "path": self._relative(target),
            "next_offset_bytes": expected_size_bytes + len(encoded),
        }

    def apply_patch(self, patch: str) -> dict[str, object]:
        operations = _parse_patch(patch)
        planned_writes: list[tuple[Path, bytes]] = []
        planned_deletes: list[Path] = []
        seen: set[Path] = set()
        with self._lock:
            for operation in operations:
                target = self._file_path(
                    operation.path,
                    must_exist=operation.action in {"Update", "Delete"},
                )
                if target in seen:
                    raise _file_error(
                        "duplicate_patch_path",
                        "A patch may operate on each path only once",
                    )
                seen.add(target)
                if operation.action == "Add":
                    if target.exists():
                        raise _file_error("file_exists", "Patch add target already exists")
                    content = _added_content(operation.body)
                    planned_writes.append((target, content.encode("utf-8")))
                elif operation.action == "Update":
                    planned_writes.append((target, _updated_content(target, operation.body)))
                else:
                    if operation.body:
                        raise _file_error(
                            "invalid_patch",
                            "Patch delete sections must not have body lines",
                        )
                    planned_deletes.append(target)

            staged: list[tuple[Path, Path]] = []
            try:
                for target, content in planned_writes:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    staged.append((target, self._stage_write(target, content)))
                for target, temporary in staged:
                    os.replace(temporary, target)
                for target in planned_deletes:
                    target.unlink()
            except OSError as error:
                raise _file_error("patch_apply_failed", "Patch could not be applied") from error
            finally:
                for _, temporary in staged:
                    if temporary.exists():
                        temporary.unlink()
        return {
            "applied": [
                {"action": operation.action.lower(), "path": operation.path}
                for operation in operations
            ]
        }

    def _file_path(self, path: str, *, must_exist: bool) -> Path:
        target = self._safe_path(path)
        if must_exist and (not target.exists() or not target.is_file()):
            raise _file_error("file_not_found", "File was not found")
        if target.exists() and not target.is_file():
            raise _file_error("not_a_file", "Path is not a regular file")
        return target

    def _directory_path(self, path: str, *, must_exist: bool) -> Path:
        target = self._safe_path(path, allow_root=True)
        if must_exist and (not target.exists() or not target.is_dir()):
            raise _file_error("directory_not_found", "Directory was not found")
        return target

    def _safe_path(self, raw: str, *, allow_root: bool = False) -> Path:
        if not isinstance(raw, str) or not raw.strip():
            raise _file_error("invalid_file_path", "File path is invalid")
        normalized = raw.replace("\\", "/")
        posix = PurePosixPath(normalized)
        windows = PureWindowsPath(raw)
        if (
            posix.is_absolute()
            or windows.is_absolute()
            or windows.drive
            or ".." in posix.parts
        ):
            raise _file_error("unsafe_file_path", "File path must stay in agent_work")
        parts = tuple(part for part in posix.parts if part not in {"", "."})
        if not parts and not allow_root:
            raise _file_error("invalid_file_path", "File path is invalid")
        target = self._root.joinpath(*parts).resolve(strict=False)
        self._require_inside(target)
        return target

    def _require_inside(self, path: Path) -> None:
        if path != self._root and not path.is_relative_to(self._root):
            raise _file_error("unsafe_file_path", "File path escapes agent_work")

    def _relative(self, path: Path) -> str:
        if path == self._root:
            return "."
        return path.relative_to(self._root).as_posix()

    def _read_utf8(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeError as error:
            raise _file_error("file_not_utf8", "File is not valid UTF-8 text") from error
        except OSError as error:
            raise _file_error("file_read_failed", "File could not be read") from error

    def _validate_utf8(self, path: Path) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        try:
            with path.open("rb") as handle:
                while chunk := handle.read(MAX_CHUNK_BYTES):
                    decoder.decode(chunk)
                decoder.decode(b"", final=True)
        except UnicodeError as error:
            raise _file_error("file_not_utf8", "File is not valid UTF-8 text") from error

    def _atomic_write(self, path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._stage_write(path, content)
        try:
            os.replace(temporary, path)
        except OSError as error:
            raise _file_error("file_write_failed", "File could not be written") from error
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _stage_write(path: Path, content: bytes) -> Path:
        descriptor, name = tempfile.mkstemp(prefix=".runfold-", dir=path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return temporary

    def _write_result(self, path: Path) -> dict[str, object]:
        return {"path": self._relative(path), "size_bytes": path.stat().st_size}

    @staticmethod
    def _validate_glob(pattern: str) -> None:
        if (
            not pattern
            or PurePosixPath(pattern).is_absolute()
            or PureWindowsPath(pattern).is_absolute()
            or PureWindowsPath(pattern).drive
            or ".." in PurePosixPath(pattern.replace("\\", "/")).parts
        ):
            raise _file_error("unsafe_file_pattern", "File pattern is unsafe")


def _decode_complete_utf8_prefix(raw: bytes, at_eof: bool) -> tuple[int, str]:
    if not raw:
        return 0, ""
    if at_eof:
        try:
            return len(raw), raw.decode("utf-8")
        except UnicodeError as error:
            raise _file_error("file_not_utf8", "File is not valid UTF-8 text") from error
    for trimmed in range(0, min(4, len(raw))):
        candidate = raw if trimmed == 0 else raw[:-trimmed]
        try:
            return len(candidate), candidate.decode("utf-8")
        except UnicodeDecodeError as error:
            if (
                error.start != len(candidate) - (error.end - error.start)
                and error.reason != "unexpected end of data"
            ):
                raise _file_error(
                    "invalid_chunk_offset",
                    "File chunk offset is not a UTF-8 boundary",
                ) from error
    raise _file_error("file_not_utf8", "File is not valid UTF-8 text")


def _parse_patch(patch: str) -> tuple[_PatchOperation, ...]:
    lines = patch.splitlines()
    if len(lines) < 2 or lines[0] != "*** Begin Patch" or lines[-1] != "*** End Patch":
        raise _file_error("invalid_patch", "Patch must have Begin Patch and End Patch")
    operations: list[_PatchOperation] = []
    index = 1
    while index < len(lines) - 1:
        match = _PATCH_HEADER.fullmatch(lines[index])
        if match is None:
            raise _file_error("invalid_patch", "Patch section header is invalid")
        action, path = match.groups()
        index += 1
        body: list[str] = []
        while index < len(lines) - 1 and _PATCH_HEADER.fullmatch(lines[index]) is None:
            body.append(lines[index])
            index += 1
        operations.append(_PatchOperation(action=action, path=path.strip(), body=tuple(body)))
    if not operations:
        raise _file_error("invalid_patch", "Patch has no operations")
    return tuple(operations)


def _added_content(lines: tuple[str, ...]) -> str:
    if any(not line.startswith("+") for line in lines):
        raise _file_error("invalid_patch", "Added file lines must start with +")
    return "\n".join(line[1:] for line in lines) + ("\n" if lines else "")


def _updated_content(path: Path, lines: tuple[str, ...]) -> bytes:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeError as error:
        raise _file_error("file_not_utf8", "Patch target is not UTF-8") from error
    newline = "\r\n" if b"\r\n" in raw else "\n"
    trailing_newline = text.endswith(("\n", "\r"))
    current = text.splitlines()
    hunks = _patch_hunks(lines)
    cursor = 0
    for hunk in hunks:
        before = [line[1:] for line in hunk if line.startswith((" ", "-"))]
        after = [line[1:] for line in hunk if line.startswith((" ", "+"))]
        if not before:
            raise _file_error("invalid_patch", "Update hunks require exact context")
        matches = [
            index
            for index in range(cursor, len(current) - len(before) + 1)
            if current[index : index + len(before)] == before
        ]
        if not matches:
            raise _file_error("patch_context_mismatch", "Patch context did not match")
        if len(matches) != 1:
            raise _file_error("patch_context_ambiguous", "Patch context is ambiguous")
        index = matches[0]
        current[index : index + len(before)] = after
        cursor = index + len(after)
    updated = newline.join(current)
    if trailing_newline:
        updated += newline
    return updated.encode("utf-8")


def _patch_hunks(lines: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    hunks: list[tuple[str, ...]] = []
    current: list[str] | None = None
    for line in lines:
        if line.startswith("@@"):
            if current is not None:
                hunks.append(tuple(current))
            current = []
            continue
        if current is None or not line.startswith((" ", "+", "-")):
            raise _file_error("invalid_patch", "Patch update hunk is invalid")
        current.append(line)
    if current is not None:
        hunks.append(tuple(current))
    if not hunks:
        raise _file_error("invalid_patch", "Patch update has no hunks")
    return tuple(hunks)


def _file_error(code: str, message: str) -> ApiError:
    return ApiError(422, code, message)
