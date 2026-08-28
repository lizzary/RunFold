from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from runfold_server.errors import ApiError
from runfold_server.runtime.files import (
    DEFAULT_CHUNK_BYTES,
    MAX_CHUNK_BYTES,
    AgentFileWorkspace,
)


class _FileToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class WriteFileInput(_FileToolInput):
    path: str = Field(min_length=1)
    content: str


class ReadFileInput(_FileToolInput):
    path: str = Field(min_length=1)
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)


class ReadFilesInput(_FileToolInput):
    paths: list[str] = Field(min_length=1)


class ListDirectoryInput(_FileToolInput):
    path: str = "."


class FindFilesInput(_FileToolInput):
    pattern: str = Field(min_length=1)


class SearchFilesInput(_FileToolInput):
    query: str = Field(min_length=1)
    path: str = "."
    pattern: str = "**/*"
    case_sensitive: bool = False


class FileInfoInput(_FileToolInput):
    path: str = Field(min_length=1)


class CountTextInput(_FileToolInput):
    text: str


class ReadFileChunkInput(_FileToolInput):
    path: str = Field(min_length=1)
    offset_bytes: int = Field(default=0, ge=0)
    chunk_bytes: int = Field(default=DEFAULT_CHUNK_BYTES, ge=1, le=MAX_CHUNK_BYTES)


class AppendFileInput(_FileToolInput):
    path: str = Field(min_length=1)
    text: str
    expected_size_bytes: int = Field(ge=0)


class ApplyPatchInput(_FileToolInput):
    patch: str = Field(min_length=1)


def create_file_tools(workspace: AgentFileWorkspace) -> tuple[BaseTool, ...]:
    async def call(method: Callable[..., dict[str, object]], *args: Any, **kwargs: Any) -> str:
        try:
            result = await asyncio.to_thread(method, *args, **kwargs)
        except ApiError as error:
            return _json(
                {"status": "error", "code": error.code, "message": error.message}
            )
        return _json({"status": "completed", **result})

    async def write_file(path: str, content: str) -> str:
        """Atomically save UTF-8 text inside the shared agent_work workspace."""
        return await call(workspace.write_file, path, content)

    async def read_file(
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        """Read a small UTF-8 file or a known inclusive line range."""
        return await call(
            workspace.read_file,
            path,
            start_line=start_line,
            end_line=end_line,
        )

    async def read_files(paths: list[str]) -> str:
        """Read several known-small UTF-8 files in one operation."""
        return await call(workspace.read_files, paths)

    async def list_directory(path: str = ".") -> str:
        """List one directory in agent_work with entry types and file sizes."""
        return await call(workspace.list_directory, path)

    async def find_files(pattern: str) -> str:
        """Find files by a relative glob pattern without leaving agent_work."""
        return await call(workspace.find_files, pattern)

    async def search_files(
        query: str,
        path: str = ".",
        pattern: str = "**/*",
        case_sensitive: bool = False,
    ) -> str:
        """Search UTF-8 file bodies and return matching paths, lines, and text."""
        return await call(
            workspace.search_files,
            query=query,
            path=path,
            pattern=pattern,
            case_sensitive=case_sensitive,
        )

    async def file_info(path: str) -> str:
        """Return byte size, character count, line count, and chunk-read advice."""
        return await call(workspace.file_info, path)

    async def count_text(text: str) -> str:
        """Count Unicode characters in supplied text."""
        return await call(workspace.count_text, text)

    async def read_file_chunk(
        path: str,
        offset_bytes: int = 0,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    ) -> str:
        """Sequentially read a UTF-8 file by byte offset with next offset and EOF."""
        return await call(
            workspace.read_file_chunk,
            path,
            offset_bytes=offset_bytes,
            chunk_bytes=chunk_bytes,
        )

    async def append_file(
        path: str,
        text: str,
        expected_size_bytes: int,
    ) -> str:
        """Append UTF-8 text only when the current byte size matches exactly."""
        return await call(
            workspace.append_file,
            path,
            text=text,
            expected_size_bytes=expected_size_bytes,
        )

    async def apply_patch(patch: str) -> str:
        """Validate and apply a complete Codex-style patch inside agent_work."""
        return await call(workspace.apply_patch, patch)

    return (
        StructuredTool.from_function(
            coroutine=write_file,
            name="write_file",
            description=(
                "Atomically save UTF-8 text in agent_work. Prefer apply_patch for modifying "
                "existing files."
            ),
            args_schema=WriteFileInput,
        ),
        StructuredTool.from_function(
            coroutine=read_file,
            name="read_file",
            description=(
                "Read a small file or known line range. Preflight unknown files with file_info; "
                "use read_file_chunk for large files."
            ),
            args_schema=ReadFileInput,
        ),
        StructuredTool.from_function(
            coroutine=read_files,
            name="read_files",
            description="Batch-read only files already known to be small.",
            args_schema=ReadFilesInput,
        ),
        StructuredTool.from_function(
            coroutine=list_directory,
            name="list_directory",
            description=(
                "List a directory. When a path appears misspelled or missing, inspect its "
                "parent directory for a close match before giving up."
            ),
            args_schema=ListDirectoryInput,
        ),
        StructuredTool.from_function(
            coroutine=find_files,
            name="find_files",
            description="Find workspace files using a relative glob pattern.",
            args_schema=FindFilesInput,
        ),
        StructuredTool.from_function(
            coroutine=search_files,
            name="search_files",
            description="Search UTF-8 text bodies under agent_work.",
            args_schema=SearchFilesInput,
        ),
        StructuredTool.from_function(
            coroutine=file_info,
            name="file_info",
            description=(
                "Preflight a file and return bytes, characters, lines, and whether sequential "
                "chunk reading is recommended."
            ),
            args_schema=FileInfoInput,
        ),
        StructuredTool.from_function(
            coroutine=count_text,
            name="count_text",
            description="Return the Unicode character count of supplied text.",
            args_schema=CountTextInput,
        ),
        StructuredTool.from_function(
            coroutine=read_file_chunk,
            name="read_file_chunk",
            description=(
                "Read large UTF-8 files sequentially from offset 0 to eof=true. Never request "
                "all chunks in parallel."
            ),
            args_schema=ReadFileChunkInput,
        ),
        StructuredTool.from_function(
            coroutine=append_file,
            name="append_file",
            description=(
                "Append without rewriting. Pass the exact current expected_size_bytes and reuse "
                "the returned next_offset_bytes for the next serial append."
            ),
            args_schema=AppendFileInput,
        ),
        StructuredTool.from_function(
            coroutine=apply_patch,
            name="apply_patch",
            description=(
                "Apply a fully validated *** Begin Patch/*** End Patch document supporting Add, "
                "Update with exact @@ context, and Delete."
            ),
            args_schema=ApplyPatchInput,
        ),
    )


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
