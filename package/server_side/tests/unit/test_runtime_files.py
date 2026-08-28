from __future__ import annotations

from pathlib import Path

import pytest

from runfold_server.errors import ApiError
from runfold_server.runtime.file_tools import create_file_tools
from runfold_server.runtime.files import FileWorkspaceService


def test_workspace_file_suite_reads_searches_and_lists(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    written = workspace.write_file("notes/report.txt", "alpha\nbeta alpha\n")
    info = workspace.file_info("notes/report.txt")
    read = workspace.read_file("notes/report.txt", start_line=2, end_line=2)
    listed = workspace.list_directory("notes")
    found = workspace.find_files("**/*.txt")
    searched = workspace.search_files(
        query="ALPHA",
        path=".",
        pattern="**/*.txt",
        case_sensitive=False,
    )

    assert written["size_bytes"] == len(b"alpha\nbeta alpha\n")
    assert info == {
        "path": "notes/report.txt",
        "size_bytes": 17,
        "characters": 17,
        "lines": 2,
        "recommend_chunked_read": False,
    }
    assert read["content"] == "beta alpha\n"
    assert listed["entries"][0]["path"] == "notes/report.txt"
    assert found["paths"] == ["notes/report.txt"]
    assert [match["line"] for match in searched["matches"]] == [1, 2]
    assert workspace.count_text("中文abc") == {"characters": 5}


def test_chunk_read_is_utf8_safe_and_append_is_retry_safe(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    original = "第一段🙂\nsecond\n"
    written = workspace.write_file("stream.txt", original)
    offset = 0
    pieces: list[str] = []
    while True:
        chunk = workspace.read_file_chunk(
            "stream.txt",
            offset_bytes=offset,
            chunk_bytes=5,
        )
        pieces.append(str(chunk["content"]))
        offset = int(chunk["next_offset_bytes"])
        if chunk["eof"]:
            break
    assert "".join(pieces) == original

    appended = workspace.append_file(
        "stream.txt",
        text="tail",
        expected_size_bytes=int(written["size_bytes"]),
    )
    assert appended["next_offset_bytes"] == len((original + "tail").encode())
    with pytest.raises(ApiError) as conflict:
        workspace.append_file(
            "stream.txt",
            text="duplicate",
            expected_size_bytes=int(written["size_bytes"]),
        )
    assert conflict.value.code == "file_size_conflict"
    assert workspace.read_file("stream.txt", start_line=None, end_line=None)[
        "content"
    ] == original + "tail"


def test_apply_patch_prevalidates_all_operations_and_preserves_newlines(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    workspace.write_file("existing.txt", "one\r\ntwo\r\n")

    with pytest.raises(ApiError) as mismatch:
        workspace.apply_patch(
            """*** Begin Patch
*** Add File: must-not-exist.txt
+new
*** Update File: existing.txt
@@
-missing
+replacement
*** End Patch"""
        )
    assert mismatch.value.code == "patch_context_mismatch"
    assert not (tmp_path / "agent_work" / "user-1" / "must-not-exist.txt").exists()

    result = workspace.apply_patch(
        """*** Begin Patch
*** Add File: added.txt
+created
*** Update File: existing.txt
@@
 one
-two
+updated
*** End Patch"""
    )
    assert [item["action"] for item in result["applied"]] == ["add", "update"]
    assert (tmp_path / "agent_work" / "user-1" / "existing.txt").read_bytes() == (
        b"one\r\nupdated\r\n"
    )

    workspace.apply_patch(
        """*** Begin Patch
*** Delete File: added.txt
*** End Patch"""
    )
    assert not (tmp_path / "agent_work" / "user-1" / "added.txt").exists()


@pytest.mark.parametrize("path", ["../escape.txt", "/absolute.txt", "C:\\escape.txt"])
def test_workspace_rejects_paths_outside_agent_work(tmp_path: Path, path: str) -> None:
    workspace = _workspace(tmp_path)

    with pytest.raises(ApiError) as captured:
        workspace.write_file(path, "forbidden")

    assert captured.value.code == "unsafe_file_path"


def test_workspace_rejects_symbolic_link_escape(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    user_root = tmp_path / "agent_work" / "user-1"
    outside = tmp_path / "outside"
    outside.mkdir()
    link = user_root / "escape-link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symbolic links are unavailable on this host")

    with pytest.raises(ApiError) as captured:
        workspace.write_file("escape-link/forbidden.txt", "forbidden")

    assert captured.value.code == "unsafe_file_path"


def test_every_agent_receives_full_file_tool_set(tmp_path: Path) -> None:
    tools = create_file_tools(_workspace(tmp_path))

    assert {tool.name for tool in tools} == {
        "write_file",
        "read_file",
        "read_files",
        "list_directory",
        "find_files",
        "search_files",
        "file_info",
        "count_text",
        "read_file_chunk",
        "append_file",
        "apply_patch",
    }
    assert "replace_content" not in {tool.name for tool in tools}


def _workspace(tmp_path: Path):
    root = tmp_path / "agent_work"
    root.mkdir(exist_ok=True)
    return FileWorkspaceService(root).for_user("user-1")
