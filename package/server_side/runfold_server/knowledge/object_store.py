from __future__ import annotations

import hashlib
import os
import shutil
import unicodedata
import uuid
from pathlib import Path
from typing import Protocol

from runfold_server.errors import ApiError, StartupError
from runfold_server.knowledge.extractors import extract_text, media_type_for, stored_media_type
from runfold_server.knowledge.models import StagedDocument

_STREAM_CHUNK = 64 * 1024


class AsyncReadable(Protocol):
    async def read(self, size: int = -1) -> bytes: ...


class ObjectStore:
    def __init__(
        self,
        *,
        objects: Path,
        staging: Path,
        upload_max_bytes: int,
        extract_max_characters: int,
        pdf_max_pages: int,
        docx_max_uncompressed_bytes: int,
    ) -> None:
        self._objects = objects.resolve(strict=True)
        self._staging = staging.resolve(strict=True)
        self._upload_max_bytes = upload_max_bytes
        self._extract_max_characters = extract_max_characters
        self._pdf_max_pages = pdf_max_pages
        self._docx_max_uncompressed_bytes = docx_max_uncompressed_bytes

    async def stage_upload(
        self, stream: AsyncReadable, original_filename: str
    ) -> StagedDocument:
        filename = _safe_filename(original_filename)
        operation_id, directory, source = self._new_stage()
        digest = hashlib.sha256()
        size = 0
        try:
            with source.open("xb") as target:
                while True:
                    value = await stream.read(_STREAM_CHUNK)
                    if not value:
                        break
                    size += len(value)
                    if size > self._upload_max_bytes:
                        raise ApiError(413, "upload_too_large", "Uploaded file exceeds the limit")
                    digest.update(value)
                    target.write(value)
            if size == 0:
                raise ApiError(422, "empty_document", "Document is empty")
            return self._finish_stage(
                operation_id=operation_id,
                directory=directory,
                source=source,
                filename=filename,
                size=size,
                content_hash=digest.hexdigest(),
            )
        except Exception:
            self.cleanup_stage(directory)
            raise

    def stage_text(self, text: str, original_filename: str) -> StagedDocument:
        filename = _safe_filename(original_filename)
        if Path(filename).suffix.lower() not in {".txt", ".md"}:
            raise ApiError(415, "text_replace_not_supported", "Text replacement requires txt or md")
        value = text.encode("utf-8")
        if len(value) > self._upload_max_bytes:
            raise ApiError(413, "upload_too_large", "Uploaded file exceeds the limit")
        operation_id, directory, source = self._new_stage()
        try:
            source.write_bytes(value)
            return self._finish_stage(
                operation_id=operation_id,
                directory=directory,
                source=source,
                filename=filename,
                size=len(value),
                content_hash=hashlib.sha256(value).hexdigest(),
            )
        except Exception:
            self.cleanup_stage(directory)
            raise

    def stage_existing(self, document_id: str, original_filename: str) -> StagedDocument:
        source_path = self.source_path(document_id)
        if not source_path.is_file():
            raise ApiError(409, "document_source_missing", "Document source is unavailable")
        operation_id, directory, staged_source = self._new_stage()
        digest = hashlib.sha256()
        size = 0
        try:
            with source_path.open("rb") as source, staged_source.open("xb") as target:
                while value := source.read(_STREAM_CHUNK):
                    size += len(value)
                    digest.update(value)
                    target.write(value)
            return self._finish_stage(
                operation_id=operation_id,
                directory=directory,
                source=staged_source,
                filename=original_filename,
                size=size,
                content_hash=digest.hexdigest(),
            )
        except Exception:
            self.cleanup_stage(directory)
            raise

    def commit(self, staged: StagedDocument, document_id: str, *, source: bool) -> None:
        directory = self.document_directory(document_id)
        directory.mkdir(exist_ok=True)
        if source:
            os.replace(staged.source, self.source_path(document_id))
        os.replace(staged.extracted, self.extracted_path(document_id))

    def cleanup_stage(self, directory: Path) -> None:
        resolved = directory.resolve(strict=False)
        if resolved == self._staging or not resolved.is_relative_to(self._staging):
            raise StartupError("unsafe_storage_path", "Staging path escapes the data directory")
        if resolved.exists():
            shutil.rmtree(resolved)

    def clear_staging(self) -> None:
        for child in self._staging.iterdir():
            resolved = child.resolve(strict=False)
            if not resolved.is_relative_to(self._staging):
                raise StartupError("unsafe_storage_path", "Staging path escapes the data directory")
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    def document_directory(self, document_id: str) -> Path:
        _validate_document_id(document_id)
        return self._safe_object_path(document_id)

    def source_path(self, document_id: str) -> Path:
        return self._safe_object_path(document_id, "source")

    def extracted_path(self, document_id: str) -> Path:
        return self._safe_object_path(document_id, "extracted.txt")

    def read_source(self, document_id: str) -> bytes:
        try:
            return self.source_path(document_id).read_bytes()
        except OSError as error:
            raise ApiError(
                409, "document_source_missing", "Document source is unavailable"
            ) from error

    def read_extracted(self, document_id: str) -> str:
        try:
            return self.extracted_path(document_id).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ApiError(
                409, "document_text_unavailable", "Extracted text is unavailable"
            ) from error

    def delete_extracted(self, document_id: str) -> None:
        self.extracted_path(document_id).unlink(missing_ok=True)

    def delete_document(self, document_id: str) -> None:
        directory = self.document_directory(document_id)
        if directory.exists():
            shutil.rmtree(directory)

    def source_metadata(self, document_id: str, original_filename: str) -> tuple[int, str, str]:
        source = self.source_path(document_id)
        digest = hashlib.sha256()
        size = 0
        with source.open("rb") as value:
            while part := value.read(_STREAM_CHUNK):
                size += len(part)
                digest.update(part)
        return size, digest.hexdigest(), stored_media_type(source, original_filename)

    def _new_stage(self) -> tuple[str, Path, Path]:
        operation_id = str(uuid.uuid4())
        directory = (self._staging / operation_id).resolve(strict=False)
        if not directory.is_relative_to(self._staging):
            raise StartupError("unsafe_storage_path", "Staging path escapes the data directory")
        directory.mkdir()
        return operation_id, directory, directory / "source"

    def _finish_stage(
        self,
        *,
        operation_id: str,
        directory: Path,
        source: Path,
        filename: str,
        size: int,
        content_hash: str,
    ) -> StagedDocument:
        media_type = media_type_for(source, filename)
        text = extract_text(
            source,
            media_type=media_type,
            max_characters=self._extract_max_characters,
            pdf_max_pages=self._pdf_max_pages,
            docx_max_uncompressed_bytes=self._docx_max_uncompressed_bytes,
        )
        extracted = directory / "extracted.txt"
        extracted.write_text(text, encoding="utf-8", newline="\n")
        return StagedDocument(
            operation_id=operation_id,
            directory=directory,
            source=source,
            extracted=extracted,
            original_filename=filename,
            media_type=media_type,
            byte_size=size,
            content_hash=content_hash,
            text=text,
        )

    def _safe_object_path(self, *parts: str) -> Path:
        path = self._objects.joinpath(*parts).resolve(strict=False)
        if path == self._objects or not path.is_relative_to(self._objects):
            raise StartupError("unsafe_storage_path", "Object path escapes the data directory")
        return path


def _safe_filename(value: str) -> str:
    filename = value.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if (
        not filename
        or len(filename) > 255
        or filename in {".", ".."}
        or any(unicodedata.category(character).startswith("C") for character in filename)
    ):
        raise ApiError(422, "invalid_filename", "Filename is invalid")
    return filename


def _validate_document_id(document_id: str) -> None:
    try:
        parsed = uuid.UUID(document_id)
    except ValueError as error:
        raise ApiError(404, "document_not_found", "Document not found") from error
    if str(parsed) != document_id:
        raise ApiError(404, "document_not_found", "Document not found")
