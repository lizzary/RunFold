from __future__ import annotations

import logging
from collections.abc import Callable

import httpx
from fastapi import FastAPI

from runfold_server.access_control.audit import AuditRepository
from runfold_server.access_control.authorization import AuthorizationService
from runfold_server.access_control.repository import AccessControlRepository
from runfold_server.access_control.service import AccessControlService
from runfold_server.config import Settings, load_settings
from runfold_server.http.app import create_app
from runfold_server.identity.passwords import Argon2PasswordHasher
from runfold_server.identity.repository import IdentityRepository
from runfold_server.identity.service import IdentityService
from runfold_server.knowledge.access_policy import KnowledgeAccessPolicy
from runfold_server.knowledge.lance_index import (
    IndexConfiguration,
    LanceIndex,
    initialize_index,
)
from runfold_server.knowledge.object_store import ObjectStore
from runfold_server.knowledge.reconciliation import ReconciliationService
from runfold_server.knowledge.repository import KnowledgeRepository
from runfold_server.knowledge.service import KnowledgeService
from runfold_server.llm.openai_embeddings import (
    OpenAIEmbeddingsClient,
    embedding_identity,
)
from runfold_server.storage.sqlite import (
    DataPaths,
    check_database_ready,
    check_local_directories,
    initialize_data_paths,
    initialize_database,
)
from runfold_server.usage.repository import UsageRepository
from runfold_server.usage.service import UsageService

_LOGGER = logging.getLogger("runfold_server.bootstrap")


def bootstrap(settings: Settings | None = None) -> FastAPI:
    current_settings = load_settings() if settings is None else settings
    paths = initialize_data_paths(current_settings.data_dir)
    initialize_database(paths.database)
    audit_repository = AuditRepository()
    identity_repository = IdentityRepository()
    access_repository = AccessControlRepository()
    knowledge_repository = KnowledgeRepository()
    password_hasher = Argon2PasswordHasher()
    identity_service = IdentityService(
        database_path=paths.database,
        repository=identity_repository,
        password_hasher=password_hasher,
        audit=audit_repository,
        session_ttl_seconds=current_settings.session_ttl_seconds,
    )
    authorization_service = AuthorizationService(paths.database, access_repository)
    object_store = ObjectStore(
        objects=paths.objects,
        staging=paths.staging,
        upload_max_bytes=current_settings.upload_max_bytes,
        extract_max_characters=current_settings.extract_max_characters,
        pdf_max_pages=current_settings.pdf_max_pages,
        docx_max_uncompressed_bytes=current_settings.docx_max_uncompressed_bytes,
    )
    index = initialize_index(
        database_path=paths.database,
        lance_path=paths.lance,
        configuration=IndexConfiguration(
            embedding_identity=embedding_identity(
                current_settings.openai_base_url,
                current_settings.embedding_model,
                current_settings.embedding_dimensions,
            ),
            model=current_settings.embedding_model,
            dimensions=current_settings.embedding_dimensions,
            chunk_size=current_settings.chunk_size,
            chunk_overlap=current_settings.chunk_overlap,
        ),
    )
    ReconciliationService(
        database_path=paths.database,
        repository=knowledge_repository,
        objects=object_store,
        index=index,
    ).run()
    access_control_service = AccessControlService(
        database_path=paths.database,
        identity=identity_service,
        identity_repository=identity_repository,
        repository=access_repository,
        authorization=authorization_service,
        audit=audit_repository,
    )
    access_control_service.ensure_administrator_exists(
        current_settings.bootstrap_admin_username,
        current_settings.bootstrap_admin_password,
    )
    if current_settings.bootstrap_admin_password is not None:
        _LOGGER.warning("bootstrap_admin_credentials_should_be_removed")
    usage_service = UsageService(
        database_path=paths.database,
        repository=UsageRepository(),
        default_max_documents=current_settings.default_max_documents,
        default_max_storage_bytes=current_settings.default_max_storage_bytes,
        default_monthly_embedding_tokens=(
            current_settings.default_monthly_embedding_tokens
        ),
    )
    http_client = httpx.AsyncClient(timeout=current_settings.llm_timeout_seconds)
    embeddings = OpenAIEmbeddingsClient(
        http_client=http_client,
        base_url=current_settings.openai_base_url,
        api_key=current_settings.openai_api_key,
        model=current_settings.embedding_model,
        dimensions=current_settings.embedding_dimensions,
        max_retries=current_settings.llm_max_retries,
    )
    knowledge_service = KnowledgeService(
        database_path=paths.database,
        identity=identity_service,
        authorization=authorization_service,
        repository=knowledge_repository,
        access_policy=KnowledgeAccessPolicy(knowledge_repository, audit_repository),
        audit=audit_repository,
        objects=object_store,
        index=index,
        embeddings=embeddings,
        usage=usage_service,
        chunk_size=current_settings.chunk_size,
        chunk_overlap=current_settings.chunk_overlap,
        embed_batch_size=current_settings.embed_batch_size,
    )
    readiness_check = _readiness_check(paths, index)
    if not readiness_check():
        raise RuntimeError("Local infrastructure did not become ready")
    return create_app(
        allowed_origins=current_settings.allowed_origins,
        readiness_check=readiness_check,
        identity_service=identity_service,
        access_control_service=access_control_service,
        knowledge_service=knowledge_service,
        shutdown=http_client.aclose,
    )


def _readiness_check(paths: DataPaths, index: LanceIndex) -> Callable[[], bool]:
    def check() -> bool:
        try:
            check_database_ready(paths.database)
            check_local_directories(paths)
            if not index.table_is_current():
                raise RuntimeError("RAG index is unavailable")
        except Exception as error:
            _LOGGER.warning(
                "readiness_check_failed",
                extra={"reason": type(error).__name__},
            )
            return False
        return True

    return check
