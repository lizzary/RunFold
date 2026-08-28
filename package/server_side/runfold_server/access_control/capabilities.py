from __future__ import annotations

IDENTITY_USER_READ = "identity.user.read"
IDENTITY_USER_MANAGE = "identity.user.manage"
IDENTITY_ROLE_READ = "identity.role.read"
IDENTITY_ROLE_MANAGE = "identity.role.manage"
AGENT_RUN = "agent.run"
RAG_DOCUMENT_UPLOAD = "rag.document.upload"
RAG_DOCUMENT_READ = "rag.document.read"
RAG_DOCUMENT_UPDATE = "rag.document.update"
RAG_DOCUMENT_DELETE = "rag.document.delete"
RAG_DOCUMENT_ACL_MANAGE = "rag.document.acl.manage"
RAG_SEARCH = "rag.search"
RAG_DOCUMENT_BYPASS_ACL = "rag.document.bypass_acl"
USAGE_SELF_READ = "usage.self.read"
USAGE_ALL_READ = "usage.all.read"
USAGE_LIMIT_MANAGE = "usage.limit.manage"
SECURITY_AUDIT_READ = "security.audit.read"

ALL_CAPABILITIES = frozenset(
    {
        IDENTITY_USER_READ,
        IDENTITY_USER_MANAGE,
        IDENTITY_ROLE_READ,
        IDENTITY_ROLE_MANAGE,
        AGENT_RUN,
        RAG_DOCUMENT_UPLOAD,
        RAG_DOCUMENT_READ,
        RAG_DOCUMENT_UPDATE,
        RAG_DOCUMENT_DELETE,
        RAG_DOCUMENT_ACL_MANAGE,
        RAG_SEARCH,
        RAG_DOCUMENT_BYPASS_ACL,
        USAGE_SELF_READ,
        USAGE_ALL_READ,
        USAGE_LIMIT_MANAGE,
        SECURITY_AUDIT_READ,
    }
)

ROOT_CAPABILITIES = frozenset(
    {
        IDENTITY_USER_MANAGE,
        IDENTITY_ROLE_MANAGE,
        RAG_DOCUMENT_BYPASS_ACL,
        USAGE_ALL_READ,
        USAGE_LIMIT_MANAGE,
        SECURITY_AUDIT_READ,
    }
)

SYSTEM_ADMIN_ROLE_ID = "00000000-0000-4000-8000-000000000001"
