BEGIN;

CREATE TABLE service_state (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    status TEXT NOT NULL CHECK (status = 'ready')
);

CREATE TABLE users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    display_name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'disabled')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE roles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
    description TEXT NOT NULL,
    is_protected INTEGER NOT NULL CHECK (is_protected IN (0, 1)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE capabilities (
    code TEXT PRIMARY KEY,
    description TEXT NOT NULL
);

CREATE TABLE auth_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE,
    expires_at TEXT NOT NULL,
    revoked_at TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE user_roles (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role_id TEXT NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    assigned_by_user_id TEXT REFERENCES users(id),
    created_at TEXT NOT NULL,
    PRIMARY KEY (user_id, role_id)
);

CREATE TABLE role_capabilities (
    role_id TEXT NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    capability_code TEXT NOT NULL REFERENCES capabilities(code),
    PRIMARY KEY (role_id, capability_code)
);

CREATE TABLE audit_events (
    id INTEGER PRIMARY KEY,
    actor_user_id TEXT REFERENCES users(id),
    action TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('allowed', 'denied')),
    resource_type TEXT NOT NULL,
    resource_id TEXT,
    reason TEXT,
    request_id TEXT NOT NULL,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_auth_sessions_user ON auth_sessions(user_id);
CREATE INDEX idx_auth_sessions_expires_at ON auth_sessions(expires_at);
CREATE INDEX idx_user_roles_role ON user_roles(role_id);

INSERT INTO service_state (singleton, status) VALUES (1, 'ready');

INSERT INTO capabilities (code, description) VALUES
    ('identity.user.read', 'List and view users'),
    ('identity.user.manage', 'Create, enable, disable, and reset users'),
    ('identity.role.read', 'View roles, capabilities, and assignments'),
    ('identity.role.manage', 'Manage roles, capabilities, and assignments'),
    ('rag.document.upload', 'Upload documents'),
    ('rag.document.read', 'Read authorized documents'),
    ('rag.document.update', 'Update authorized documents'),
    ('rag.document.delete', 'Delete authorized documents'),
    ('rag.document.acl.manage', 'Manage authorized document ACLs'),
    ('rag.search', 'Search authorized documents'),
    ('rag.document.bypass_acl', 'Bypass document ACL with a concrete operation capability'),
    ('usage.self.read', 'View own aggregate usage and limits'),
    ('usage.all.read', 'View aggregate usage and limits for any user'),
    ('usage.limit.manage', 'Manage user limits'),
    ('security.audit.read', 'View security audit events');

INSERT INTO roles (id, name, description, is_protected, created_at, updated_at) VALUES
    ('00000000-0000-4000-8000-000000000001', 'system_admin', 'Protected system administrator', 1, '1970-01-01T00:00:00+00:00', '1970-01-01T00:00:00+00:00'),
    ('00000000-0000-4000-8000-000000000002', 'knowledge_manager', 'Manage authorized knowledge', 0, '1970-01-01T00:00:00+00:00', '1970-01-01T00:00:00+00:00'),
    ('00000000-0000-4000-8000-000000000003', 'contributor', 'Contribute authorized knowledge', 0, '1970-01-01T00:00:00+00:00', '1970-01-01T00:00:00+00:00'),
    ('00000000-0000-4000-8000-000000000004', 'reader', 'Read authorized knowledge', 0, '1970-01-01T00:00:00+00:00', '1970-01-01T00:00:00+00:00');

INSERT INTO role_capabilities (role_id, capability_code)
SELECT '00000000-0000-4000-8000-000000000001', code FROM capabilities;

INSERT INTO role_capabilities (role_id, capability_code) VALUES
    ('00000000-0000-4000-8000-000000000002', 'identity.user.read'),
    ('00000000-0000-4000-8000-000000000002', 'identity.role.read'),
    ('00000000-0000-4000-8000-000000000002', 'rag.document.upload'),
    ('00000000-0000-4000-8000-000000000002', 'rag.document.read'),
    ('00000000-0000-4000-8000-000000000002', 'rag.document.update'),
    ('00000000-0000-4000-8000-000000000002', 'rag.document.delete'),
    ('00000000-0000-4000-8000-000000000002', 'rag.document.acl.manage'),
    ('00000000-0000-4000-8000-000000000002', 'rag.search'),
    ('00000000-0000-4000-8000-000000000002', 'usage.self.read'),
    ('00000000-0000-4000-8000-000000000003', 'rag.document.upload'),
    ('00000000-0000-4000-8000-000000000003', 'rag.document.read'),
    ('00000000-0000-4000-8000-000000000003', 'rag.document.update'),
    ('00000000-0000-4000-8000-000000000003', 'rag.search'),
    ('00000000-0000-4000-8000-000000000003', 'usage.self.read'),
    ('00000000-0000-4000-8000-000000000004', 'rag.document.read'),
    ('00000000-0000-4000-8000-000000000004', 'rag.search'),
    ('00000000-0000-4000-8000-000000000004', 'usage.self.read');

COMMIT;
