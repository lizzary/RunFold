BEGIN;

CREATE TABLE service_state (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    status TEXT NOT NULL CHECK (status = 'ready')
);

INSERT INTO service_state (singleton, status) VALUES (1, 'ready');

COMMIT;

