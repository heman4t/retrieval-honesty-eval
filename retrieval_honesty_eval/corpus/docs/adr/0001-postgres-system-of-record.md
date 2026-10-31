# ADR-0001: PostgreSQL is the system of record

Status: accepted

## Decision

PostgreSQL is the single system of record for webhook subscriptions and delivery attempts. Redis may cache subscription lookups, but a cache miss must read PostgreSQL, never a stale copy.

## Consequences

Every write goes to PostgreSQL first; caches are filled only after the write commits.
