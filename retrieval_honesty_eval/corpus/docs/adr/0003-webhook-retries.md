# ADR-0003: Webhook retries

Status: superseded by ADR-0004

## Decision

A failed webhook delivery is retried 3 times at a fixed interval of 60 seconds, then dropped.
