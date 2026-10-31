# ADR-0004: Webhook retry backoff

Status: accepted (supersedes ADR-0003)

## Decision

A failed webhook delivery is retried up to 8 times with exponential backoff, starting at 30 seconds and capped at 1 hour between attempts. After the eighth failure the delivery moves to the dead-letter queue; it is never dropped.

## Why

Fixed-interval retries overloaded receivers that were recovering, and dropped deliveries broke goal G1.
