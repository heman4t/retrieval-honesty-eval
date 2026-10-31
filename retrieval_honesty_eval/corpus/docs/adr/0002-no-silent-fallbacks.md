# ADR-0002: No silent fallbacks

Status: accepted

## Decision

A component that cannot produce a correct value must raise an error. It must not return a default value, retry with weaker inputs, or log and continue.

## Exceptions

None. A caller that can genuinely proceed without a value must check for its absence explicitly.
