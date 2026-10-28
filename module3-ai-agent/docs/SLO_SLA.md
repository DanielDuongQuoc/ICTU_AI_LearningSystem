# SLO / SLA (Phase 20)

> **NO PRODUCTION TRAFFIC DATA.** Everything below is a *target proposal*. No SLO has been verified and none is claimed.
> There is no SLA: the module has no contractual commitments and no multi-instance/HA deployment.

Evidence rules: Target / Observed / Status are always separate. Status is PASS, FAIL or INCONCLUSIVE
(too few samples or no data; minimum 30 requests for latency/availability, 20 for event ratios).
Harness runs (`--offline-harness`, fake LLM) measure code paths only and are labelled `PASS (harness only)`.

| SLO | Proposed target | Observed | Status | Data source |
|---|---|---|---|---|
| Availability (non-5xx share) | >= 99.0% of requests per 30 days | none | NOT VERIFIED | needs `/metrics` from real traffic |
| `/health/live` latency | p95 <= 100 ms | not measured | NOT VERIFIED | |
| Retrieve (`/api/rag/retrieve`) p95 | <= 2 s | harness only: <= 50 ms (40 req, fake embedding) | INCONCLUSIVE for production | real embedding model latency unmeasured |
| Question generation p95 | <= 90 s | not measurable offline | NOT VERIFIED | local LLM on CPU is slow; tune on real data |
| Evaluation p95 | <= 60 s | not measurable offline | NOT VERIFIED | |
| LLM success rate | >= 95% | none | NOT VERIFIED | |
| Document processing success | >= 95% (excluding client-invalid PDFs) | none | NOT VERIFIED | |
| Index write success | >= 99% | none | NOT VERIFIED | |
| Stream completion (`ok`) | >= 95% | none | NOT VERIFIED | |

LLM targets are deliberately loose: a local model is far slower than a typical API; they must be re-based after the
first week of real measurements, not before.

## Error budget
Budget = allowed failures = `(1 - availability target) x requests` in the window. Example: 99.0% with 10,000 requests
leaves 100 failed requests. Consumed budget = observed 5xx count (from `module3_http_requests_total{status=~"5.."}`);
4xx and `RATE_LIMITED` are client errors and do not consume it. Right now: **NO PRODUCTION TRAFFIC DATA**, so consumed
budget is unknown. Burn policy suggestion: if > 50% of the budget is used before mid-window, freeze AI-version promotions.

## Measuring
`python -m scripts.check_reliability --url http://host:8000/metrics [--baseline baseline.json] [--strict]`
Percentiles are bucket upper bounds (coarse). Regression vs baseline is INCONCLUSIVE without a baseline file.
