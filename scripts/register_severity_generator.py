"""Synthetic register-diverse severity generator for the grade corpus.

The ordinal grade artifact is log-domain (LogHub-trained): 100% exact on
LogHub-style lines, ~50-62% on business phrasing (measured,
reports/training-report.md). This generator writes the SAME incident kinds in
distinct registers - syslog, app-log, business English, SRE incident
language - across the four severity tiers, so the model learns tier
semantics, not register phrasing.

Every row is (text, family, tier_idx) with tier_idx into
[INFO, WARNING, ERROR, CRITICAL]. Deterministic under the seed. Template
families are numbered so the corpus builder can split train/dev at the
template level, never the row level: a template family's rows all land on
one side.
"""
from __future__ import annotations

import random

REGISTERS = ["syslog", "applog", "business", "sre"]

# Template families: (family_id, register, tier, template).
FAMILIES: list[tuple[str, str, int, str]] = [
    # --- INFO ---
    ("i-sys-1", "syslog", 0,
     "{host} {svc}[{n}]: session opened for user {user} by (uid={n})"),
    ("i-sys-2", "syslog", 0,
     "{host} systemd[1]: Started Daily apt download activities."),
    ("i-sys-3", "syslog", 0,
     "{host} CRON[{n}]: ({user}) CMD (cd {path} && ./sync.sh)"),
    ("i-app-1", "applog", 0,
     "{t} INFO  {svc} Scheduled reconciliation started for region {region}"),
    ("i-app-2", "applog", 0,
     "{t} INFO  {svc} Processed {n} webhook deliveries in {ms}ms, 0 retries"),
    ("i-app-3", "applog", 0,
     "{t} INFO  {svc} Cache warmed with {n} entries for tenant {cust}"),
    ("i-biz-1", "business", 0,
     "The month-end close reports generated on schedule and went to finance"),
    ("i-biz-2", "business", 0,
     "All {n} retail locations completed the nightly price-file update"),
    ("i-biz-3", "business", 0,
     "Customer {cust} upgraded to the enterprise plan and is provisioned"),
    ("i-biz-4", "business", 0,
     "Uptime for {svc} was 100 percent last week across all regions"),
    ("i-sre-1", "sre", 0,
     "Weekly capacity review: {svc} headroom at 4.2x peak, no action needed"),
    ("i-sre-2", "sre", 0,
     "Chaos exercise passed: killed {n} pods in {region}, recovery within SLO"),
    # --- WARNING ---
    ("w-sys-1", "syslog", 1,
     "{host} kernel: TCP: request_sock_TCP: Possible SYN flooding on port {n}. "
     "Sending cookies."),
    ("w-sys-2", "syslog", 1,
     "{host} {svc}[{n}]: rate limiting requests from {user}, {n} dropped in "
     "the last minute"),
    ("w-app-1", "applog", 1,
     "{t} WARN  {svc} Connection pool at 90 percent capacity for tenant "
     "{cust}, consider raising the limit"),
    ("w-app-2", "applog", 1,
     "{t} WARN  {svc} Retry attempt {n} of 5 for order {cust}, upstream "
     "latency {ms}ms"),
    ("w-app-3", "applog", 1,
     "{t} WARN  {svc} Disk usage on the ingest volume reached 85 percent"),
    ("w-biz-1", "business", 1,
     "The email delivery provider is throttling us, campaign sends are "
     "delayed by up to twenty minutes"),
    ("w-biz-2", "business", 1,
     "Support ticket backlog crossed one hundred open tickets for the first "
     "time this quarter"),
    ("w-biz-3", "business", 1,
     "Our payment processor notified us of a fee change that takes effect "
     "next month"),
    ("w-biz-4", "business", 1,
     "The trial-to-paid conversion rate dropped below target two weeks in "
     "a row"),
    ("w-sre-1", "sre", 1,
     "Error budget for {svc} burn rate at 2.4x, on track to exhaust the "
     "monthly budget early"),
    ("w-sre-2", "sre", 1,
     "Certificate for {svc}.{region} expires in 14 days and the auto-renew "
     "job has failed twice"),
    # --- ERROR ---
    ("e-sys-1", "syslog", 2,
     "{host} {svc}[{n}]: ERROR: Could not write to {path}: No space left on "
     "device"),
    ("e-sys-2", "syslog", 2,
     "{host} {svc}[{n}]: Failed to bind to port {n}: Address already in use, "
     "exiting"),
    ("e-app-1", "applog", 2,
     "{t} ERROR {svc} Payment capture failed for order {cust}: card declined "
     "by issuer after 3 attempts"),
    ("e-app-2", "applog", 2,
     "{t} ERROR {svc} Invoice {cust} generation aborted: tax table missing "
     "effective date"),
    ("e-app-3", "applog", 2,
     "{t} ERROR {svc} Sync to warehouse failed after {n} attempts, batch of "
     "{n} records rolled back"),
    ("e-biz-1", "business", 2,
     "Customers on the annual plan are being double-charged this cycle and "
     "support is overwhelmed with refund requests"),
    ("e-biz-2", "business", 2,
     "The data import for client {cust} failed overnight and their dashboard "
     "has been empty since Monday"),
    ("e-biz-3", "business", 2,
     "Checkout conversion dropped to zero in {region} because the currency "
     "selector stopped loading"),
    ("e-biz-4", "business", 2,
     "New signups cannot verify their email addresses, the verification link "
     "returns an expired-token page"),
    ("e-sre-1", "sre", 2,
     "Deploy 4821 rolled back: {svc} error rate at 31 percent, p99 latency "
     "6.5s in {region}"),
    ("e-sre-2", "sre", 2,
     "Kafka consumer group for {svc} is stuck, {n} messages lagging and "
     "rising"),
    # --- CRITICAL ---
    ("c-sys-1", "syslog", 3,
     "{host} kernel: BUG: unable to handle kernel NULL pointer dereference "
     "at {path} - system halted"),
    ("c-sys-2", "syslog", 3,
     "{host} {svc}[{n}]: FATAL: database directory {path} corrupted, "
     "shutting down"),
    ("c-app-1", "applog", 3,
     "{t} FATAL {svc} Cluster quorum lost in {region}, all writes rejected "
     "since {t}"),
    ("c-app-2", "applog", 3,
     "{t} CRIT  {svc} Data corruption detected in shard {n}: {n} orders "
     "affected, writes suspended globally"),
    ("c-biz-1", "business", 3,
     "The entire platform is down for all customers, status page updated, "
     "executive team briefed"),
    ("c-biz-2", "business", 3,
     "We confirmed a data breach affecting {cust} accounts and have engaged "
     "external forensics"),
    ("c-biz-3", "business", 3,
     "The backup restore failed and the primary ledger database is "
     "unrecoverable, we are declaring a disaster"),
    ("c-sre-1", "sre", 3,
     "Sev1 declared: complete outage of {svc} in all regions, no ETA until "
     "the storage vendor responds"),
    ("c-sre-2", "sre", 3,
     "Prod incident: ransomware indicators on the build fleet, all "
     "credentials rotated, CI frozen"),
]

_VALUE_POOL = {
    "host": ["web-01", "web-02", "db-primary", "cache-03", "worker-7", "edge-lon-2"],
    "user": ["alice", "svc-deploy", "backup", "www-data", "maria"],
    "svc": ["nginx", "orders-api", "billing-worker", "auth-svc", "searchd", "postgres"],
    "path": ["/var/lib/app", "/etc/nginx/conf.d", "/data/ledger", "/srv/uploads"],
    "region": ["eu-west", "us-east", "ap-south", "eu-central"],
    "cust": ["acme", "globex", "initech", "umbrella", "hooli"],
    "t": ["2026-10-06T08:14:02Z", "2026-10-06T09:41:55Z", "2026-10-05T22:03:11Z"],
}


def build_rows(seed: int, per_family: int = 60) -> list[dict]:
    """Deterministic (text, family, template, register, gold) rows."""
    rng = random.Random(seed)
    rows = []
    for family_id, register, tier, template in FAMILIES:
        for _ in range(per_family):
            vals = {k: rng.choice(v) for k, v in _VALUE_POOL.items()}
            vals["n"] = rng.randint(2, 9999)
            vals["m"] = rng.randint(2, 99)
            vals["ms"] = rng.choice(["120", "458", "1", "2031", "88"])
            rows.append({
                "text": template.format(**vals),
                "family": "severity",
                "template": family_id,
                "register": register,
                "gold": tier,
            })
    return rows


def template_split(rows: list[dict], holdout_frac: float = 0.15, seed: int = 42):
    """Split at the template-family level: every row of a family goes to one
    side, so dev rows are never paraphrases of trained templates."""
    import random as _r

    rng = _r.Random(seed)
    families = sorted({r["template"] for r in rows})
    rng.shuffle(families)
    n_hold = max(1, round(len(families) * holdout_frac))
    holdout = set(families[:n_hold])
    train = [r for r in rows if r["template"] not in holdout]
    dev = [r for r in rows if r["template"] in holdout]
    return train, dev
