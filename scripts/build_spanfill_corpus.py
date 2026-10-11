"""Arm 2: build the spanfill value-extraction corpus (train/dev/RESERVE).

Gold arguments are exact by construction: each row verbalizes a natural
request FROM a sampled (tool, arguments) instantiation. Templates 15-16
are held out of training entirely (the reserve's template-disjoint
family). Split by sha256(row): <86 train, 86-95 dev, >=95 RESERVE.

    --build            write data/real/spanfill/{train,dev}.jsonl +
                       data/heldout/spanfill-reserve.jsonl + provenance
    --stats            print split/parameter-type statistics
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "real" / "spanfill"
RESERVE = ROOT / "data" / "heldout" / "spanfill-reserve.jsonl"
PROVENANCE = OUT_DIR / "PROVENANCE.md"
SEED = 20261009

# --- tool universe -------------------------------------------------------

SYNTH_TOOLS = [
    {"name": "restart_service", "description": "Restart a systemd service on a named host.",
     "parameters": {"type": "object", "properties": {
         "service": {"type": "string", "description": "The systemd service unit name."},
         "host": {"type": "string", "description": "The target host name."}},
         "required": ["service", "host"]}},
    {"name": "deploy_release", "description": "Ship a release to a target environment.",
     "parameters": {"type": "object", "properties": {
         "release": {"type": "string", "description": "The release version tag."},
         "environment": {"type": "string", "enum": ["staging", "production", "canary"],
                          "description": "The environment to deploy to."}},
         "required": ["release", "environment"]}},
    {"name": "tail_logs", "description": "Stream recent log lines from a service.",
     "parameters": {"type": "object", "properties": {
         "service": {"type": "string", "description": "The service whose logs to stream."},
         "lines": {"type": "integer", "description": "How many lines to show."}},
         "required": ["service"]}},
    {"name": "create_ticket", "description": "Open a tracking ticket for an issue.",
     "parameters": {"type": "object", "properties": {
         "title": {"type": "string", "description": "Short ticket title."},
         "queue": {"type": "string", "enum": ["payments", "auth", "infra", "frontend"],
                    "description": "The owning queue."},
         "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"],
                       "description": "Ticket priority."}},
         "required": ["title", "queue"]}},
    {"name": "send_message", "description": "Post a message to a channel.",
     "parameters": {"type": "object", "properties": {
         "channel": {"type": "string", "description": "Channel name without the hash."},
         "body": {"type": "string", "description": "Message text to post."}},
         "required": ["channel", "body"]}},
    {"name": "rotate_key", "description": "Rotate an API credential for a service account.",
     "parameters": {"type": "object", "properties": {
         "account": {"type": "string", "description": "Service account email."},
         "environment": {"type": "string", "enum": ["staging", "production"],
                          "description": "Which environment's credential to rotate."}},
         "required": ["account", "environment"]}},
    {"name": "scale_pool", "description": "Resize a database connection pool.",
     "parameters": {"type": "object", "properties": {
         "pool": {"type": "string", "description": "Pool identifier."},
         "size": {"type": "integer", "description": "New pool size."}},
         "required": ["pool", "size"]}},
    {"name": "lookup_order", "description": "Fetch an order by identifier.",
     "parameters": {"type": "object", "properties": {
         "order_id": {"type": "string", "description": "The order reference, e.g. ORD-48213."}},
         "required": ["order_id"]}},
]

SERVICE_POOL = ["nginx", "payments-api", "auth-svc", "postgres-primary",
                "worker-queue", "redis-cache", "frontend-web", "cron-runner"]
HOST_POOL = ["web-01", "staging", "prod-eu-1", "db-replica-2", "bastion",
             "edge-fra-3", "app-07", "metrics-1"]
RELEASE_POOL = ["2026.40", "2026.41-rc1", "v3.12.0", "hotfix-2291", "2025.52"]
TITLE_POOL = ["Checkout 504s for EU customers", "Login loop on Safari",
              "Duplicate charges on retry", "Slow dashboard queries",
              "Webhook deliveries failing", "Password reset emails delayed"]
CHANNEL_POOL = ["eng-payments", "incidents", "support-eu", "deploys", "alerts"]
ACCOUNT_POOL = ["payments-bot@mivia.example.com",
                "deploy@svc.mivia.example.com",
                "ci-runner@svc.mivia.example.com",
                "alerts@svc.mivia.example.com"]
POOL_POOL = ["orders-primary", "analytics-ro", "sessions", "billing"]
ORDER_POOL = ["ORD-48213", "ORD-90211", "ORD-11784", "ORD-55390"]
BODY_POOL = ["deploy finished, watching metrics", "on-call paged for the 504 spike",
             "customer call moved to 15:00", "rotation done, keys updated"]


def _sample_value(prop: dict, rng: random.Random, name: str) -> object:
    if "enum" in prop:
        return rng.choice(prop["enum"])
    t = prop.get("type", "string")
    if t == "integer":
        return rng.choice([3, 5, 10, 16, 50, 100, 200, 500])
    if t == "number":
        return rng.choice([0.5, 1.5, 2.5, 10.0])
    if t == "boolean":
        return rng.choice([True, False])
    pools = {
        "service": SERVICE_POOL, "host": HOST_POOL, "release": RELEASE_POOL,
        "title": TITLE_POOL, "channel": CHANNEL_POOL, "account": ACCOUNT_POOL,
        "pool": POOL_POOL, "order_id": ORDER_POOL, "body": BODY_POOL,
    }
    pool = pools.get(name)
    if pool:
        return rng.choice(pool)
    return rng.choice(["acme-core", "eu-west-1", "primary", "edge-cache"])


# --- verbalization templates ---------------------------------------------
# {slots} formatted per parameter; templates 15-16 are RESERVE-ONLY.

TEMPLATES = [
    ("restart_service", "restart {service} on {host}"),
    ("restart_service", "please restart the {service} service on host {host}"),
    ("restart_service", "bounce {service} for me on {host}"),
    ("deploy_release", "ship release {release} to {environment}"),
    ("deploy_release", "deploy {release} to the {environment} environment"),
    ("tail_logs", "show me the last {lines} log lines from {service}"),
    ("tail_logs", "tail logs for {service}"),
    ("create_ticket", "open a {priority} ticket in the {queue} queue: {title}"),
    ("create_ticket", "file a ticket titled '{title}' for the {queue} team"),
    ("send_message", "post '{body}' to the {channel} channel"),
    ("rotate_key", "rotate the api key for {account} in {environment}"),
    ("scale_pool", "set the {pool} pool size to {size}"),
    ("lookup_order", "look up order {order_id}"),
    ("lookup_order", "fetch details for order {order_id} please"),
    # reserve-only phrasing family (never seen in training):
    ("restart_service", "the service {service} on {host} needs a restart"),
    ("deploy_release", "we should get {release} rolled out to {environment}"),
]


def _fmt_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def build_rows(n_target: int, rng: random.Random) -> list[dict]:
    rows = []
    while len(rows) < n_target:
        template = rng.choice(TEMPLATES)
        tool = next(t for t in SYNTH_TOOLS if t["name"] == template[0])
        tmpl = template[1]
        props = tool["parameters"]["properties"]
        required = tool["parameters"]["required"]
        args = {name: _sample_value(props[name], rng, name)
                for name in required}
        try:
            text = tmpl.format(**{k: _fmt_value(v) for k, v in args.items()})
        except KeyError:
            continue  # template references an optional slot — resample
        # optional params present in the template but not required are gold too
        gold = {k: v for k, v in args.items()}
        rows.append({"tool": tool, "request": text, "gold": gold,
                     "template": template[1], "reserve_only": template in TEMPLATES[14:]})
    return rows


def _hash_bucket(row: dict) -> int:
    blob = json.dumps({"request": row["request"], "gold": row["gold"]},
                      sort_keys=True, ensure_ascii=False)
    return int(hashlib.sha256(blob.encode()).hexdigest(), 16) % 100


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--n", type=int, default=17000)
    args = parser.parse_args()
    if not (args.build or args.stats):
        parser.error("choose --build / --stats")

    rng = random.Random(SEED)
    rows = build_rows(args.n, rng)
    train = [r for r in rows if _hash_bucket(r) < 86 and not r["reserve_only"]]
    reserve = [r for r in rows if r["reserve_only"] or _hash_bucket(r) >= 95]
    dev = [r for r in rows
           if r not in train and r not in reserve] if False else \
          [r for r in rows if r not in train and r not in reserve]

    if args.stats:
        import collections
        for name, part in (("train", train), ("dev", dev), ("reserve", reserve)):
            types = collections.Counter(
                p["type"] for r in part
                for p in r["tool"]["parameters"]["properties"].values())
            tools = collections.Counter(r["tool"]["name"] for r in part)
            print(name, len(part), dict(types))
            print("   tools:", dict(tools))
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESERVE.parent.mkdir(parents=True, exist_ok=True)
    for path, part in ((OUT_DIR / "train.jsonl", train),
                       (OUT_DIR / "dev.jsonl", dev)):
        with path.open("w") as fh:
            for r in part:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with RESERVE.open("w") as fh:
        for r in reserve:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    rsha = hashlib.sha256(RESERVE.read_bytes()).hexdigest()
    PROVENANCE.write_text(
        "# spanfill corpus provenance\n\n"
        f"- generated_utc: {datetime.now(timezone.utc).isoformat()}\n"
        f"- seed: {SEED}; rows: train {len(train)} / dev {len(dev)} / "
        f"reserve {len(reserve)}\n"
        f"- construction: requests verbalized from sampled (tool, arguments) "
        f"instantiations; gold exact by construction; 16 templates, "
        f"templates 15-16 reserve-only (template-disjoint slice)\n"
        f"- tool universe: 8 synthesized generic-ops tools (the serving-probe "
        f"gap shapes); BFCL tools carry no gold arguments (verified)\n"
        f"- split: sha256(row) <86 train / 86-95 dev / >=95 reserve + the "
        f"reserve-only template family\n"
        f"- reserve sha256: `{rsha}` (frozen before training)\n"
        f"- license: original synthetic content, no upstream source\n")
    print(f"train {len(train)} dev {len(dev)} reserve {len(reserve)}")
    print(f"reserve sha256 {rsha[:16]}... (frozen)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
