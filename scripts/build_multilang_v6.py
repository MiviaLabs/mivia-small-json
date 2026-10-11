#!/usr/bin/env python3
"""v9 multilingual round (queued fix): v8's bare-fragment log rows WITHOUT
the retention cost — fragments ship in MIXED forms (bare + timestamped) so
the log_event signal anchors on canonical shapes, and 150 FRESH real German
tickets (disjoint from the 1,500 already used and the eval slice) anchor the
fragile non-EN ticket surface that v8 pushed to 0.98.

Targets the two measured residuals of the promoted v5 router:
1. BFCL catalogue rows (100 -> 90.7 on calls): tool-catalogue + request
   text was OOD-flagged because config-iac is the only "listy technical"
   training mass and tool catalogues sit in its basin. Adds composed,
   high-diversity catalogue+request rows as ``tool_call`` examples, in
   BOTH layouts — typed signatures and BFCL-native JSON definitions
   (v6 measured: typed-only training left JSON catalogues at no_tool_call).
2. The 2-row tickets-non-en retention miss: broader commerce/informal
   multilingual tickets (new topics, same request/complaint style).

Output: data/real/multilang-tickets/multilang-tickets-v9.jsonl = v5 +
new rows (both catalogue formats + commerce/informal + mixed log
fragments + fresh de tickets). Disjointness verified against every eval surface and corpus;
sha-pinned by tests/test_ood_aux_disjoint.py.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_ood_aux_hf import _norm, banned_texts  # noqa: E402

TICKETS_DIR = ROOT / "data/real/multilang-tickets"
V5 = TICKETS_DIR / "multilang-tickets-v5.jsonl"
DE_FORMAL = TICKETS_DIR / "multilang-de-tickets.jsonl"
CSV_PATH = (ROOT / "data/incoming/external/incident-severity"
            / "tickets-multi-lang-20k.csv")
OUT = TICKETS_DIR / "multilang-tickets-v9.jsonl"
SEED = 20261012

# ---------------------------------------------------------------------------
# Tool-catalogue rows (tool_call). Composed for coverage — the generalizing
# features are the shapes (typed signatures, catalogue layout, an explicit
# request referencing a tool's intent), so the generator draws names,
# params, catalogue sizes and request phrasings from wide pools.
# ---------------------------------------------------------------------------
TOOL_POOL = [
    ("get_weather", [("city", "string"), ("unit", "string")]),
    ("search_flights", [("origin", "string"), ("destination", "string"),
                        ("date", "string")]),
    ("book_flight", [("flight_id", "string"), ("passenger", "string")]),
    ("cancel_booking", [("booking_id", "string")]),
    ("list_calendar_events", [("day", "string")]),
    ("create_calendar_event", [("title", "string"), ("time", "string")]),
    ("send_email", [("to", "string"), ("subject", "string"),
                    ("body", "string")]),
    ("search_emails", [("query", "string"), ("max_results", "integer")]),
    ("create_task", [("title", "string"), ("due", "string"),
                     ("priority", "string")]),
    ("list_tasks", [("status", "string")]),
    ("complete_task", [("task_id", "string")]),
    ("query_database", [("sql", "string")]),
    ("list_tables", []),
    ("describe_table", [("table", "string")]),
    ("read_file", [("path", "string")]),
    ("write_file", [("path", "string"), ("content", "string")]),
    ("list_directory", [("path", "string")]),
    ("search_files", [("pattern", "string"), ("root", "string")]),
    ("get_pod_status", [("namespace", "string"), ("pod", "string")]),
    ("restart_pod", [("namespace", "string"), ("pod", "string")]),
    ("scale_deployment", [("name", "string"), ("replicas", "integer")]),
    ("fetch_url", [("url", "string")]),
    ("convert_currency", [("amount", "number"), ("from", "string"),
                          ("to", "string")]),
    ("track_shipment", [("tracking_id", "string")]),
    ("find_restaurants", [("location", "string"), ("cuisine", "string")]),
    ("reserve_table", [("restaurant_id", "string"), ("time", "string"),
                       ("party_size", "integer")]),
    ("get_stock_quote", [("symbol", "string")]),
    ("translate_text", [("text", "string"), ("target_lang", "string")]),
    ("summarize_document", [("document_id", "string")]),
    ("lookup_customer", [("email", "string")]),
    ("create_invoice", [("customer_id", "string"), ("amount", "number")]),
    ("schedule_meeting", [("attendees", "string"), ("slot", "string")]),
    ("get_jira_ticket", [("key", "string")]),
    ("create_jira_ticket", [("project", "string"), ("summary", "string"),
                            ("type", "string")]),
    ("run_query", [("dataset", "string"), ("metrics", "string")]),
]
REQUESTS = [
    "User request: book me a flight from {a} to {b} next Friday",
    "User request: what's the weather in {a} tomorrow?",
    "User request: find all emails about {a} from last month",
    "User request: create a high priority task to review {a}",
    "User request: read the file at {a} and summarize it",
    "User request: restart the {a} pod in the staging namespace",
    "User request: convert 250 from EUR to JPY",
    "User request: track shipment {id} please",
    "User request: book a table for four at a {a} place Saturday 8pm",
    "User request: what's the stock price of {a} right now",
    "User request: translate the following to German, danke",
    "User request: scale {a} to 4 replicas",
    "User request: create a bug ticket in SUP for: {a} fails on load",
    "User request: show my calendar for {a}",
    "User request: search for files matching {a} under src/",
    "User request: cancel booking {id}",
    "User request: query the customers table for {a}",
    "User request: get the status of pod {id} in prod",
    "User request: fetch and summarize {a} docs page",
    "User request: schedule a meeting with the {a} team this week",
]
FILLERS = ["Berlin", "Tokyo", "Lisbon", "the payments service", "Q3 sales",
           "invoice 8841", "auth-7f9c2", "SAP-4471", "ACME", "onboarding",
           "the api-gateway", "release 2.4", "PR-331", "inventory",
           "search", "postgres", "frontend"]
CATALOGUE_HEADERS = ["Tools available:", "Available tools:", "Tools:",
                     "You can call the following tools:", "Tool catalogue:"]
# BFCL's native layout: a JSON array of tool definitions with name/
# description/parameters. Both catalogue formats must be in-distribution
# (v6 measured: typed-signature-only training left JSON catalogues at
# no_tool_call 0.62-0.80).
NATURAL_REQUESTS = [
    "Can you book me a flight from {a} to {b} next Friday?",
    "What's the weather in {a} tomorrow?",
    "Find all emails about {a} from last month please",
    "I need a high priority task to review {a}",
    "Please read the file at {a} and summarize it",
    "Restart the {a} pod in the staging namespace",
    "Convert 250 from EUR to JPY for me",
    "Track shipment {id}",
    "Book a table for four somewhere {a} Saturday 8pm",
    "Stock price of {a}?",
    "Translate this to German, thanks",
    "Scale {a} to 4 replicas",
    "Create a bug ticket in SUP: {a} fails on load",
    "Show my calendar for {a}",
    "Search for files matching {a} under src/",
    "Cancel booking {id}",
    "Query the customers table for {a}",
    "What's the status of pod {id} in prod?",
    "Fetch and summarize the {a} docs page",
    "Schedule a meeting with the {a} team this week",
]


def _fill(req: str, rng: random.Random) -> str:
    return (req.replace("{a}", rng.choice(FILLERS))
               .replace("{b}", rng.choice(FILLERS))
               .replace("{id}", f"{rng.randint(1000, 9999)}"))


def _typed_catalogue(rng: random.Random, tools) -> str:
    parts = []
    for name, params in tools:
        if params:
            plist = ", ".join(f"{p}: {t}" for p, t in params)
            parts.append(f"{name}({plist})")
        else:
            parts.append(f"{name}()")
    header = rng.choice(CATALOGUE_HEADERS)
    req = _fill(rng.choice(REQUESTS + NATURAL_REQUESTS), rng)
    sep = rng.choice(["\n\n", "\n", " "])
    return f"{header} {', '.join(parts)}{sep}{req}"


def _json_catalogue(rng: random.Random, tools) -> str:
    defs = []
    for name, params in tools:
        entry: dict = {"name": name,
                       "description": f"Call this to {name.replace('_', ' ')}."}
        if params:
            entry["parameters"] = {
                "type": "object",
                "properties": {p: {"type": t} for p, t in params},
                "required": [p for p, _ in params],
            }
        defs.append(entry)
    req = _fill(rng.choice(REQUESTS + NATURAL_REQUESTS), rng)
    return f'Tools available:\n{json.dumps(defs, ensure_ascii=False)}\n\n{req}'


def _compose_catalogue_rows(rng: random.Random, n: int) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    builders = [_typed_catalogue, _typed_catalogue, _json_catalogue]
    while len(rows) < n:
        k = rng.choice([2, 3, 3, 4, 5, 6, 8])
        tools = rng.sample(TOOL_POOL, k)
        text = rng.choice(builders)(rng, tools)
        key = _norm(text)
        if key in seen:
            continue
        seen.add(key)
        rows.append(text)
    return rows


# ---------------------------------------------------------------------------
# Broader commerce/informal multilingual tickets (issue): new topics in the
# same request/complaint register (subscriptions, licenses, access,
# bookings, attachments, notifications).
# ---------------------------------------------------------------------------
INFORMAL_EXTRA: dict[str, list[str]] = {
    "de": [
        "Können Sie mir die Rechnungen der letzten sechs Monate als PDF schicken?",
        "Mein Lizenzschlüssel wird abgelehnt, obwohl ich erst verlängert habe.",
        "Wie aktiviere ich die Zwei-Faktor-Authentifizierung für das Team?",
        "Die Benachrichtigungen kommen bei mir gar nicht mehr an.",
        "Ich brauche Zugriff auf das Archiv für ein Audit, geht das?",
        "Die App zeigt seit dem Update keine Anhänge mehr an.",
        "Kann ich meine Sitzung auf ein anderes Gerät übertragen?",
        "Unser Firmenkonto braucht drei weitere Sitze, wie läuft das?",
    ],
    "fr": [
        "Pouvez-vous m'envoyer les factures des six derniers mois en PDF ?",
        "Ma clé de licence est refusée alors que je viens de renouveler.",
        "Comment activer la double authentification pour l'équipe ?",
        "Je ne reçois plus les notifications du tout.",
        "Il me faut un accès aux archives pour un audit, c'est possible ?",
        "L'appli n'affiche plus les pièces jointes depuis la mise à jour.",
        "Puis-je transférer ma session sur un autre appareil ?",
        "Notre compte entreprise a besoin de trois sièges en plus.",
    ],
    "es": [
        "¿Pueden enviarme las facturas de los últimos seis meses en PDF?",
        "Mi clave de licencia se rechaza aunque acabo de renovar.",
        "¿Cómo activo la autenticación de dos factores para el equipo?",
        "Ya no me llegan las notificaciones.",
        "Necesito acceso al archivo para una auditoría, ¿es posible?",
        "La app ya no muestra los adjuntos desde la actualización.",
        "¿Puedo pasar mi sesión a otro dispositivo?",
        "Nuestra cuenta de empresa necesita tres puestos más.",
    ],
    "pl": [
        "Możecie przesłać faktury z ostatnich sześciu miesięcy w PDF?",
        "Mój klucz licencyjny jest odrzucany, choć właśnie przedłużyłem.",
        "Jak włączyć dwuskładnikowe logowanie dla zespołu?",
        "Powiadomienia w ogóle do mnie nie dochodzą.",
        "Potrzebuję dostępu do archiwum na audyt, to możliwe?",
        "Aplikacja od aktualizacji nie pokazuje załączników.",
        "Czy mogę przenieść sesję na inne urządzenie?",
        "Nasze konto firmowe potrzebuje trzech dodatkowych miejsc.",
    ],
    "pt": [
        "Podem enviar-me as faturas dos últimos seis meses em PDF?",
        "A minha chave de licença é recusada embora tenha acabado de renovar.",
        "Como ativo a autenticação de dois fatores para a equipa?",
        "Já não recebo notificações nenhuma.",
        "Preciso de acesso ao arquivo para uma auditoria, é possível?",
        "A app deixou de mostrar os anexos desde a atualização.",
        "Posso passar a minha sessão para outro dispositivo?",
        "A nossa conta empresarial precisa de três lugares a mais.",
    ],
    "it": [
        "Potete inviarmi le fatture degli ultimi sei mesi in PDF?",
        "La mia chiave di licenza viene rifiutata anche se l'ho appena rinnovata.",
        "Come attivo l'autenticazione a due fattori per il team?",
        "Non ricevo più le notifiche.",
        "Mi serve l'accesso all'archivio per un audit, si può?",
        "L'app non mostra più gli allegati dall'aggiornamento.",
        "Posso trasferire la sessione su un altro dispositivo?",
        "Il nostro account aziendale ha bisogno di altri tre posti.",
    ],
    "nl": [
        "Kunnen jullie de facturen van de afgelopen zes maanden als PDF sturen?",
        "Mijn licentiesleutel wordt geweigerd terwijl ik net verlengd heb.",
        "Hoe zet ik twee-staps-verificatie aan voor het team?",
        "Ik krijg helemaal geen meldingen meer.",
        "Ik heb toegang tot het archief nodig voor een audit, kan dat?",
        "De app laat bijlagen niet meer zien sinds de update.",
        "Kan ik mijn sessie overzetten naar een ander apparaat?",
        "Ons zakelijke account heeft drie extra plekken nodig.",
    ],
    "ja": [
        "過去６か月分の請求書をPDFで送っていただけますか？",
        "更新したばかりなのにライセンスキーが拒否されます。",
        "チームの二段階認証を有効にするにはどうすればいいですか。",
        "通知がまったく届かなくなりました。",
        "監査のためにアーカイブへのアクセスが必要です。可能ですか。",
        "アップデート以降、アプリが添付ファイルを表示しません。",
        "セッションを別のデバイスに移せますか？",
        "法人アカウントにあと３ライセンス追加したいのですが。",
    ],
}



# ---------------------------------------------------------------------------
# Bare log fragments (log_event): timestamp-less single/double lines like
# "ERROR [auth]: token expired" (v6b measured: these sat in commit's basin
# after the catalogue round; with timestamps the same lines route 1.0).
# ---------------------------------------------------------------------------
LOG_LEVELS = ["ERROR", "WARN", "INFO", "FATAL", "CRITICAL", "DEBUG", "TRACE", "WARNING", "NOTICE"]
LOG_COMPONENTS = ["auth", "db", "api", "worker", "cache", "queue", "scheduler", "renderer",
                  "indexer", "gateway", "session", "upload", "billing", "search", "mailer",
                  "pool", "proxy", "router", "config", "tls", "grpc", "ws", "kafka", "redis"]
LOG_MESSAGES = [
    "token expired", "connection refused", "pool exhausted", "timeout after 30s",
    "handshake failed", "retrying in 5s", "circuit breaker open", "backpressure detected",
    "certificate verify failed", "dead letter queued", "session revoked", "rate limited",
    "disk usage 91%", "memory pressure high", "leader election lost", "lease expired",
    "schema drift detected", "consumer lag 12k", "unhealthy probe", "throttled by upstream",
    "stale cache entry", "write conflict", "rollback completed", "startup interrupted",
    "graceful shutdown", "partial write discarded", "index rebuild stalled",
]


def _compose_log_fragments(rng: random.Random, n: int) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    while len(rows) < n:
        level = rng.choice(LOG_LEVELS)
        comp = rng.choice(LOG_COMPONENTS)
        msg = rng.choice(LOG_MESSAGES)
        line = f"{level} [{comp}]: {msg}"
        if rng.random() < 0.3:
            line += (f" (attempt {rng.randint(1, 5)}, "
                     f"{rng.choice(['ms', 's'])}={rng.randint(10, 9000)})")
        if rng.random() < 0.18:
            line += (f"\\n{rng.choice(LOG_LEVELS)} "
                     f"[{rng.choice(LOG_COMPONENTS)}]: "
                     f"{rng.choice(LOG_MESSAGES)}")
        # Mixed forms (the v9 fix): 40% canonical timestamped shapes so the
        # log_event signal anchors on syslog/ISO layouts and does not dilute
        # the OOD boundary for short casual text (v8 was all-bare and cost
        # tickets-non-en 3 retention rows).
        if rng.random() < 0.4:
            if rng.random() < 0.5:
                line = ("Jun 14 "
                        f"{rng.randint(10, 23):02d}:{rng.randint(10, 59):02d}:"
                        f"{rng.randint(10, 59):02d} combo {comp}["
                        f"{rng.randint(1000, 9999)}]: " + line)
            else:
                line = (f"2026-03-0{rng.randint(1, 9)}T"
                        f"{rng.randint(10, 23):02d}:{rng.randint(10, 59):02d}:"
                        f"{rng.randint(10, 59):02d}Z " + line)
        key = _norm(line)
        if key in seen:
            continue
        seen.add(key)
        rows.append(line)
    return rows


def fresh_de_tickets(rng: random.Random, n: int = 150) -> list[str]:
    """150 MORE real German tickets: disjoint from the eval slice AND the
    1,500 already used (frozen multilang-de-tickets.jsonl)."""
    import csv

    used = {_norm(r["text"]) for r in _load_jsonl(DE_FORMAL)}
    used |= {_norm(r["text"]) for r in _load_jsonl(V5)}
    # the eval slice, recomputed exactly as the eval reads it
    eval_slice = set()
    taken = 0
    with CSV_PATH.open(encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            if taken >= 300:
                break
            if row.get("language") != "de":
                continue
            body = (row.get("body") or "").strip()
            if len(body) >= 4:
                eval_slice.add(_norm(body))
                taken += 1
    ban = banned_texts(include_fetch_only=False)
    pool = []
    seen: set[str] = set()
    with CSV_PATH.open(encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("language") != "de":
                continue
            body = (row.get("body") or "").strip()
            if len(body) < 40:
                continue
            key = _norm(body)
            if key in used or key in eval_slice or key in ban or key in seen:
                continue
            seen.add(key)
            pool.append(body)
    rng.shuffle(pool)
    took = pool[:n]
    assert len(took) == n, f"only {len(took)} fresh de tickets"
    return took


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8")
            .splitlines() if line.strip()]


def main() -> int:
    ban = banned_texts(include_fetch_only=False)
    rng = random.Random(SEED)

    catalogue_rows = [
        {"text": t, "category": "tool_call"}
        for t in _compose_catalogue_rows(rng, 420)
    ]
    log_rows = [{"text": t, "category": "log_event"}
                for t in _compose_log_fragments(rng, 200)]
    de_rows = [{"text": t, "category": "issue"} for t in fresh_de_tickets(rng)]
    informal_rows = [{"text": t, "category": "issue"}
                     for texts in INFORMAL_EXTRA.values() for t in texts]

    new_rows = (catalogue_rows + informal_rows + log_rows + de_rows)
    keys = [_norm(r["text"]) for r in new_rows]
    assert len(keys) == len(set(keys)), "duplicates in v6 additions"
    overlap = [k for k in set(keys) if k in ban]
    assert not overlap, f"contamination: {overlap[:3]}"

    prev = [json.loads(line) for line in V5.read_text(encoding="utf-8")
            .splitlines() if line.strip()]
    rows = prev + new_rows
    OUT.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8")
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    OUT.with_suffix(OUT.suffix + ".sha256").write_text(digest + "\n",
                                                       encoding="utf-8")
    print(f"multilang-tickets-v9.jsonl: {len(rows)} rows "
          f"(v5 {len(prev)} + {len(catalogue_rows)} catalogue + "
          f"{len(informal_rows)} commerce + {len(log_rows)} log fragments "
          f"+ {len(de_rows)} fresh de tickets), sha256 {digest[:12]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
