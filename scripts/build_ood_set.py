"""Build the out-of-domain (OOD) evaluation set for the energy fast-path gate.

The energy gate on the router's 0.85 fast path was calibrated (Phase 0) on a
thin 18-row OOD reference set. This script builds a larger, more diverse set
of **non-developer** text so the threshold can be retuned and the AUROC
measured on more than a handful of rows.

Sources
-------
Real public text, fetched on demand from the Hugging Face datasets-server
REST API with the stdlib (no new dependency):

- ``fancyzhx/ag_news``   — news headlines + ledes (news)
- ``Salesforce/wikitext`` — Wikipedia prose (general prose)
- ``SetFit/emotion``     — personal/social statements (emails/chat register)
- ``SetFit/sst2``        — movie-review prose (opinion/product register)

Licenses: AG News is a public news corpus (Zhang et al. 2015); wikitext-2 is
CC BY-SA 3.0 (Wikipedia-derived); SetFit emotion and SST-2 are public
research sets. Rows are stored as-is for evaluation only and never trained
on. The bulky per-source raw dumps land in the gitignored ``raw/`` subdir;
the combined, deduplicated set is committed for reproducible measurement.

Curated synthetic text fills the categories no public set here covers:
invoices/receipts, business email, recipes, weather, legal, and product
descriptions. Written by hand, not templated, so the gate is not detecting a
generator.

Usage
-----
    .venv/bin/python scripts/build_ood_set.py            # fetch + write
    .venv/bin/python scripts/build_ood_set.py --offline  # curated only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "incoming" / "ood"
OUT_SET = OUT_DIR / "ood-set.jsonl"
RAW_DIR = OUT_DIR / "raw"
SEED = 20261006
SERVER = "https://datasets-server.huggingface.co/rows"

# Real sources: (dataset, config, split, want). Sampling is seeded and
# deterministic; re-running fetches the same rows as long as upstream order
# is stable (it is for these parquet-backed sets).
REAL_SOURCES = [
    ("fancyzhx/ag_news", "default", "train", 100),
    ("Salesforce/wikitext", "wikitext-2-raw-v1", "train", 60),
    ("SetFit/emotion", "default", "train", 50),
    ("SetFit/sst2", "default", "train", 50),
]

# Hand-written non-developer text. Deliberately varied in length, phrasing
# and number, so a threshold that separates these is not just detecting a
# template. (category, text)
CURATED: list[tuple[str, str]] = [
    # --- invoices / receipts -------------------------------------------------
    ("invoice", "Invoice #INV-2026-991: Payment of $450 for cloud consulting "
                "services due on 2026-11-01"),
    ("invoice", "INVOICE 2026-0042 — Acme Corporation — 40 hrs consulting @ "
                "$150/hr — Subtotal $7,200 — Tax 8.25% — Total Due $7,794.00"),
    ("invoice", "Your invoice for March is ready. Amount due: £1,284.50. "
                "Payment terms: net 30. Reference: GB-4471."),
    ("invoice", "Tax Invoice\nSupplier: Northwind Traders Ltd\nABN 12 345 678 "
                "901\nGoods: 3x standing desk, 2x monitor arm\nTotal incl GST: "
                "AUD 2,145.00"),
    ("invoice", "Factura núm. 2026-118 por servicios de mantenimiento, "
                "importe 890,00 €, vencimiento 30 días."),
    ("receipt", "RECEIPT\nCappuccino x2 ........ 7.80\nAlmond croissant ... "
                "3.20\nSubtotal 11.00\nCard ****4417 11.00\nThank you!"),
    ("receipt", "Store #0912  12/03/2026 18:42\nMilk 1gal 4.29\nEggs dozen "
                "5.49\nBread 3.99\nTOTAL 13.77\nVISA 13.77"),
    ("receipt", "Order confirmation: 2 items, $58.40, shipped to 44 Baker "
                "St, arriving Tuesday. Order #A10-99321."),
    ("invoice", "Remittance advice: we have paid invoice INV-7781 in full. "
                "Balance outstanding is now $0.00."),
    ("invoice", "Purchase order PO-55231 raised for 500 units of part "
                "A-19, unit price $2.14, delivery week 12."),
    ("receipt", "Fuel receipt — 42.7 L — $71.24 — pump 6 — 2026-03-14 "
                "09:12 — thank you for visiting."),
    ("invoice", "Statement of account: opening balance $1,020.44, two "
                "payments received, closing balance $310.44."),
    ("invoice", "VAT invoice for advertising services rendered in February, "
                "net €4,000, VAT 20% €800, total €4,800."),
    ("receipt", "Grocery total: $86.13. Savings this visit: $9.40. "
                "Loyalty points earned: 86."),
    ("invoice", "Final notice: invoice 2025-331 remains unpaid 60 days past "
                "due. Late fee of $45.00 has been applied."),
    # --- business email ------------------------------------------------------
    ("email", "Hi Sarah, just following up on the contract draft I sent last "
              "week. Could you confirm the delivery dates? Thanks, Tom"),
    ("email", "Dear Mr. Okafor, thank you for your application. We would like "
              "to invite you to an interview on Thursday at 2pm."),
    ("email", "Team, reminder that the quarterly report is due Friday. Please "
              "send your sections to me by end of day Wednesday."),
    ("email", "Hello, I am writing to reschedule our meeting. Would Tuesday "
              "afternoon work instead of Monday morning? Best regards, Aisha"),
    ("email", "Just a note to say the parcel arrived safely, thank you for "
              "packing it so carefully."),
    ("email", "Dear residents, the water will be shut off between 9am and 1pm "
              "on Tuesday for scheduled maintenance."),
    ("email", "Thanks for dinner last night, we had a lovely time. Let's do "
              "it again soon. — Priya"),
    ("email", "I wanted to check whether you received the documents I mailed "
              "on the 3rd. Please let me know if anything is missing."),
    ("email", "Newsletter: our spring collection is here, plus free shipping "
              "on orders over $50 this weekend only."),
    ("email", "Apologies for the delay in replying — I was travelling. Here "
              "is the information you asked for."),
    ("email", "Could you please send the recipe for that lemon cake? "
              "Everyone loved it at the party."),
    ("email", "Dear customer, your subscription renews on 1 April at $9.99. "
              "You can cancel anytime from your account."),
    ("email", "Hi, I'd like to book a table for four on Saturday evening "
              "around 7:30 if you have space."),
    ("email", "Following our call, I've attached the revised quote. Let me "
              "know if the numbers look right."),
    ("email", "Happy birthday! Hope you have a wonderful day surrounded by "
              "people you love."),
    # --- recipes -------------------------------------------------------------
    ("recipe", "Cream the butter and sugar until pale, beat in the eggs one "
               "at a time, fold in the flour, then bake at 180C for 25 min."),
    ("recipe", "Ingredients: 2 cups flour, 1 tsp baking soda, 1/2 tsp salt, "
               "1 cup butter, 3/4 cup sugar, 2 eggs, 1 tsp vanilla."),
    ("recipe", "Bring a large pot of salted water to a boil, add the "
               "spaghetti, and cook until al dente, about 9 minutes."),
    ("recipe", "For the marinade, whisk soy sauce, honey, garlic and ginger, "
               "then pour over the chicken and chill for two hours."),
    ("recipe", "Roast the vegetables at 200C for 40 minutes, turning once, "
               "until the edges caramelise and the centres are tender."),
    ("recipe", "Simmer the tomatoes with onion and basil for 30 minutes, "
               "then blend until smooth and season to taste."),
    ("recipe", "Knead the dough for ten minutes, cover, and let it rise in a "
               "warm place until doubled in size."),
    ("recipe", "Serve the soup with a swirl of cream and a scattering of "
               "freshly chopped parsley."),
    ("recipe", "Melt the chocolate over a bain-marie, stir in the cream, and "
               "pour the ganache over the cooled tart."),
    ("recipe", "Toast the spices in a dry pan until fragrant, then grind "
               "them and add to the curry."),
    ("recipe", "Preheat the oven to 220C. Toss the potatoes in oil and "
               "rosemary, spread on a tray, and roast for 35 minutes."),
    ("recipe", "Whisk the egg whites to stiff peaks and fold them gently "
               "into the batter to keep the air in."),
    ("recipe", "For a fluffier pancake, rest the batter for fifteen minutes "
               "before cooking over medium heat."),
    ("recipe", "Drain the chickpeas, mash with tahini, lemon and garlic, and "
               "thin with a little water until smooth."),
    ("recipe", "Chill the dough for an hour so it rolls out cleanly without "
               "sticking to the worktop."),
    # --- weather -------------------------------------------------------------
    ("weather", "Today will be mostly cloudy with a high of 14 degrees and a "
                "gentle breeze from the west."),
    ("weather", "Severe thunderstorm warning in effect until 9pm; expect "
                "heavy rain, hail and gusts up to 60 mph."),
    ("weather", "Tomorrow: sunny spells, scattered showers in the afternoon, "
                "temperature around 18C."),
    ("weather", "Overnight lows will dip below freezing, so watch for icy "
                "patches on untreated roads."),
    ("weather", "A heat advisory is in place for the weekend, with heat index "
                "values near 105F and poor air quality."),
    ("weather", "The hurricane is expected to make landfall late Thursday, "
                "bringing storm surge and widespread flooding."),
    ("weather", "Coastal fog will linger through the morning, lifting to "
                "reveal clear skies by early afternoon."),
    ("weather", "A cold front moves through tonight, dropping temperatures "
                "by ten degrees and bringing gusty northwest winds."),
    ("weather", "Rainfall totals of one to two inches are forecast, with "
                "localised flooding possible in low-lying areas."),
    ("weather", "The extended outlook calls for a dry week with above-average "
                "temperatures and light winds."),
    ("weather", "Snow accumulation of 4 to 6 inches is expected above 5,000 "
                "feet, with travel impacts over the passes."),
    ("weather", "Air quality is unhealthy for sensitive groups today; limit "
                "outdoor exertion and keep windows closed."),
    ("weather", "Sunrise at 6:42, sunset at 19:58; UV index moderate, so "
                "sunscreen is advised midday."),
    ("weather", "Tropical storm warnings remain for the islands as the system "
                "moves west-northwest at 12 mph."),
    ("weather", "A blizzard warning is in effect; visibilities will drop "
                "below a quarter mile in blowing snow."),
    # --- legal ---------------------------------------------------------------
    ("legal", "The parties agree that this Agreement shall be governed by the "
              "laws of the State of Delaware, without regard to conflict rules."),
    ("legal", "In witness whereof, the parties have executed this contract as "
              "of the date first written above."),
    ("legal", "Section 4.2 — Limitation of Liability. Neither party shall be "
              "liable for indirect or consequential damages."),
    ("legal", "This lease is for a term of twelve months, commencing on 1 "
              "May, with rent payable on the first of each month."),
    ("legal", "The Tenant shall not assign or sublet the premises without the "
              "prior written consent of the Landlord."),
    ("legal", "All intellectual property created under this agreement "
              "remains the sole property of the Company."),
    ("legal", "Notice of termination must be delivered in writing at least "
              "thirty days before the intended termination date."),
    ("legal", "WHEREAS the Seller desires to sell and the Buyer desires to "
              "purchase the assets described in Schedule A."),
    ("legal", "The employee agrees to maintain the confidentiality of all "
              "proprietary information during and after employment."),
    ("legal", "Any dispute arising hereunder shall be resolved by binding "
              "arbitration in accordance with the rules of the AAA."),
    ("legal", "This warranty is exclusive and in lieu of all other "
              "warranties, express or implied, including merchantability."),
    ("legal", "The court held that the defendant's conduct constituted a "
              "material breach of the implied covenant of good faith."),
    ("legal", "Pursuant to clause 7, either party may terminate for "
              "convenience on sixty days' written notice."),
    ("legal", "The foregoing terms supersede all prior negotiations and "
              "understandings between the parties."),
    ("legal", "Force Majeure: neither party shall be liable for failure to "
              "perform caused by events beyond its reasonable control."),
    # --- product descriptions ------------------------------------------------
    ("product", "These wireless earbuds deliver 30 hours of playtime, active "
                "noise cancellation, and a secure, comfortable fit."),
    ("product", "The cast-iron skillet is pre-seasoned and oven-safe to "
                "500F, with a helper handle for easy lifting."),
    ("product", "A lightweight merino wool sweater that regulates "
                "temperature and resists odour on long hikes."),
    ("product", "This ergonomic office chair features adjustable lumbar "
                "support, a breathable mesh back, and smooth-rolling casters."),
    ("product", "Stainless steel water bottle, double-walled and vacuum "
                "insulated, keeps drinks cold for 24 hours."),
    ("product", "Our organic cotton sheets are sateen-woven for a soft "
                "hand feel and get softer with every wash."),
    ("product", "A compact 4K action camera with image stabilisation, "
                "waterproof housing, and a 3-hour battery."),
    ("product", "The ceramic nonstick pan heats evenly and lets you cook "
                "with little oil, and it is dishwasher safe."),
    ("product", "This backpack has a padded 16-inch laptop sleeve, a "
                "water-resistant shell, and hidden anti-theft pockets."),
    ("product", "Bright, dimmable LED desk lamp with adjustable colour "
                "temperature and a built-in USB charging port."),
    ("product", "The blender's 1200-watt motor crushes ice and frozen fruit "
                "in seconds, with dishwasher-safe parts."),
    ("product", "A hardcover notebook with 200 gsm paper, an elastic "
                "closure, and a ribbon marker for everyday notes."),
    ("product", "These running shoes use a responsive foam midsole and a "
                "breathable knit upper for long-distance comfort."),
    ("product", "The smart thermostat learns your schedule and can cut "
                "heating bills by up to 15 percent."),
    ("product", "A stainless-steel chef's knife with a full tang, hand-sharpened "
                "edge, and a balanced, non-slip handle."),
    ("product", "This pet camera streams 1080p video, tosses treats, and "
                "sends alerts when your dog barks."),
    ("product", "Lightweight aluminium travel tripod folds to 15 inches and "
                "supports cameras up to 8 kg."),
    ("product", "A plush memory-foam pillow with a cooling gel layer for "
                "hot sleepers and a washable cover."),
    ("product", "The rechargeable beard trimmer offers 40 length settings "
                "and up to 90 minutes of cordless runtime."),
    ("product", "These noise-isolating earplugs reduce up to 33 dB and come "
                "with a compact aluminium carry case."),
]


def fetch(dataset: str, config: str, split: str, want: int,
          pages: int = 4, length: int = 100) -> list[dict]:
    """Fetch up to ``pages*length`` rows, return the raw row dicts."""
    rows: list[dict] = []
    for page in range(pages):
        offset = page * length
        query = urllib.parse.urlencode({
            "dataset": dataset, "config": config, "split": split,
            "offset": offset, "length": length})
        req = urllib.request.Request(
            f"{SERVER}?{query}", headers={"User-Agent": "mivia-ood-build/1"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.load(resp)
        rows.extend(r["row"] for r in payload.get("rows", []))
        if len(rows) >= want * 3:
            break
    return rows


def _clean(text: str) -> str:
    text = (text or "").replace("\\b", " ").replace("\\n", " ").strip()
    return " ".join(text.split())


def _usable(text: str, min_len: int = 60) -> bool:
    if len(text) < min_len:
        return False
    if text.startswith("=") or text.startswith("@"):
        return False
    printable = sum(c.isprintable() for c in text)
    return printable / max(len(text), 1) >= 0.9


def build_real(rng: random.Random) -> list[dict]:
    out: list[dict] = []
    for dataset, config, split, want in REAL_SOURCES:
        rows = fetch(dataset, config, split, want)
        # Bucket by label when present so the sample spans classes, then
        # fall back to a flat seeded sample.
        buckets: dict = {}
        for row in rows:
            text = _clean(row.get("text", ""))
            if _usable(text):
                buckets.setdefault(row.get("label", 0), []).append(text)
        picked: list[str] = []
        if buckets and len(buckets) > 1:
            per = max(1, want // len(buckets))
            for label in sorted(buckets, key=str):
                pool = buckets[label]
                rng.shuffle(pool)
                picked.extend(pool[:per])
        if len(picked) < want:
            flat = [t for pool in buckets.values() for t in pool]
            rng.shuffle(flat)
            for text in flat:
                if text not in picked:
                    picked.append(text)
                if len(picked) >= want:
                    break
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        raw_path = RAW_DIR / f"{dataset.replace('/', '__')}.jsonl"
        with raw_path.open("w", encoding="utf-8") as fh:
            for text in picked[:want]:
                fh.write(json.dumps({"source": dataset, "text": text},
                                    ensure_ascii=False) + "\n")
        for text in picked[:want]:
            out.append({"source": dataset, "category": "real", "text": text})
        print(f"  {dataset}: {len(picked[:want])} rows")
    return out


def build_curated() -> list[dict]:
    return [{"source": "curated-synthetic", "category": cat, "text": text}
            for cat, text in CURATED]


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _real_corpus() -> set[str]:
    """Normalized text already in the in-domain corpora, for dedup."""
    seen: set[str] = set()
    for path in (ROOT / "data" / "real").glob("*/*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ("raw", "text"):
                if row.get(key):
                    seen.add(_norm(row[key]))
    return seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", action="store_true",
                    help="skip the network fetch; write the curated set only")
    args = ap.parse_args()

    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    if not args.offline:
        print("fetching real non-developer text ...")
        rows.extend(build_real(rng))
    rows.extend(build_curated())

    # Dedup against the in-domain corpora and within the OOD set.
    in_domain = _real_corpus()
    seen: set[str] = set()
    kept: list[dict] = []
    for row in rows:
        key = _norm(row["text"])
        if key in seen or key in in_domain:
            continue
        seen.add(key)
        kept.append(row)
    kept.sort(key=lambda r: (r["category"], r["source"], r["text"]))

    with OUT_SET.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(kept):
            row = {"id": f"ood-{i:04d}", **row}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    digest = hashlib.sha256(OUT_SET.read_bytes()).hexdigest()
    by_cat: dict[str, int] = {}
    for row in kept:
        by_cat[row["category"]] = by_cat.get(row["category"], 0) + 1
    print(f"wrote {len(kept)} rows -> {OUT_SET.relative_to(ROOT)}")
    print(f"sha256 {digest}")
    print("by category:", json.dumps(by_cat, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
