"""Extra generated layers of the ledgerkit repo (used by build_repo.py).

Bank-API sync providers, currency and holiday tables, a merchant table, and their
tests. Deterministic: fixed tables and fixed random seeds.
"""
from __future__ import annotations

import random

SAMPLE = [("2026-03-05", "Coffee Shop", -350), ("2026-03-31", "ACME Payroll", 123456)]

# name, title, base url, list path, auth scheme, items key, next key, date field, amount path, memo field
PROVIDERS = [
    ("gocardless", "GoCardless Bank Account Data", "https://bankaccountdata.gocardless.com/api/v2",
     "/accounts/{account}/transactions/", "Bearer", "booked", "next", "bookingDate",
     ("transactionAmount", "amount"), "remittanceInformationUnstructured"),
    ("plaid", "Plaid", "https://production.plaid.com", "/transactions/sync", "Bearer", "added", "next_cursor",
     "date", ("amount",), "name"),
    ("truelayer", "TrueLayer", "https://api.truelayer.com/data/v1", "/accounts/{account}/transactions", "Bearer",
     "results", "next", "timestamp", ("amount",), "description"),
    ("tink", "Tink", "https://api.tink.com/data/v2", "/transactions", "Bearer", "transactions", "nextPageToken",
     ("dates", "booked"), ("amount", "value"), ("descriptions", "display")),
    ("yapily", "Yapily", "https://api.yapily.com", "/accounts/{account}/transactions", "Basic", "data", "next",
     "bookingDateTime", ("transactionAmount", "amount"), "description"),
    ("saltedge", "Salt Edge", "https://www.saltedge.com/api/v6", "/transactions", "App-id", "data", "next_id",
     "made_on", ("amount",), "description"),
    ("finapi", "finAPI", "https://live.finapi.io/api/v2", "/transactions", "Bearer", "transactions", "nextPage",
     "bankBookingDate", ("amount",), "purpose"),
    ("nordigen", "Nordigen (legacy)", "https://ob.nordigen.com/api/v2", "/accounts/{account}/transactions/",
     "Bearer", "booked", "next", "valueDate", ("transactionAmount", "amount"), "remittanceInformationUnstructured"),
]

CURRENCIES = [("EUR", "Euro", 2), ("USD", "US Dollar", 2), ("GBP", "Pound Sterling", 2), ("CHF", "Swiss Franc", 2),
              ("JPY", "Yen", 0), ("SEK", "Swedish Krona", 2), ("NOK", "Norwegian Krone", 2),
              ("DKK", "Danish Krone", 2), ("PLN", "Zloty", 2), ("CZK", "Czech Koruna", 2), ("HUF", "Forint", 2),
              ("RON", "Romanian Leu", 2), ("BGN", "Bulgarian Lev", 2), ("ISK", "Iceland Krona", 0),
              ("CAD", "Canadian Dollar", 2), ("AUD", "Australian Dollar", 2), ("NZD", "New Zealand Dollar", 2),
              ("MXN", "Mexican Peso", 2), ("BRL", "Brazilian Real", 2), ("ARS", "Argentine Peso", 2),
              ("CLP", "Chilean Peso", 0), ("COP", "Colombian Peso", 2), ("PYG", "Guarani", 0),
              ("UYU", "Peso Uruguayo", 2), ("PEN", "Sol", 2), ("CNY", "Yuan Renminbi", 2),
              ("HKD", "Hong Kong Dollar", 2), ("SGD", "Singapore Dollar", 2), ("KRW", "Won", 0),
              ("INR", "Indian Rupee", 2), ("IDR", "Rupiah", 2), ("THB", "Baht", 2), ("TRY", "Turkish Lira", 2),
              ("ZAR", "Rand", 2), ("ILS", "New Israeli Sheqel", 2), ("AED", "UAE Dirham", 2),
              ("KWD", "Kuwaiti Dinar", 3), ("BHD", "Bahraini Dinar", 3), ("TND", "Tunisian Dinar", 3)]

COUNTRIES = {
    "DE": [("01-01", "Neujahr"), ("05-01", "Tag der Arbeit"), ("10-03", "Tag der Deutschen Einheit"),
           ("12-25", "1. Weihnachtstag"), ("12-26", "2. Weihnachtstag")],
    "FR": [("01-01", "Jour de l'an"), ("05-01", "Fete du Travail"), ("05-08", "Victoire 1945"),
           ("07-14", "Fete nationale"), ("08-15", "Assomption"), ("11-01", "Toussaint"), ("11-11", "Armistice"),
           ("12-25", "Noel")],
    "GB": [("01-01", "New Year's Day"), ("12-25", "Christmas Day"), ("12-26", "Boxing Day")],
    "NL": [("01-01", "Nieuwjaarsdag"), ("04-27", "Koningsdag"), ("05-05", "Bevrijdingsdag"),
           ("12-25", "Eerste Kerstdag"), ("12-26", "Tweede Kerstdag")],
    "ES": [("01-01", "Ano Nuevo"), ("01-06", "Epifania"), ("05-01", "Dia del Trabajador"), ("08-15", "Asuncion"),
           ("10-12", "Fiesta Nacional"), ("11-01", "Todos los Santos"), ("12-06", "Constitucion"),
           ("12-08", "Inmaculada Concepcion"), ("12-25", "Navidad")],
    "IT": [("01-01", "Capodanno"), ("01-06", "Epifania"), ("04-25", "Liberazione"), ("05-01", "Festa del Lavoro"),
           ("06-02", "Festa della Repubblica"), ("08-15", "Ferragosto"), ("11-01", "Ognissanti"),
           ("12-08", "Immacolata"), ("12-25", "Natale"), ("12-26", "Santo Stefano")],
    "AT": [("01-01", "Neujahr"), ("01-06", "Heilige Drei Koenige"), ("05-01", "Staatsfeiertag"),
           ("08-15", "Mariae Himmelfahrt"), ("10-26", "Nationalfeiertag"), ("11-01", "Allerheiligen"),
           ("12-08", "Mariae Empfaengnis"), ("12-25", "Christtag"), ("12-26", "Stefanitag")],
    "CH": [("01-01", "Neujahr"), ("08-01", "Bundesfeiertag"), ("12-25", "Weihnachten"), ("12-26", "Stephanstag")],
    "US": [("01-01", "New Year's Day"), ("06-19", "Juneteenth"), ("07-04", "Independence Day"),
           ("11-11", "Veterans Day"), ("12-25", "Christmas Day")],
    "PY": [("01-01", "Ano Nuevo"), ("03-01", "Dia de los Heroes"), ("05-01", "Dia del Trabajador"),
           ("05-14", "Independencia"), ("05-15", "Independencia"), ("06-12", "Paz del Chaco"),
           ("08-15", "Fundacion de Asuncion"), ("12-08", "Virgen de Caacupe"), ("12-25", "Navidad")],
}

MERCHANT_CATEGORIES = ["groceries", "eating out", "transport", "utilities", "health", "shopping", "travel",
                       "entertainment", "education", "insurance", "home", "fees"]
_SYL = ["al", "ber", "cor", "dan", "el", "fen", "gro", "hal", "ik", "jo", "kel", "lun", "mar", "nor", "or",
        "pel", "quin", "ros", "sil", "tor", "ul", "ven", "wal", "xan", "yor", "zel"]
_SUFFIX = ["mart", "market", "foods", "express", "store", "hub", "point", "house", "co", "direct", "plus",
           "online", "center", "station", "bistro", "line"]


def merchants() -> list:
    rng = random.Random(7)
    seen, out = set(), []
    while len(out) < 360:
        name = (rng.choice(_SYL) + rng.choice(_SYL) + rng.choice(["", rng.choice(_SYL)])).capitalize()
        name = f"{name} {rng.choice(_SUFFIX).capitalize()}"
        if name.lower() not in seen and "coffee" not in name.lower():
            seen.add(name.lower())
            out.append((name, MERCHANT_CATEGORIES[len(out) % len(MERCHANT_CATEGORIES)]))
    return out


def _keys(path) -> list:
    return [path] if isinstance(path, str) else list(path)


def _access(var: str, path) -> str:
    return var + "".join(f"[{k!r}]" for k in _keys(path))


def provider_module(spec) -> str:
    name, title, base, path, scheme, items, nxt, datef, amount, memo = spec
    cls = "".join(p.capitalize() for p in name.split("_")) + "Provider"
    return (
        f'"""{title} API client: builds requests and parses transaction pages.\n\n'
        "Network I/O is done by the caller (see ledgerkit.sync.runner); this module is pure.\n"
        '"""\n'
        "from ledgerkit.errors import LedgerError\n"
        "from ledgerkit.importers.base import ImportedRow\n"
        "from ledgerkit.money import parse_amount\n"
        "from ledgerkit.utils.text import clean_memo\n\n"
        f"BASE_URL = {base!r}\nLIST_PATH = {path!r}\nAUTH_SCHEME = {scheme!r}\n"
        f"ITEMS_KEY = {items!r}\nNEXT_KEY = {nxt!r}\n\n\n"
        f"class {cls}:\n"
        f"    name = {name!r}\n\n"
        "    def build_request(self, token: str, account: str, since: str, page=None) -> dict:\n"
        "        if not token:\n"
        f'            raise LedgerError(400, "{name}: missing API token")\n'
        '        params = {"date_from": since}\n'
        "        if page:\n"
        "            params[NEXT_KEY] = page\n"
        "        return {\n"
        '            "method": "GET",\n'
        '            "url": BASE_URL + LIST_PATH.format(account=account),\n'
        '            "headers": {"Authorization": f"{AUTH_SCHEME} {token}", "Accept": "application/json"},\n'
        '            "params": params,\n'
        "        }\n\n"
        "    def parse_page(self, payload: dict, account: str):\n"
        '        """Return (rows, next_page_token or None)."""\n'
        "        try:\n"
        "            items = payload[ITEMS_KEY]\n"
        "        except (KeyError, TypeError):\n"
        f'            raise LedgerError(422, "{name}: unexpected payload") from None\n'
        "        rows = []\n"
        "        for n, item in enumerate(items, start=1):\n"
        "            try:\n"
        f"                date = str({_access('item', datef)})[:10]\n"
        f"                cents = parse_amount(str({_access('item', amount)}))\n"
        f"                memo = clean_memo(str({_access('item', memo)}))\n"
        "            except (KeyError, TypeError):\n"
        f'                raise LedgerError(422, f"{name}: item {{n}} is missing fields") from None\n'
        "            rows.append(ImportedRow(date, account, cents, memo, self.name, n))\n"
        "        return rows, payload.get(NEXT_KEY) or None\n\n\n"
        f"PROVIDER = {cls}()\n"
    )


def _nest(path, value):
    for key in reversed(_keys(path)):
        value = {key: value}
    return value


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def provider_payload(spec) -> dict:
    name, title, base, path, scheme, items, nxt, datef, amount, memo = spec
    rows = []
    for iso, text, cents in SAMPLE:
        sign = "-" if cents < 0 else ""
        amt = f"{sign}{abs(cents) // 100}.{abs(cents) % 100:02d}"
        rows.append(_merge(_merge(_nest(datef, iso + "T10:00:00Z"), _nest(amount, amt)), _nest(memo, text)))
    return {items: rows, nxt: "page-2"}


def _registry(pkg: str, doc: str, names: list, attr: str, what: str) -> str:
    joined = ", ".join(names)
    return (
        f'"""{doc}"""\n'
        "from ledgerkit.errors import LedgerError\n"
        f"from ledgerkit.{pkg} import {joined}\n\n"
        f"{what.upper()}S = {{m.{attr}.name: m.{attr} for m in ({joined},)}}\n\n\n"
        "def get(name: str):\n"
        "    try:\n"
        f"        return {what.upper()}S[name]\n"
        "    except KeyError:\n"
        f'        raise LedgerError(400, f"unknown {what}: {{name}}") from None\n'
    )


SYNC_RUNNER = '''"""Drive a provider page by page with an injected transport (tests pass a fake)."""
from ledgerkit.errors import LedgerError
from ledgerkit.importers.base import apply

MAX_PAGES = 50


def sync(ledger, provider, transport, token: str, account: str, since: str) -> int:
    """transport(request_dict) -> payload dict. Returns how many entries were added."""
    page, added = None, 0
    for _ in range(MAX_PAGES):
        payload = transport(provider.build_request(token, account, since, page))
        rows, page = provider.parse_page(payload, account)
        added += apply(ledger, rows)
        if not page:
            return added
    raise LedgerError(409, f"{provider.name}: more than {MAX_PAGES} pages")
'''


def sync_tests() -> str:
    cases = "\n".join(f"    ({p[0]!r}, {p[6]!r}, {provider_payload(p)!r})," for p in PROVIDERS)
    expected = [(d, m, c) for d, m, c in SAMPLE]
    return (
        "import pytest\n\n"
        "from ledgerkit import sync\n"
        "from ledgerkit.errors import LedgerError\n"
        "from ledgerkit.ledger import Ledger\n"
        "from ledgerkit.sync.runner import sync as run_sync\n\n"
        f"CASES = [\n{cases}\n]\n\n\n"
        '@pytest.mark.parametrize("name,next_key,payload", CASES)\n'
        "def test_parse_page(name, next_key, payload):\n"
        '    rows, nxt = sync.get(name).parse_page(payload, "bank")\n'
        f"    assert [(r.date, r.memo, r.amount_cents) for r in rows] == {expected!r}\n"
        '    assert nxt == "page-2"\n\n\n'
        '@pytest.mark.parametrize("name,next_key,payload", CASES)\n'
        "def test_request_needs_token(name, next_key, payload):\n"
        "    provider = sync.get(name)\n"
        '    req = provider.build_request("t", "acc1", "2026-01-01", page="p2")\n'
        '    assert req["headers"]["Authorization"].endswith(" t") and req["params"][next_key] == "p2"\n'
        "    with pytest.raises(LedgerError):\n"
        '        provider.build_request("", "acc1", "2026-01-01")\n\n\n'
        '@pytest.mark.parametrize("name,next_key,payload", CASES)\n'
        "def test_runner_follows_pages(name, next_key, payload):\n"
        "    last = {k: v for k, v in payload.items() if k != next_key}\n"
        "    pages = iter([payload, last])\n"
        '    assert run_sync(Ledger(), sync.get(name), lambda req: next(pages), "t", "bank", "2026-01-01") == 2\n'
    )


def currencies_module() -> str:
    rows = "\n".join(f"    {code!r}: ({name!r}, {minor})," for code, name, minor in CURRENCIES)
    return (
        '"""ISO 4217 currencies: code -> (name, minor units)."""\n'
        "from ledgerkit.errors import LedgerError\n\n"
        f"CURRENCIES = {{\n{rows}\n}}\n\n\n"
        "def minor_units(code: str) -> int:\n"
        "    try:\n"
        "        return CURRENCIES[code.upper()][1]\n"
        "    except KeyError:\n"
        '        raise LedgerError(400, f"unknown currency: {code}") from None\n\n\n'
        "def name_of(code: str) -> str:\n"
        "    return CURRENCIES[code.upper()][0]\n"
    )


def holidays_module() -> str:
    lines = [f'    ("{c}", "{y}-{md}"): {label!r},' for c, days in COUNTRIES.items()
             for y in (2024, 2025, 2026, 2027) for md, label in days]
    return (
        '"""Fixed-date bank holidays per country (movable feasts are not modelled)."""\n'
        "from ledgerkit.utils.dates import add_days, parse_iso\n\n"
        "HOLIDAYS = {\n" + "\n".join(lines) + "\n}\n\n\n"
        "def is_holiday(iso: str, country: str) -> bool:\n"
        "    return (country, iso) in HOLIDAYS\n\n\n"
        "def is_business_day(iso: str, country: str) -> bool:\n"
        "    return parse_iso(iso).weekday() < 5 and not is_holiday(iso, country)\n\n\n"
        "def next_business_day(iso: str, country: str) -> str:\n"
        "    day = add_days(iso, 1)\n"
        "    while not is_business_day(day, country):\n"
        "        day = add_days(day, 1)\n"
        "    return day\n"
    )


def merchants_module() -> str:
    rows = "\n".join(f"    {name.lower()!r}: {cat!r}," for name, cat in merchants())
    return (
        '"""Known merchants (lower-case name -> category)."""\n'
        f"MERCHANTS = {{\n{rows}\n}}\n\n\n"
        "def merchant_category(memo: str):\n"
        "    text = memo.lower()\n"
        "    for name, category in MERCHANTS.items():\n"
        "        if name in text:\n"
        "            return category\n"
        "    return None\n"
    )


def data_tests() -> str:
    name, cat = merchants()[5]
    return (
        "import pytest\n\n"
        "from ledgerkit.calendar import holidays\n"
        "from ledgerkit.data import currencies\n"
        "from ledgerkit.errors import LedgerError\n"
        "from ledgerkit.rules.merchants import MERCHANTS, merchant_category\n\n\n"
        "def test_currencies():\n"
        '    assert currencies.minor_units("jpy") == 0\n'
        '    assert currencies.minor_units("KWD") == 3\n'
        "    with pytest.raises(LedgerError):\n"
        '        currencies.minor_units("XXX")\n\n\n'
        "def test_holidays():\n"
        '    assert holidays.is_holiday("2026-10-03", "DE")\n'
        '    assert not holidays.is_business_day("2026-12-25", "GB")\n'
        '    assert holidays.next_business_day("2026-12-24", "DE") == "2026-12-28"\n\n\n'
        "def test_merchants():\n"
        f"    assert len(MERCHANTS) == {len(merchants())}\n"
        f'    assert merchant_category("CARD PAYMENT {name.upper()} 0042") == {cat!r}\n'
        '    assert merchant_category("zzzz") is None\n'
    )


def files() -> dict:
    out = {f"src/ledgerkit/sync/{p[0]}.py": provider_module(p) for p in PROVIDERS}
    out["src/ledgerkit/sync/__init__.py"] = _registry("sync", "Bank API providers, by name.",
                                                      [p[0] for p in PROVIDERS], "PROVIDER", "provider")
    out["src/ledgerkit/sync/runner.py"] = SYNC_RUNNER
    out["src/ledgerkit/data/__init__.py"] = '"""Reference data tables."""\n'
    out["src/ledgerkit/data/currencies.py"] = currencies_module()
    out["src/ledgerkit/calendar/__init__.py"] = '"""Business-day calendars."""\n'
    out["src/ledgerkit/calendar/holidays.py"] = holidays_module()
    out["src/ledgerkit/rules/merchants.py"] = merchants_module()
    out["tests/unit/test_sync.py"] = sync_tests()
    out["tests/unit/test_data.py"] = data_tests()
    return out
