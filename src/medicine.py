"""Medicine lookup using the free openFDA drug label API (https://open.fda.gov/apis/drug/label/).

Returns uses, warnings and side effects only. Dosage sections are deliberately left out.
openFDA covers US-labelled medicines; for local brand names, search by the generic name
(e.g. "paracetamol" -> try "acetaminophen").
"""
import logging
import re
import threading
import time

import requests

log = logging.getLogger(__name__)

_API = "https://api.fda.gov/drug/label.json"
_SECTIONS = [
    ("purpose", "Purpose"),
    ("indications_and_usage", "Uses"),
    ("warnings", "Warnings"),
    ("boxed_warning", "Boxed warning"),
    ("do_not_use", "Do not use if"),
    ("ask_doctor", "Ask a doctor before use if"),
    ("stop_use", "Stop use and ask a doctor if"),
    ("adverse_reactions", "Side effects"),
    ("pregnancy_or_breast_feeding", "Pregnancy / breast-feeding"),
]
# Common names outside the US -> US generic name
_ALIASES = {"paracetamol": "acetaminophen", "panadol": "acetaminophen", "salbutamol": "albuterol",
            "adrenaline": "epinephrine", "brufen": "ibuprofen", "disprin": "aspirin"}

_cache: dict[str, tuple[float, dict | None]] = {}
_cache_lock = threading.Lock()
_TTL = 6 * 3600


def _clean(text: str, limit: int = 900) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def lookup(name: str) -> dict | None:
    key = re.sub(r"[^a-z0-9 \-]", "", name.lower()).strip()
    if not key:
        return None
    key = _ALIASES.get(key, key)
    with _cache_lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < _TTL:
            return hit[1]

    q = f'openfda.generic_name:"{key}" OR openfda.brand_name:"{key}" OR openfda.substance_name:"{key}"'
    try:
        resp = requests.get(_API, params={"search": q, "limit": 1}, timeout=8)
        if resp.status_code == 404:
            result = None
        else:
            resp.raise_for_status()
            item = resp.json()["results"][0]
            fda = item.get("openfda", {})
            sections = []
            for field, label in _SECTIONS:
                if item.get(field):
                    sections.append({"label": label, "text": _clean(" ".join(item[field]))})
            result = {
                "query": name,
                "brand": ", ".join(fda.get("brand_name", [])[:3]) or None,
                "generic": ", ".join(fda.get("generic_name", [])[:3]) or key,
                "route": ", ".join(fda.get("route", [])[:3]) or None,
                "manufacturer": ", ".join(fda.get("manufacturer_name", [])[:1]) or None,
                "sections": sections,
                "source": "U.S. FDA drug label (openFDA)",
            }
    except requests.RequestException:
        log.warning("openFDA lookup failed", exc_info=True)
        raise

    with _cache_lock:
        if len(_cache) > 500:
            _cache.clear()
        _cache[key] = (time.time(), result)
    return result
