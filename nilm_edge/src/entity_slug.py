from __future__ import annotations

import re
import unicodedata


ENTITY_SLUG_FALLBACK = "appliance"
LATIN_TRANSLITERATION = str.maketrans({
    "\u00c6": "Ae",
    "\u00e6": "ae",
    "\u00d0": "D",
    "\u00f0": "d",
    "\u0110": "D",
    "\u0111": "d",
    "\u0141": "L",
    "\u0142": "l",
    "\u00d8": "O",
    "\u00f8": "o",
    "\u00de": "Th",
    "\u00fe": "th",
    "\u0152": "Oe",
    "\u0153": "oe",
    "\u00df": "ss",
})


def slugify_entity_suffix(value: str) -> str:
    text = str(value or "").strip().translate(LATIN_TRANSLITERATION)
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", ascii_text).strip("_").lower()
    slug = re.sub(r"_+", "_", slug)
    return slug or ENTITY_SLUG_FALLBACK
