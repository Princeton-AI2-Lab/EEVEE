import hashlib
import re


def safe_filename_fragment(text: str, fallback: str = "file") -> str:
    """Convert arbitrary text into a filesystem-safe filename fragment."""
    safe = re.sub(r"[^\w.-]+", "_", text, flags=re.UNICODE).strip("._")
    return safe or fallback


def make_unique_filename_fragment(
    text: str,
    used_fragments: set[str],
    fallback: str = "file",
) -> str:
    """Return a collision-safe filename fragment and record it in used_fragments."""
    base = safe_filename_fragment(text, fallback=fallback)
    fragment = base
    if fragment in used_fragments:
        suffix = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
        fragment = f"{base}_{suffix}"
        i = 2
        while fragment in used_fragments:
            fragment = f"{base}_{suffix}_{i}"
            i += 1
    used_fragments.add(fragment)
    return fragment
