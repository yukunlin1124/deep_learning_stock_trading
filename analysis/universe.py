"""Hand-curated TWSE large-cap universe.

Selected from well-known Taiwan blue-chip code ranges:
- 1101-1326: foods, materials, plastics, textiles
- 2002-2208: steel, autos, transport
- 2301-2498: semiconductors, electronics, LCDs
- 2801-2891: banks, financial holdings
- High-recognition individual tech / consumer names

The list is filtered against the live symbol map at runtime, so any
delisted/renamed code is dropped automatically.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stock_project_for_class"))
from stock_api import load_symbol_map  # noqa: E402

# Hand-picked code list (~140 entries pre-filter).
CANDIDATE_CODES: list[str] = (
    # Foods, plastics, textiles, materials
    [str(c) for c in range(1101, 1111)]
    + [str(c) for c in range(1201, 1218)]
    + [str(c) for c in range(1301, 1322)]
    + [str(c) for c in range(1402, 1410)]
    # Steel, machinery, transport, autos
    + [str(c) for c in range(2002, 2010)]
    + [str(c) for c in range(2104, 2110)]
    + [str(c) for c in range(2201, 2208)]
    + ["2207", "2227"]
    # Electronics: semiconductors, LCDs, EMS, components
    + ["2301", "2303", "2308", "2317", "2324", "2327", "2330", "2337",
       "2344", "2347", "2353", "2354", "2356", "2357", "2360", "2376",
       "2379", "2382", "2383", "2385", "2388", "2392", "2395", "2408",
       "2409", "2412", "2421", "2439", "2441", "2449", "2451", "2454",
       "2474", "2480", "2492", "2498"]
    # Finance / banks / FHCs
    + ["2801", "2809", "2812", "2816", "2820", "2823", "2832", "2834",
       "2845", "2849", "2867", "2880", "2881", "2882", "2883", "2884",
       "2885", "2886", "2887", "2888", "2890", "2891", "2892", "5880"]
    # Selected mid/large tech and consumer
    + ["3008", "3034", "3037", "3045", "3231", "3406", "3443", "3481",
       "3673", "3711", "4904", "4938", "5871", "6239", "6415", "6505",
       "8046", "8454", "9904", "9910", "9921", "9933", "9945"]
)


def get_universe() -> list[str]:
    """Return ordered list of valid TWSE 4-digit codes in our universe."""
    m = load_symbol_map()
    out: list[str] = []
    seen: set[str] = set()
    for code in CANDIDATE_CODES:
        if code in seen:
            continue
        info = m.get(code)
        if info is None:
            continue
        if info.get("type") != "TWSE":
            continue
        out.append(code)
        seen.add(code)
    return out


if __name__ == "__main__":
    u = get_universe()
    print(f"universe size: {len(u)}")
    print(u)
