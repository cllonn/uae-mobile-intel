"""
One-time (re-runnable) step: extract OSM place names (neighbourhood/suburb/quarter/village/
town/city, whichever exists nearest) from the already-downloaded OSM GCC States PBF, resolve
one human-readable `area_name` per H3 zone by nearest-place-node distance, and write it onto
`zone_quarter_table.parquet` -- the single shared table every downstream script (run_pipeline.py
-> zone_priority.parquet -> build_dashboard.py and copilot_tools.py) already reads. This
replaces the dashboard's truncated-H3-fragment zone label ("Zone 43a392") with a real local
place name ("Muwaileh"), mirroring exactly how scripts/build_emirate_field.py added `emirate`.

Why OSM place nodes (not a new geocoding service): the GCC PBF is already in the repo, already
licensed and attributed (ODbL), and carries `place=neighbourhood/suburb/quarter/village/town/
city` nodes with `name`/`name:en` tags -- no new dataset, API key, or licence to document, and
no live geocoding at dashboard load time (the brief's own "every number traceable/reproducible"
requirement applies just as much to a label as to a score: it must be computed once, on disk,
not re-derived or guessed at render time).

Method: nearest-place-node assignment, not point-in-polygon -- OSM neighbourhood/suburb
boundaries (where they exist as ways/relations at all) are far patchier in this extract than
POINT place nodes are, so this uses the same "one representative point per feature, matched by
distance" pattern the rest of this repo's OSM extraction already relies on
(07_osm_feature_extraction.ipynb's road midpoint, this file's own centroid-in-polygon emirate
join). Finer-grained tags (neighbourhood/suburb/quarter) are preferred over coarser ones
(town/city) when both are within reach, each within its own max-distance bound -- see
PLACE_TIERS below -- so a zone is never labelled with a city name when a real, nearby
neighbourhood name is available, but a genuinely under-mapped area still gets a real (if
coarser) name rather than none. A zone with no place node within any tier's bound gets an
explicit, honest "Unnamed area -- <emirate>" fallback (never a guessed or interpolated name).

Run: python -m scripts.build_area_name_field
"""
import re
import sys
from pathlib import Path

import geopandas as gpd
import h3
import osmium
import osmium.filter as ofilter
import pandas as pd
from shapely.geometry import Point

# A handful of OSM place nodes in this extract carry only a native-script `name` (no
# `name:en`, checked against Nominatim -- these small settlements have no established English
# form anywhere). Every area name the dashboard/copilot shows must be Latin-script, so each is
# romanized here by pronunciation, never translated by meaning (e.g. "حارة الشندغة" ->
# "Harat Al Shindagha", not "Al Shindagha Quarter"). This table is hand-checked against the
# exact strings this extract actually produces; ARABIC_FALLBACK_MAP below is a best-effort
# letter-map safety net in case a PBF refresh ever surfaces a name not in this table.
ARABIC_NAME_TRANSLITERATIONS = {
    "الخيس": "Al Khees",
    "اليارية": "Al Yariyah",
    "بو حصا": "Bu Hasa",
    "جريرة": "Jareera",
    "خضراء السلم": "Khadraa Al Salam",
    "حارة الشندغة": "Harat Al Shindagha",
    "محضر بن عصيان": "Mahdar Bin Isyan",
}

_ARABIC_RE = re.compile(r"[؀-ۿ]")

# Informal consonant/long-vowel map (unvocalized Arabic script has no short vowels to recover
# automatically) -- good enough to guarantee Latin script for a name this table doesn't cover,
# not a substitute for hand-checking a real transliteration into the table above.
ARABIC_FALLBACK_MAP = {
    "ا": "a", "أ": "a", "إ": "i", "آ": "aa", "ب": "b", "ت": "t", "ث": "th", "ج": "j",
    "ح": "h", "خ": "kh", "د": "d", "ذ": "dh", "ر": "r", "ز": "z", "س": "s", "ش": "sh",
    "ص": "s", "ض": "d", "ط": "t", "ظ": "z", "ع": "'", "غ": "gh", "ف": "f", "ق": "q",
    "ك": "k", "ل": "l", "م": "m", "ن": "n", "ه": "h", "و": "w", "ي": "y", "ة": "a",
    "ء": "'", "ى": "a",
}


def transliterate_if_arabic(name: str) -> str:
    """Romanize a native-script OSM place name by pronunciation. Curated table first (every
    Arabic-script name actually present in this extract is in there); anything unrecognised
    falls through the letter-map so raw Arabic script never reaches the dashboard."""
    if not _ARABIC_RE.search(name):
        return name
    if name in ARABIC_NAME_TRANSLITERATIONS:
        return ARABIC_NAME_TRANSLITERATIONS[name]
    romanized = "".join(ARABIC_FALLBACK_MAP.get(ch, ch) for ch in name.replace("ال", "Al "))
    return " ".join(w.capitalize() for w in romanized.split())

PBF_PATH = Path("data/raw/osm/gcc-states-latest.osm.pbf")
PLACE_NODES_OUT = Path("data/raw/boundary/uae_place_nodes.geojson")
AREA_NAMES_OUT = Path("data/processed/area_names_uae.parquet")
ZONE_QUARTER_PATH = Path("data/processed/zone_quarter_table.parquet")
POPULATION_ZONES_PATH = Path("data/processed/population_zones_uae.parquet")
EMIRATE_ZONES_PATH = Path("data/processed/emirate_zones_uae.parquet")
BOUNDARY_PATH = Path("data/raw/boundary/uae_boundary.geojson")

EQUAL_AREA_CRS = "EPSG:6933"  # never UTM 40N -- brief flags up to ~0.7% overstatement in western UAE

# Finer-grained place tags are preferred over coarser ones, but only within their own reach --
# a genuine neighbourhood name up to 10km away beats a city name that happens to be closer, but
# a city/town name is still a real, honest label for a zone with no mapped neighbourhood at all
# (common in industrial/rural/edge peer-group zones -- see project_gt_challenge memory on why
# those 4 peer groups exist). Order here is preference order, not a hard cutoff by itself; each
# tier's own max distance (below) is what actually gates whether it's accepted.
PLACE_TIERS: list[tuple[str, float]] = [
    ("neighbourhood", 10.0),
    ("suburb", 10.0),
    ("quarter", 10.0),
    ("village", 20.0),
    ("town", 20.0),
    ("hamlet", 20.0),
    ("city", 30.0),
]
PLACE_TAGS = [tag for tag, _ in PLACE_TIERS]


def extract_place_nodes() -> gpd.GeoDataFrame:
    """Single streaming pass over the GCC PBF, filtered in C++ (osmium.filter.KeyFilter) to
    nodes carrying a `place` tag -- the same "let the C++ layer skip untagged nodes" pattern
    07_osm_feature_extraction.ipynb uses for POIs. A `place` node's own coordinate IS its
    representative point (no polygon/centroid step needed, unlike buildings/roads)."""

    class PlaceHandler(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.records = []

        def node(self, n):
            if not n.location.valid():
                return
            tags = dict(n.tags)
            place = tags.get("place")
            if place not in PLACE_TAGS:
                return
            name = tags.get("name:en") or tags.get("name")
            if not name:
                return
            self.records.append({
                "place": place, "name": name,
                "lon": n.location.lon, "lat": n.location.lat,
            })

    print("Streaming the GCC PBF for place=* nodes (filtered in C++, this is fast)...")
    handler = PlaceHandler()
    handler.apply_file(str(PBF_PATH), filters=[ofilter.KeyFilter("place")])
    print(f"  {len(handler.records)} named place nodes across the GCC extract")

    gdf = gpd.GeoDataFrame(
        handler.records,
        geometry=[Point(r["lon"], r["lat"]) for r in handler.records],
        crs="EPSG:4326",
    )

    uae_boundary = gpd.read_file(BOUNDARY_PATH).to_crs("EPSG:4326")
    kept_idx = gpd.sjoin(gdf, uae_boundary[["geometry"]], predicate="within", how="inner").index
    gdf = gdf.loc[kept_idx].reset_index(drop=True)
    print(f"  {len(gdf)} within the real UAE boundary polygon (never a bounding box)")
    print("  by place tag:\n" + gdf["place"].value_counts().to_string())
    return gdf


def assign_area_names(place_gdf: gpd.GeoDataFrame, h3_cells: list[str],
                       emirate_by_cell: dict[str, str]) -> pd.DataFrame:
    """For each H3 cell centroid, the nearest place node within its tier's own max-distance
    bound, preferring finer tiers (PLACE_TIERS order) over coarser ones -- never just "the
    single nearest point regardless of tag". A cell with nothing within any tier's bound falls
    back to an explicit 'Unnamed area -- <emirate>' label rather than a guessed name."""
    centroids = pd.DataFrame(
        [(cell, *h3.cell_to_latlng(cell)) for cell in h3_cells],
        columns=["h3_cell", "lat", "lon"],
    )
    points = gpd.GeoDataFrame(
        centroids, geometry=gpd.points_from_xy(centroids["lon"], centroids["lat"]), crs="EPSG:4326"
    ).to_crs(EQUAL_AREA_CRS)

    place_proj = place_gdf.to_crs(EQUAL_AREA_CRS)

    best_name = {c: None for c in h3_cells}
    best_place_tag = {c: None for c in h3_cells}
    best_dist_km = {c: None for c in h3_cells}

    remaining = points.copy()
    for tag, max_km in PLACE_TIERS:
        if remaining.empty:
            break
        tier_points = place_proj[place_proj["place"] == tag]
        if tier_points.empty:
            continue
        nearest = gpd.sjoin_nearest(
            remaining[["h3_cell", "geometry"]], tier_points[["name", "geometry"]],
            distance_col="dist_m",
        )
        nearest = nearest.drop_duplicates(subset="h3_cell")  # a tie can otherwise duplicate a row
        within_reach = nearest[nearest["dist_m"] <= max_km * 1000]
        for _, row in within_reach.iterrows():
            best_name[row["h3_cell"]] = row["name"]
            best_place_tag[row["h3_cell"]] = tag
            best_dist_km[row["h3_cell"]] = round(row["dist_m"] / 1000, 2)
        remaining = remaining[~remaining["h3_cell"].isin(within_reach["h3_cell"])]

    records = []
    for cell in h3_cells:
        emirate = emirate_by_cell.get(cell)
        if best_name[cell] is not None:
            records.append({
                "h3_cell": cell, "area_name": best_name[cell],
                "area_name_source": best_place_tag[cell],
                "area_name_distance_km": best_dist_km[cell],
                "area_name_is_fallback": False,
            })
        else:
            fallback_label = f"Unnamed area – {emirate}" if emirate else "Unnamed area"
            records.append({
                "h3_cell": cell, "area_name": fallback_label,
                "area_name_source": "fallback", "area_name_distance_km": None,
                "area_name_is_fallback": True,
            })
    return pd.DataFrame(records)


def main():
    if PLACE_NODES_OUT.exists():
        print(f"Reusing cached {PLACE_NODES_OUT} (delete it to force re-extraction from the PBF)")
        place_gdf = gpd.read_file(PLACE_NODES_OUT)
    else:
        place_gdf = extract_place_nodes()
        PLACE_NODES_OUT.parent.mkdir(parents=True, exist_ok=True)
        place_gdf.to_file(PLACE_NODES_OUT, driver="GeoJSON")
        print(f"Saved: {PLACE_NODES_OUT}")

    n_before = place_gdf["name"].apply(lambda s: bool(_ARABIC_RE.search(s))).sum()
    place_gdf["name"] = place_gdf["name"].apply(transliterate_if_arabic)
    print(f"Romanized {n_before} native-script place names to Latin script")

    zq = pd.read_parquet(ZONE_QUARTER_PATH)
    pop = pd.read_parquet(POPULATION_ZONES_PATH)
    emirate_zones = pd.read_parquet(EMIRATE_ZONES_PATH)
    emirate_by_cell = emirate_zones.set_index("h3_cell")["emirate"].to_dict()

    # Same "measured OR populated" universe as scripts/build_emirate_field.py -- an area name is
    # a static, per-cell geographic fact like emirate, not something that should vary by whether
    # a given quarter happened to have Ookla samples.
    all_cells = sorted(set(zq["h3_cell"]) | set(pop["h3_cell"]))
    print(f"\nResolving area names for {len(all_cells)} unique H3 zones "
          f"({zq['h3_cell'].nunique()} measured + {pop['h3_cell'].nunique()} populated, "
          f"{len(all_cells)} in their union)...")
    area_names = assign_area_names(place_gdf, all_cells, emirate_by_cell)

    AREA_NAMES_OUT.parent.mkdir(parents=True, exist_ok=True)
    area_names.to_parquet(AREA_NAMES_OUT, index=False)
    print(f"Saved: {AREA_NAMES_OUT}")

    n_real = (~area_names["area_name_is_fallback"]).sum()
    n_fallback = area_names["area_name_is_fallback"].sum()
    print(f"\n{n_real} of {len(area_names)} zones got a real OSM place name "
          f"({100 * n_real / len(area_names):.1f}%); {n_fallback} fell back to 'Unnamed area'.")
    print("\nBy source tag:")
    print(area_names["area_name_source"].value_counts().to_string())

    # Merge onto the shared master table -- the single source of truth every downstream script
    # (run_pipeline.py, build_dashboard.py, copilot_tools.py) reads from. Same pattern as
    # scripts/build_emirate_field.py's own merge, right down to the drop-if-rerun guard.
    if "area_name" in zq.columns:
        zq = zq.drop(columns=["area_name", "area_name_source", "area_name_distance_km", "area_name_is_fallback"])
    zq = zq.merge(area_names, on="h3_cell", how="left")
    zq.to_parquet(ZONE_QUARTER_PATH, index=False)
    print(f"\nMerged 'area_name' onto {ZONE_QUARTER_PATH} ({len(zq)} rows). "
          f"Re-run `python -m src.run_pipeline` next to propagate it into zone_priority.parquet.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
