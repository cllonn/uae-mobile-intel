"""
One-time (re-runnable) step: extract the 7 UAE emirate administrative boundaries from the
already-downloaded OSM GCC States PBF, assign every H3 zone to one emirate by centroid-in-
polygon, and write `emirate` onto `zone_quarter_table.parquet` -- the single shared table
every downstream script (run_pipeline.py -> zone_priority.parquet -> build_dashboard.py and
copilot_tools.py) already reads. This replaces the dashboard-only nearest-city-anchor
approximation with one authoritative field, computed once.

Why OSM (not a new boundary source): the GCC PBF is already in the repo, already licensed and
attributed (ODbL), and carries clean `ISO3166-2` tags (AE-AZ, AE-DU, AE-SH, AE-AJ, AE-UQ, AE-RK,
AE-FU) on the admin_level=4 boundary relations -- no new dataset or licence to document.

Run: python -m scripts.build_emirate_field
"""
import io
import sys
from pathlib import Path

import geopandas as gpd
import h3
import osmium
import osmium.geom as ogeom
import pandas as pd
import shapely.wkb as wkblib
from shapely.ops import polygonize, unary_union

PBF_PATH = Path("data/raw/osm/gcc-states-latest.osm.pbf")
BOUNDARY_OUT = Path("data/raw/boundary/uae_emirates.geojson")
EMIRATE_ZONES_OUT = Path("data/processed/emirate_zones_uae.parquet")
ZONE_QUARTER_PATH = Path("data/processed/zone_quarter_table.parquet")
POPULATION_ZONES_PATH = Path("data/processed/population_zones_uae.parquet")

# admin_level=4 boundary relations for the 7 emirates, keyed by their OSM ISO3166-2 tag --
# more reliable than matching on `name`/`name:en`, which mix Arabic, English, and
# "Ras al-Khaimah" vs. "Ras Al Khaimah" -style spelling variants across tags.
ISO_TO_EMIRATE = {
    "AE-AZ": "Abu Dhabi", "AE-DU": "Dubai", "AE-SH": "Sharjah", "AE-AJ": "Ajman",
    "AE-UQ": "Umm Al Quwain", "AE-RK": "Ras Al Khaimah", "AE-FU": "Fujairah",
}


def extract_emirate_boundaries() -> gpd.GeoDataFrame:
    """Two-pass osmium read: pass 1 finds the 7 target relations and the way IDs they
    reference (role=outer only -- emirate boundaries have no meaningful inner holes, so
    inner-role members are dropped rather than adding multipolygon-hole handling for a
    case that doesn't occur here); pass 2 resolves those ways' geometry (needs
    locations=True to look up node coordinates) and assembles each emirate's boundary
    from its member line segments via shapely.polygonize, the same "treat as ways, not
    full multipolygon areas" simplification 07_osm_feature_extraction.ipynb already uses
    for buildings/landuse in this repo."""

    class RelationPass(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            # way_id -> list of emirates. A shared inter-emirate border way is legitimately
            # a member of *two* relations (e.g. the RAK/Fujairah boundary) -- mapping to a
            # single emirate would silently drop that segment from whichever emirate lost
            # the overwrite, leaving its ring unclosed (which is exactly what happened here).
            self.way_to_emirates: dict[int, list[str]] = {}

        def relation(self, r):
            tags = dict(r.tags)
            if tags.get("boundary") != "administrative" or tags.get("admin_level") != "4":
                return
            emirate = ISO_TO_EMIRATE.get(tags.get("ISO3166-2"))
            if emirate is None:
                return
            for m in r.members:
                if m.type == "w" and m.role == "outer":
                    self.way_to_emirates.setdefault(m.ref, []).append(emirate)

    class WayPass(osmium.SimpleHandler):
        def __init__(self, way_to_emirates):
            super().__init__()
            self.way_to_emirates = way_to_emirates
            all_emirates = {e for emirates in way_to_emirates.values() for e in emirates}
            self.lines_by_emirate = {name: [] for name in all_emirates}
            self._wkbfab = ogeom.WKBFactory()

        def way(self, w):
            emirates = self.way_to_emirates.get(w.id)
            if emirates is None:
                return
            try:
                wkb = self._wkbfab.create_linestring(w)
            except RuntimeError:
                return
            line = wkblib.loads(wkb, hex=True)
            for emirate in emirates:
                self.lines_by_emirate[emirate].append(line)

    print("Pass 1/2: scanning relations for the 7 emirate boundaries...")
    rel_pass = RelationPass()
    rel_pass.apply_file(str(PBF_PATH))
    found = {e for emirates in rel_pass.way_to_emirates.values() for e in emirates}
    missing = set(ISO_TO_EMIRATE.values()) - found
    if missing:
        raise RuntimeError(f"Could not find boundary relations for: {missing}")
    print(f"  {len(rel_pass.way_to_emirates)} member ways across {len(found)} emirates")

    print("Pass 2/2: resolving way geometry (needs node locations, this is the slow pass)...")
    way_pass = WayPass(rel_pass.way_to_emirates)
    way_pass.apply_file(str(PBF_PATH), locations=True)

    records = []
    for emirate, lines in way_pass.lines_by_emirate.items():
        # polygonize() nodes and rings the raw line segments itself -- pre-merging with
        # unary_union first (tried initially) can dissolve shared edges between separate
        # rings (e.g. mainland + an island part, as Ras Al Khaimah has) and leaves no
        # closed ring to polygonize at all. Feed it the raw ways directly instead.
        polygons = list(polygonize(lines))
        if not polygons:
            raise RuntimeError(f"{emirate}: no closed polygon could be assembled from its boundary ways")
        records.append({"emirate": emirate, "geometry": unary_union(polygons)})

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    print(gdf["emirate"].tolist())
    return gdf


def assign_zones_to_emirates(emirate_gdf: gpd.GeoDataFrame, h3_cells: list[str]) -> pd.DataFrame:
    """Centroid-in-polygon join: one point per H3 cell (h3.cell_to_latlng), matched against
    the 7 emirate polygons. A handful of coastal/edge cells can miss every polygon (the
    union of emirate boundaries doesn't perfectly match the national boundary used
    elsewhere in this pipeline, e.g. small islands) -- those fall back to nearest emirate
    by distance, so every zone still gets exactly one of the 7 labels, never "Unknown"."""
    centroids = pd.DataFrame(
        [(cell, *h3.cell_to_latlng(cell)) for cell in h3_cells],
        columns=["h3_cell", "lat", "lon"],
    )
    points = gpd.GeoDataFrame(
        centroids, geometry=gpd.points_from_xy(centroids["lon"], centroids["lat"]), crs="EPSG:4326"
    )

    joined = gpd.sjoin(points, emirate_gdf[["emirate", "geometry"]], how="left", predicate="within")
    joined = joined.drop_duplicates(subset="h3_cell")  # a point exactly on a shared border could match twice

    unmatched = joined["emirate"].isna()
    if unmatched.any():
        print(f"  {unmatched.sum()} zone(s) missed every polygon (edge/coastal) -- assigning nearest emirate")
        nearest = gpd.sjoin_nearest(points.loc[unmatched, ["h3_cell", "geometry"]], emirate_gdf[["emirate", "geometry"]])
        joined.loc[unmatched, "emirate"] = joined.loc[unmatched, "h3_cell"].map(
            nearest.set_index("h3_cell")["emirate"]
        )

    return joined[["h3_cell", "emirate"]].reset_index(drop=True)


def main():
    if BOUNDARY_OUT.exists():
        print(f"Reusing cached {BOUNDARY_OUT} (delete it to force re-extraction from the PBF)")
        emirate_gdf = gpd.read_file(BOUNDARY_OUT)
    else:
        emirate_gdf = extract_emirate_boundaries()
        BOUNDARY_OUT.parent.mkdir(parents=True, exist_ok=True)
        emirate_gdf.to_file(BOUNDARY_OUT, driver="GeoJSON")
        print(f"Saved: {BOUNDARY_OUT}")

    zq = pd.read_parquet(ZONE_QUARTER_PATH)
    pop = pd.read_parquet(POPULATION_ZONES_PATH)
    # Union of every H3 cell Ookla ever measured *and* every H3 cell WorldPop records any
    # population in -- not just the measured subset. Population denominators (e.g. "% of
    # Dubai's population represented") need the full populated universe, not only the cells
    # that happen to have a Speedtest sample; this is the same measured-vs-eligible
    # distinction the T0 coverage audit already makes.
    all_cells = sorted(set(zq["h3_cell"]) | set(pop["h3_cell"]))
    print(f"\nAssigning {len(all_cells)} unique H3 zones to emirates "
          f"({zq['h3_cell'].nunique()} measured + {pop['h3_cell'].nunique()} populated, "
          f"{len(all_cells)} in their union)...")
    emirate_zones = assign_zones_to_emirates(emirate_gdf, all_cells)

    EMIRATE_ZONES_OUT.parent.mkdir(parents=True, exist_ok=True)
    emirate_zones.to_parquet(EMIRATE_ZONES_OUT, index=False)
    print(f"Saved: {EMIRATE_ZONES_OUT}")

    print("\nZone counts by emirate:")
    counts = emirate_zones["emirate"].value_counts()
    print(counts.to_string())
    missing_emirates = set(ISO_TO_EMIRATE.values()) - set(counts.index)
    if missing_emirates:
        print(f"WARNING: these emirates got zero zones: {missing_emirates}")
    else:
        print("\nAll 7 emirates present, including Umm Al Quwain -- confirmed.")

    # Merge onto the shared master table -- the single source of truth every downstream
    # script (run_pipeline.py, build_dashboard.py, copilot_tools.py) reads from.
    if "emirate" in zq.columns:
        zq = zq.drop(columns=["emirate"])
    zq = zq.merge(emirate_zones, on="h3_cell", how="left")
    zq.to_parquet(ZONE_QUARTER_PATH, index=False)
    print(f"\nMerged 'emirate' onto {ZONE_QUARTER_PATH} ({len(zq)} rows). "
          f"Re-run `python -m src.run_pipeline` next to propagate it into zone_priority.parquet.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
