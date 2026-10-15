
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pydeck as pdk
import streamlit as st

# --- Config -----------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "_data"

# Boundary files
BOROUGHS_FILE = DATA_DIR / "Borough_Boundaries_20261012.geojson"
CDS_FILE = DATA_DIR / "Community_Districts_20261012.geojson"

# Feature files
SIP_INTERSECTIONS_FILE = DATA_DIR / "VZV_Street_Improvement_Projects_(SIP)_Intersections_20261011.geojson"
SIP_CORRIDORS_FILE = DATA_DIR / "VZV_Street_Improvement_Projects_(SIP)_Corridor_20261011.geojson"
CAPITAL_INTERSECTIONS_FILE = DATA_DIR / "Street_and_Highway_Capital_Reconstruction_Projects_-_Intersection_20261013.geojson"

# Map initial setup
MAP_VIEW = pdk.ViewState(latitude=40.73, longitude=-73.95, zoom=10)


# --- Layer Configurations ----------------------------------------------------------

# Feature data to visualize
LAYERS = {
    "sip_intersections": {
        "label": "Intersection projects",
        "chart_name": "Intersections",
        "rgb": (200, 30, 0)
    },
    "sip_corridors": {
        "label": "Corridor projects",
        "chart_name": "Corridors",
        "rgb": (200, 200, 0)
    },
    # "capital_intersections": {
    #     "label": "Capital Intersection projects",
    #     "chart_name": "Capital Intersections",
    #     "rgb": (66, 135, 245)
    # }
}

# Geographic areas for filtering and showing visual borders
AREAS = {
    "borough": {
        "file": BOROUGHS_FILE,
        "col": "boroname",
        "display_col": "boroname",
        "label": "Borough",
        "rgb": (20, 128, 17),
        "width": 10
    },
    "cd": {
        "file": CDS_FILE,
        "col": "boro_cd",
        "display_col": "cd_name",   # REQUIRES DECODING
        "label": "Community District",
        "rgb": (20, 128, 17),
        "width": 10
    },
}

# --- Helpers ----------------------------------------------------------
def rgba(key, alpha=200):
    """List form for pydeck, e.g. [200, 30, 0, 160]."""
    return [*LAYERS[key]["rgb"], alpha]

def css_rgb(key):
    """String form for the HTML swatch."""
    r, g, b = LAYERS[key]["rgb"]
    return f"rgb({r}, {g}, {b})"

def hex_color(key):
    """Hex form for st.bar_chart."""
    return "#{:02x}{:02x}{:02x}".format(*LAYERS[key]["rgb"])

# Borough code to name mapping
BORO_CODE_TO_NAME = {
    1: "Manhattan",
    2: "Bronx",
    3: "Brooklyn",
    4: "Queens",
    5: "Staten Island",
}

# Decodes community district codes into names (e.g., 101 -> "Manhattan 1")
def cd_name_from_code(code):
    code = int(code)
    boro = BORO_CODE_TO_NAME[code // 100]   # gets boro code and maps it to the name
    district = code % 100   # gets the district number
    return f"{boro} {district}"

# Sorts community district names intuitively
def cd_sort_key(name):
    """'Bronx 10' -> ('Bronx', 10), so districts sort numerically within a borough."""
    boro, num = name.rsplit(" ", 1)
    return (boro, int(num))


# --- Data loading -----------------------------------------------------------

# Load GeoJSON file into GeoDataFrame, dropping empty geometries
@st.cache_data
def read_geo(path):
    """
    Generic: any GeoJSON -> GeoDataFrame in lon/lat, no empty geometries.
    Returns geodataframe
    """
    gdf = gpd.read_file(path)

    # Reproject if needed
    if gdf.crs is None:
        print(f"Warning: {path.name} has no CRS defined. Cannot safely convert.")
    else:
        if gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(4326)

    # Check geometry validity
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        print(f"Warning: {invalid.sum()} invalid geometries in {path.name}")

    return gdf[gdf.geometry.notna()].copy() # drop empty geometry rows

# Prepare boundary layer with special set up
@st.cache_data
def load_areas(key):
    """
    Boundary layer reduced to its ID column and geometry.
    Returns modified geodataframe and display column
    """
    cfg = AREAS[key]    # gets config dictionary for layer details

    # LOAD DATA
    # Load data to gdf with only specified column and geometry
    area = read_geo(cfg["file"])[[cfg["col"], "geometry"]]

    # POST-PROCESS DATA
    # Clean up polygon geometry (STILL NEEDED?)
    area["geometry"] = area.buffer(0)

    # CDs require decoding for more readable names
    if key == "cd":
        # Create new cd_name column and store decoded name
        area["cd_name"] = area[cfg["col"]].apply(cd_name_from_code)

    # Determine which column should be used for tagging
    display_col = cfg.get("display_col", cfg["col"])

    return area, display_col

# Join areas to features to facilitate data filtering
def tag_areas(gdf, areas, col):
    # Determine spatial predicate
    if gdf.geometry.iloc[0].geom_type == "Point":
        predicate = "within"
    else:
        predicate = "intersects"

    # Spatial join
    joined = gpd.sjoin(gdf[["geometry"]], areas, how="left", predicate=predicate)

    # Collect unique area names per project
    def collect_areas(values):
        values = [v for v in values if v is not None]
        return pd.Series(values).explode().unique().tolist()

    by_project = joined.groupby(level=0)
    area_values = by_project[col]

    return area_values.agg(collect_areas)


@st.cache_data
def load_projects(path):
    """Project layer: dates, plus one list-of-areas column per area layer."""
    gdf = read_geo(path)

    # GET DATE INFORMATION
    # Store end_date as datetime
    gdf["end_date"] = pd.to_datetime(gdf["end_date"], errors="coerce")
    # Store end_year by extracting from end_date datetime
    gdf["end_year"] = gdf["end_date"].dt.year.astype("Int64")

    # Join areas to projects
    for key, cfg in AREAS.items():
        area, display_col = load_areas(key)
        # Add column to geodataframe with list of area names for each project
        gdf[f"area_{key}"] = tag_areas(gdf, area, display_col)
    return gdf


def to_geojson(gdf):
    """Map-only copy as a GeoJSON dict (pydeck can't serialize datetimes or lists)."""
    out = gdf.copy()

    # Convert datetimes to strings
    for col in out.select_dtypes(include=["datetime", "datetimetz"]).columns:
        out[col] = out[col].astype(str)

    # Convert list columns to comma-separated strings
    for col in out.columns:
        if out[col].apply(lambda x: isinstance(x, list)).any():
            out[col] = out[col].apply(lambda lst: ", ".join(lst) if isinstance(lst, list) else lst)

    # Convert pandas NA to None
    out = out.where(pd.notnull(out), None)

    return out.__geo_interface__


def polygons_for_pydeck(gdf, display_col):
    """Convert GeoDataFrame polygons into a list of dicts for PolygonLayer."""
    polys = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        props = {display_col: row[display_col]}

        if geom.geom_type == "Polygon":
            polys.append({
                "polygon": list(geom.exterior.coords),
                "properties": props
            })

        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                polys.append({
                    "polygon": list(poly.exterior.coords),
                    "properties": props
                })

    return polys


# GeoDataFrames
sip_intersections = load_projects(SIP_INTERSECTIONS_FILE)
sip_corridors = load_projects(SIP_CORRIDORS_FILE)


# --- Sidebar controls -------------------------------------------------------
st.sidebar.header("Layers")

def layer_toggle(key):
    """Add toggle with color-matched swatch."""
    swatch_col, toggle_col = st.sidebar.columns([1, 6], vertical_alignment="center")
    swatch_col.markdown(
        f'<div style="width:16px;height:16px;border-radius:3px;'
        f'background:{css_rgb(key)}"></div>',
        unsafe_allow_html=True,
    )
    return toggle_col.toggle(LAYERS[key]["label"], value=True)

show_intersections = layer_toggle("sip_intersections")
show_corridors = layer_toggle("sip_corridors")

# -------------------------------------------------
st.sidebar.header("Filters")

# Slider bounds come from the data; dropna() ignores blank years.
all_years = pd.concat([sip_intersections.end_year, sip_corridors.end_year]).dropna()
min_year, max_year = int(all_years.min()), int(all_years.max())

year_range = st.sidebar.slider(
    "End year", min_year, max_year, (min_year, max_year)
)
include_ongoing = st.sidebar.checkbox(
    "Include projects with no end date", value=True
)


# -------------------------------------------------
st.sidebar.subheader("Areas")

# One multiselect per area layer. Options come from the boundary file itself.
# Nothing selected = no filter on that area layer.
area_selection = {}
for key, cfg in AREAS.items():
    # load_areas now returns TWO things
    areas, display_col = load_areas(key)

    # extract unique list of area names
    options = areas[display_col].dropna().unique().tolist()

    # special sorting for CDs only
    if key == "cd":
        options = sorted(options, key=cd_sort_key)
    else:
        options = sorted(options)

    area_selection[key] = st.sidebar.multiselect(cfg["label"], options)




# EXPLAIN
def in_areas(series, selected):
    chosen = set(selected)
    return series.apply(lambda lst: any(item in chosen for item in lst))



def apply_filters(gdf):
    """Keep rows passing the year range AND every area selection."""
    keep = gdf.end_year.between(*year_range).fillna(False)
    if include_ongoing:
        keep = keep | gdf.end_year.isna()
    for key, selected in area_selection.items():
        if selected:
            keep = keep & in_areas(gdf[f"area_{key}"], selected)
    return gdf[keep]

# Filtered features
sip_intersections_f = apply_filters(sip_intersections)
sip_corridors_f = apply_filters(sip_corridors)


# --- Build layers -----------------------------------------------------------



# Drawn in order: area outlines first (bottom), then corridors, then points.
layers = []



def area_layer(key, selected):
    """Return a pydeck PolygonLayer for the given area type and selection."""
    if not selected:
        return None

    cfg = AREAS[key]
    areas, display_col = load_areas(key)

    shown = areas[areas[display_col].isin(selected)]
    poly_data = polygons_for_pydeck(shown, display_col)

    return pdk.Layer(
        "PolygonLayer",
        data=poly_data,
        get_polygon="polygon",
        stroked=True,
        filled=True,
        get_fill_color=[*cfg["rgb"], 50],
        get_line_color=[*cfg["rgb"], 255],
        get_line_width=cfg["width"],
        pickable=False,
    )


# Show selected areas
for key, selected in area_selection.items():
    layer = area_layer(key, selected)
    if layer:
        layers.append(layer)

if show_corridors:
    layers.append(pdk.Layer(
        "GeoJsonLayer",
        data=to_geojson(sip_corridors_f),
        get_line_color=rgba("sip_corridors", 150),
        get_line_width=10,
        line_width_min_pixels=2,
        pickable=True,
    ))

if show_intersections:
    layers.append(pdk.Layer(
        "GeoJsonLayer",
        data=to_geojson(sip_intersections_f),
        get_point_radius=100,
        point_radius_min_pixels=3,
        get_fill_color=rgba("sip_intersections", 150),
        pickable=True,
    ))


# --- Page -------------------------------------------------------------------
st.title("NYC Street Projects")


# --- Map --------------------------------------------------------------------
st.pydeck_chart(pdk.Deck(
    layers=layers,
    initial_view_state=MAP_VIEW,
    # GeoJsonLayer fields live under "properties", so the prefix is needed.
    # Replace project_name with a real column from your data.
    tooltip={"text": "{properties.project_name}"},
))


# --- Charts ------------------------------------------------------------------

# Projects by year
# One column per visible layer, so st.bar_chart stacks them by layer.
counts = {}
colors = []

# Store filtered layers into one list, loop through list each time counting all features

# If selected, get counts of projects
if show_intersections:
    counts["Intersections"] = sip_intersections_f.end_year.value_counts()
    colors.append(hex_color("sip_intersections"))
if show_corridors:
    counts["Corridors"] = sip_corridors_f.end_year.value_counts()
    colors.append(hex_color("sip_corridors"))

if counts:
    years = range(year_range[0], year_range[1] + 1)
    year_chart_df = (
        pd.DataFrame(counts)
        .reindex(years)   # include every year in range, even with no projects
        .fillna(0)
        .astype(int)
    )
    year_chart_df.index = year_chart_df.index.astype(str)  # avoids "2,023"-style labels
    st.subheader("Projects by end year")
    st.bar_chart(year_chart_df, color=colors)
else:
    st.info("Turn on a layer to see the chart.")



# Projects by area
selected_boros = area_selection["borough"]
selected_cds = area_selection["cd"]

# Step 1: decide which tag column to count and which categories to show
if selected_cds:
    area_key = "cd"
    categories = sorted(selected_cds, key=cd_sort_key)
    chart_title = "Projects by selected community district"
elif selected_boros:
    area_key = "cd"
    cds, cd_col = load_areas("cd")
    all_cds = cds[cd_col].dropna().unique().tolist()
    # a CD name looks like "Brooklyn 3", so everything before the number is the borough
    categories = sorted(
        [cd for cd in all_cds if cd.rsplit(" ", 1)[0] in selected_boros],
        key=cd_sort_key,
    )
    chart_title = "Projects by community district"
else:
    area_key = "borough"
    boros, boro_col = load_areas("borough")
    categories = sorted(boros[boro_col].dropna().unique().tolist())
    chart_title = "Projects by borough"

# Step 2: count each visible layer's filtered features per category
filtered_layers = [
    ("sip_intersections", show_intersections, sip_intersections_f),
    ("sip_corridors", show_corridors, sip_corridors_f),
]

area_counts = {}
area_colors = []
for layer_key, is_shown, gdf in filtered_layers:
    if is_shown:
        # explode: one row per (project, area) pair, then count each area
        area_counts[LAYERS[layer_key]["chart_name"]] = (
            gdf[f"area_{area_key}"].explode().value_counts()
        )
        area_colors.append(hex_color(layer_key))

# Step 3: draw
if area_counts:
    area_chart_df = (
        pd.DataFrame(area_counts)
        .reindex(categories)   # only the categories above, zeros where no projects
        .fillna(0)
        .astype(int)
    )
    st.subheader(chart_title)
    st.bar_chart(area_chart_df, color=area_colors, sort=False)
else:
    st.info("Turn on a layer to see the chart.")