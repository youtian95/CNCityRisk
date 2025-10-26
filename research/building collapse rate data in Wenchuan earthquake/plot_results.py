
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cnmaps import get_adm_maps
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
from matplotlib.contour import QuadContourSet
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from shapely.geometry import Polygon, MultiPolygon
from pyproj import Geod

from CNCityRisk import Utilities
from CNCityRisk.Utilities import DEMFetcher


SURVEY_DATA_PATH = Path(r'research\building collapse rate data in Wenchuan earthquake\data_20250904_160202_with_coords.csv')
MODEL_GRID_PATH = Path(r'CNCityRisk\results\Maps\CollapseRateContours\CollapseRate_Contour_Event_M7.9_Hypo104.11_31.65.csv')
CONTOUR_LEVELS = np.linspace(0.0, 1.0, 11)
COLORMAP = plt.get_cmap('Oranges')
VALUE_NORM = Normalize(0.0, 1.0)
ELEVATION_CMAP = plt.get_cmap('Oranges')
DENSITY_CMAP = plt.get_cmap('YlOrRd')
DENSITY_STEP_DEGREES = 0.05
BUILDING_LAT_MIN = 30.7678
BUILDING_LAT_MAX = 32.7578
BUILDING_LON_MIN = 102.8883
BUILDING_LON_MAX = 105.7345
GEOD_WGS84 = Geod(ellps='WGS84')


def get_collapse_rate_value(class_name: str) -> float:
    """Return the representative collapse-rate percentage for a survey classification."""

    rate_mapping = {
        '90-100%': 95,
        '80-90%': 85,
        '70-80%': 75,
        '60-70%': 65,
        '50-60%': 55,
        '40-50%': 45,
        '30-40%': 35,
        '20-30%': 25,
        '10-20%': 15,
        '0-10%': 5
    }

    return rate_mapping.get(class_name, 50)


def load_survey_data(path: Path) -> pd.DataFrame:
    """Load survey collapse-rate data containing lon/lat and categorical classes."""

    return pd.read_csv(path)


def load_model_grid(path: Path) -> pd.DataFrame | None:
    """Load modelled collapse-rate grid exported by CNCityRisk, if available."""

    if path.exists():
        return pd.read_csv(path)

    print(f"Warning: collapse-rate grid file not found: {path}")
    return None


def compute_plot_bounds(df: pd.DataFrame, buffer_ratio: float = 0) -> tuple[float, float, float, float]:
    """Return plotting bounds (xmin, xmax, ymin, ymax) with an outer buffer."""

    lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
    lat_min, lat_max = df['latitude'].min(), df['latitude'].max()

    lon_buffer = (lon_max - lon_min) * buffer_ratio
    lat_buffer = (lat_max - lat_min) * buffer_ratio

    return (
        float(lon_min - lon_buffer),
        float(lon_max + lon_buffer),
        float(lat_min - lat_buffer),
        float(lat_max + lat_buffer)
    )


def _iter_geometries(map_data):
    if map_data is None:
        return
    if hasattr(map_data, 'geometry') and map_data.geometry is not None:
        for geom in map_data.geometry:
            yield geom
        return
    if isinstance(map_data, (Polygon, MultiPolygon)):
        if isinstance(map_data, MultiPolygon):
            for geom in map_data.geoms:
                yield geom
        else:
            yield map_data
        return
    if isinstance(map_data, dict) and 'geometry' in map_data:
        yield from _iter_geometries(map_data['geometry'])
        return
    if isinstance(map_data, list):
        for item in map_data:
            yield from _iter_geometries(item)


def add_base_map(ax: plt.Axes) -> None:
    """Overlay Sichuan county boundaries for geographic context."""

    county_maps = get_adm_maps(province='四川省', level='区县', only_polygon=True)
    for geom in _iter_geometries(county_maps):
        ax.add_geometries(
            [geom],
            crs=ccrs.PlateCarree(),
            edgecolor='b',
            facecolor='none',
            linewidth=0.8,
            alpha=0.8
        )


def configure_axis(ax: plt.Axes, bounds: tuple[float, float, float, float]) -> None:
    """Set extent, ticks, and graticule for a Mercator axis."""

    xmin, xmax, ymin, ymax = bounds
    ax.set_extent([xmin, xmax, ymin, ymax], crs=ccrs.PlateCarree())

    lon_ticks = np.arange(np.ceil(xmin), np.floor(xmax) + 1, 1.0)
    lat_ticks = np.arange(np.ceil(ymin), np.floor(ymax) + 1, 1.0)
    if lon_ticks.size == 0:
        lon_ticks = np.array([np.round(xmin)])
    if lat_ticks.size == 0:
        lat_ticks = np.array([np.round(ymin)])

    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=10)

    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.4,
        linestyle='--',
        color='gray',
        alpha=0.4,
        xlocs=lon_ticks,
        ylocs=lat_ticks
    )


def plot_survey_map(
    ax: plt.Axes,
    survey_df: pd.DataFrame,
    contour_levels: np.ndarray,
    bounds: tuple[float, float, float, float]
) -> QuadContourSet | None:
    """Plot survey collapse data as scatter with an optional tricontour backdrop."""

    configure_axis(ax, bounds)
    add_base_map(ax)

    scatter_lons = survey_df['longitude'].to_numpy()
    scatter_lats = survey_df['latitude'].to_numpy()
    collapse_percent = survey_df['class_name'].map(get_collapse_rate_value).to_numpy()
    collapse_prob = collapse_percent / 100.0
    point_sizes = 20 + collapse_prob * 80

    contour: QuadContourSet | None = None
    if scatter_lons.size >= 3:
        contour = ax.tricontourf(
            scatter_lons,
            scatter_lats,
            collapse_prob,
            levels=contour_levels,
            cmap=COLORMAP,
            norm=VALUE_NORM,
            alpha=0.85,
            transform=ccrs.PlateCarree()
        )

    ax.scatter(
        scatter_lons,
        scatter_lats,
        c=collapse_prob,
        cmap=COLORMAP,
        norm=VALUE_NORM,
        s=point_sizes,
        alpha=0.85,
        edgecolors='black',
        linewidth=0.3,
        transform=ccrs.PlateCarree()
    )

    ax.set_title('Wenchuan Earthquake Collapse Rate (Survey)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude (°)', fontsize=12)
    ax.set_ylabel('Latitude (°)', fontsize=12)

    class_counts = survey_df['class_name'].value_counts()
    handles: list[Line2D] = []
    for class_name in sorted(class_counts.index, key=lambda name: get_collapse_rate_value(name), reverse=True):
        value = get_collapse_rate_value(class_name) / 100.0
        size = 20 + value * 80
        marker_size = np.sqrt(size)
        handles.append(
            Line2D(
                [],
                [],
                marker='o',
                linestyle='None',
                markerfacecolor=COLORMAP(VALUE_NORM(value)),
                markeredgecolor='black',
                markeredgewidth=0.3,
                markersize=marker_size,
                label=f'{class_name} ({class_counts[class_name]} points)'
            )
        )

    if handles:
        legend = ax.legend(
            handles=handles,
            loc='upper right',
            bbox_to_anchor=(1.0, 1.0),
            frameon=True,
            fancybox=True
        )
        legend.set_title('Collapse Rate Classes', prop={'weight': 'bold'})

    return contour


def plot_model_map(
    ax: plt.Axes,
    model_df: pd.DataFrame | None,
    contour_levels: np.ndarray,
    bounds: tuple[float, float, float, float]
) -> QuadContourSet | None:
    """Plot modelled collapse probability; annotate when data is unavailable."""

    configure_axis(ax, bounds)
    add_base_map(ax)
    ax.set_title('Wenchuan Earthquake Collapse Rate (Model)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude (°)', fontsize=12)
    ax.set_ylabel('Latitude (°)', fontsize=12)

    if model_df is None or model_df.empty:
        ax.text(
            0.5,
            0.5,
            'Model data unavailable',
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=14,
            color='gray'
        )
        return None

    return ax.tricontourf(
        model_df['longitude'],
        model_df['latitude'],
        model_df['mean_collapse_probability'],
        levels=contour_levels,
        cmap=COLORMAP,
        norm=VALUE_NORM,
        alpha=0.85,
        transform=ccrs.PlateCarree()
    )


def compute_point_elevations(
    bounds: tuple[float, float, float, float],
    step_deg: float = 0.05
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Sample elevation on a regular lon/lat grid covering the provided bounds."""

    xmin, xmax, ymin, ymax = bounds
    if not np.all(np.isfinite(bounds)):
        return None
    if xmax <= xmin or ymax <= ymin:
        return None
    if step_deg <= 0:
        raise ValueError('step_deg must be positive')

    lon_values = np.arange(xmin, xmax + step_deg * 0.5, step_deg, dtype=float)
    lat_values = np.arange(ymin, ymax + step_deg * 0.5, step_deg, dtype=float)

    if lon_values.size == 0:
        lon_values = np.array([xmin], dtype=float)
    if lat_values.size == 0:
        lat_values = np.array([ymin], dtype=float)

    if lon_values[-1] < xmax:
        lon_values = np.append(lon_values, xmax)
    if lat_values[-1] < ymax:
        lat_values = np.append(lat_values, ymax)

    lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
    coords = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
    if coords.size == 0:
        return None

    dem_fetcher = DEMFetcher()
    elevation_cache: dict[tuple[float, float], float] = {}
    elevation_values = np.empty(coords.shape[0], dtype=float)

    for idx, (lon, lat) in enumerate(coords):
        key = (round(lon, 6), round(lat, 6))
        if key not in elevation_cache:
            try:
                elevation_cache[key] = float(dem_fetcher.get_elevation(lat, lon))
            except Exception:
                elevation_cache[key] = np.nan
        elevation_values[idx] = elevation_cache[key]

    return lon_grid.ravel(), lat_grid.ravel(), elevation_values


def load_building_data() -> pd.DataFrame | None:
    """Load building footprint centers within predefined regional bounds."""

    try:
        _, (lon_min, lon_max, lat_min, lat_max) = Utilities.generate_region_name(
            lon_min=BUILDING_LON_MIN,
            lon_max=BUILDING_LON_MAX,
            lat_min=BUILDING_LAT_MIN,
            lat_max=BUILDING_LAT_MAX
        )
        bld_gdf = Utilities.get_RegionBldData(
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max
        )
    except Exception as exc:
        print(f"Warning: failed to load building data: {exc}")
        return None

    if bld_gdf is None or bld_gdf.empty:
        return None

    if 'longitude' not in bld_gdf.columns or 'latitude' not in bld_gdf.columns:
        geometry = getattr(bld_gdf, 'geometry', None)
        if geometry is None:
            return None
        centroids = geometry.centroid
        bld_gdf = bld_gdf.assign(
            longitude=centroids.x,
            latitude=centroids.y
        )

    coords_df = pd.DataFrame({
        'longitude': pd.to_numeric(bld_gdf['longitude'], errors='coerce'),
        'latitude': pd.to_numeric(bld_gdf['latitude'], errors='coerce')
    }).dropna()

    if coords_df.empty:
        return None

    return coords_df


def compute_building_density_grid(
    building_df: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    step_deg: float = DENSITY_STEP_DEGREES
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Aggregate building counts on a regular grid for density visualization."""

    if building_df is None or building_df.empty:
        return None
    if step_deg <= 0:
        raise ValueError('step_deg must be positive')

    xmin, xmax, ymin, ymax = bounds
    lon_edges = np.arange(xmin, xmax + step_deg, step_deg, dtype=float)
    lat_edges = np.arange(ymin, ymax + step_deg, step_deg, dtype=float)

    if lon_edges.size < 2:
        lon_edges = np.array([xmin, xmax], dtype=float)
    if lat_edges.size < 2:
        lat_edges = np.array([ymin, ymax], dtype=float)

    counts, lat_edges, lon_edges = np.histogram2d(
        building_df['latitude'].to_numpy(dtype=float),
        building_df['longitude'].to_numpy(dtype=float),
        bins=[lat_edges, lon_edges]
    )

    if not np.isfinite(counts).any() or float(np.nanmax(counts)) <= 0.0:
        return None

    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2.0
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2.0
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)

    density = np.zeros_like(counts, dtype=float)
    for i in range(lat_edges.size - 1):
        lat_bottom = lat_edges[i]
        lat_top = lat_edges[i + 1]
        for j in range(lon_edges.size - 1):
            lon_left = lon_edges[j]
            lon_right = lon_edges[j + 1]
            lon_poly = [lon_left, lon_right, lon_right, lon_left, lon_left]
            lat_poly = [lat_bottom, lat_bottom, lat_top, lat_top, lat_bottom]
            area_m2, _ = GEOD_WGS84.polygon_area_perimeter(lon_poly, lat_poly)
            area_km2 = abs(area_m2) / 1_000_000.0
            if area_km2 <= 0:
                density[i, j] = np.nan
            else:
                density[i, j] = counts[i, j] / area_km2

    if not np.isfinite(density).any() or float(np.nanmax(density)) <= 0.0:
        return None

    return lon_grid, lat_grid, density


def plot_building_density_map(
    ax: plt.Axes,
    building_df: pd.DataFrame | None,
    bounds: tuple[float, float, float, float],
    step_deg: float = DENSITY_STEP_DEGREES
) -> QuadContourSet | None:
    """Plot building density contour derived from footprint distribution."""

    configure_axis(ax, bounds)
    add_base_map(ax)
    ax.set_title('Wenchuan Region Building Density', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude (°)', fontsize=12)
    ax.set_ylabel('Latitude (°)', fontsize=12)

    if building_df is None or building_df.empty:
        ax.text(
            0.5,
            0.5,
            'No building data',
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=14,
            color='gray'
        )
        return None

    density_grid = compute_building_density_grid(building_df, bounds, step_deg=step_deg)
    if density_grid is None:
        ax.text(
            0.5,
            0.5,
            'Insufficient building density for contouring',
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=14,
            color='gray'
        )
        return None

    lon_grid, lat_grid, density = density_grid
    max_density = float(np.nanmax(density))
    if np.isclose(max_density, 0.0):
        return None

    base_levels = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 200.0, 300.0])
    if max_density > base_levels[-1]:
        levels = np.append(base_levels, max_density)
    else:
        levels = base_levels[base_levels <= max_density]
        if levels[-1] < max_density:
            levels = np.append(levels, max_density)
    if levels.size < 2:
        levels = np.array([0.0, max_density], dtype=float)

    color_positions = np.linspace(0.0, 1.0, len(levels) - 1)
    discrete_cmap = ListedColormap(DENSITY_CMAP(color_positions))
    norm = BoundaryNorm(levels, discrete_cmap.N)

    contour = ax.contourf(
        lon_grid,
        lat_grid,
        density,
        levels=levels,
        cmap=discrete_cmap,
        norm=norm,
        alpha=0.8,
        transform=ccrs.PlateCarree()
    )

    return contour


def plot_elevation_map(
    ax: plt.Axes,
    lon_values: np.ndarray,
    lat_values: np.ndarray,
    elevation_values: np.ndarray,
    bounds: tuple[float, float, float, float]
) -> tuple[QuadContourSet | None, float | None, float | None]:
    """Render elevation data as contours derived from model grid points."""

    configure_axis(ax, bounds)
    add_base_map(ax)

    valid_mask = np.isfinite(elevation_values)
    valid_elevations = elevation_values[valid_mask]
    raw_min = float(np.nanmin(valid_elevations))
    raw_max = float(np.nanmax(valid_elevations))
    min_elev = np.floor(raw_min / 1000.0) * 1000.0
    max_elev = np.ceil(raw_max / 1000.0) * 1000.0

    if np.isclose(min_elev, max_elev):
        min_elev -= 500.0
        max_elev += 500.0
        contour_levels = np.linspace(min_elev, max_elev, 101)
    else:
        contour_levels = np.linspace(min_elev, max_elev, 101)

    contour = ax.tricontourf(
        lon_values[valid_mask],
        lat_values[valid_mask],
        valid_elevations,
        levels=contour_levels,
        cmap=ELEVATION_CMAP,
        transform=ccrs.PlateCarree(),
    )

    ax.scatter(
        lon_values,
        lat_values,
        c=elevation_values,
        cmap=ELEVATION_CMAP,
        norm=None,
        s=10,
        alpha=0.4,
        transform=ccrs.PlateCarree()
    )

    ax.set_title('Wenchuan Region Elevation', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude (°)', fontsize=12)
    ax.set_ylabel('Latitude (°)', fontsize=12)

    return contour, float(min_elev), float(max_elev)


def save_survey_map(survey_df: pd.DataFrame, bounds: tuple[float, float, float, float], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.Mercator()})
    contour = plot_survey_map(ax, survey_df, CONTOUR_LEVELS, bounds)

    tick_values = np.linspace(0.0, 1.0, 11)
    mappable = contour if contour is not None else ScalarMappable(norm=VALUE_NORM, cmap=COLORMAP)
    if contour is None:
        mappable.set_array([])

    colorbar = fig.colorbar(
        mappable,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        shrink=0.7,
        ticks=tick_values
    )
    colorbar.set_label('Mean Collapse Probability')
    colorbar.set_ticklabels([
        '0' if np.isclose(value, 0.0) else '1' if np.isclose(value, 1.0) else f'{value:.1f}'
        for value in tick_values
    ])
    colorbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def save_model_map(model_df: pd.DataFrame | None, bounds: tuple[float, float, float, float], output_path: Path) -> bool:
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.Mercator()})
    contour = plot_model_map(ax, model_df, CONTOUR_LEVELS, bounds)

    if contour is not None:
        tick_values = np.linspace(0.0, 1.0, 11)
        colorbar = fig.colorbar(
            contour,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            shrink=0.7,
            ticks=tick_values
        )
        colorbar.set_label('Mean Collapse Probability')
        colorbar.set_ticklabels([
            '0' if np.isclose(value, 0.0) else '1' if np.isclose(value, 1.0) else f'{value:.1f}'
            for value in tick_values
        ])
        colorbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

    return contour is not None


def save_elevation_map(
    lon_values: np.ndarray,
    lat_values: np.ndarray,
    elevation_values: np.ndarray,
    bounds: tuple[float, float, float, float],
    output_path: Path
) -> bool:
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.Mercator()})
    contour, min_elev, max_elev = plot_elevation_map(ax, lon_values, lat_values, elevation_values, bounds)

    if contour is not None and min_elev is not None and max_elev is not None:
        tick_step = 1000.0
        tick_start = np.floor(min_elev / tick_step) * tick_step
        tick_end = np.ceil(max_elev / tick_step) * tick_step
        tick_values = np.arange(tick_start, tick_end + tick_step, tick_step)
        if tick_values.size == 0:
            tick_values = np.array([min_elev, max_elev])
        elif tick_values.size == 1:
            tick_values = np.array([min_elev, max_elev])

        colorbar = fig.colorbar(
            contour,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            shrink=0.7
        )
        colorbar.set_label('Elevation (m)')
        colorbar.set_ticks(tick_values)
        colorbar.set_ticklabels([str(int(round(value))) for value in tick_values])
        colorbar.ax.tick_params(labelsize=10)

    if contour is None:
        plt.close(fig)
        return False

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    return True


def save_building_density_map(
    building_df: pd.DataFrame | None,
    bounds: tuple[float, float, float, float],
    output_path: Path,
    step_deg: float = DENSITY_STEP_DEGREES
) -> bool:
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.Mercator()})
    contour = plot_building_density_map(ax, building_df, bounds, step_deg=step_deg)

    if contour is not None:
        levels = contour.levels
        tick_values = (levels[:-1] + levels[1:]) / 2.0
        tick_labels: list[str] = []
        for start, end in zip(levels[:-1], levels[1:]):
            if np.isclose(start, end):
                label = f"{start:.0f}"
            else:
                start_str = f"{start:.0f}" if np.isclose(start, round(start)) else f"{start:.1f}"
                end_str = f"{end:.0f}" if np.isclose(end, round(end)) else f"{end:.1f}"
                label = f"{start_str}-{end_str}"
            tick_labels.append(label)
        colorbar = fig.colorbar(
            contour,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            shrink=0.7
        )
        colorbar.set_label('Building Density (buildings/km²)', fontsize=11)
        colorbar.set_ticks(tick_values)
        colorbar.set_ticklabels(tick_labels)
        colorbar.ax.tick_params(labelsize=10)

    else:
        plt.close(fig)
        return False

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    return True


def generate_building_density_product(
    building_df: pd.DataFrame | None,
    bounds: tuple[float, float, float, float],
    output_path: Path
) -> tuple[bool, str | None]:
    try:
        generated = save_building_density_map(building_df, bounds, output_path)
        if generated:
            return True, None
        if building_df is None or building_df.empty:
            return False, "Missing building data"
        return False, "Building density insufficient to generate contour"
    except Exception as exc:
        print(f"Warning: failed to render building density map: {exc}")
        return False, f"Building density map generation failed: {exc}"


def generate_elevation_product(
    bounds: tuple[float, float, float, float],
    output_path: Path
) -> tuple[bool, str | None]:
    try:
        elevation_data = compute_point_elevations(bounds)
        if elevation_data is None:
            return False, "Invalid bounds; cannot sample elevation data"
        lon_values, lat_values, elevation_values = elevation_data
        generated = save_elevation_map(lon_values, lat_values, elevation_values, bounds, output_path)
        if generated:
            return True, None
        return False, "Insufficient elevation points to generate contour"
    except Exception as exc:
        print(f"Warning: failed to retrieve elevation data: {exc}")
        return False, f"Elevation sampling error: {exc}"

def main() -> None:
    survey_df = load_survey_data(SURVEY_DATA_PATH)
    model_df = load_model_grid(MODEL_GRID_PATH)
    bounds = compute_plot_bounds(survey_df)
    building_df = load_building_data()

    survey_output = Path('wenchuan_earthquake_collapse_rate_survey.pdf')
    model_output = Path('wenchuan_earthquake_collapse_rate_model.pdf')
    elevation_output = Path('wenchuan_earthquake_elevation_map.pdf')
    density_output = Path('wenchuan_earthquake_building_density.pdf')

    save_survey_map(survey_df, bounds, survey_output)
    model_has_data = save_model_map(model_df, bounds, model_output)
    density_generated, density_message = generate_building_density_product(building_df, bounds, density_output)
    elevation_generated, elevation_message = generate_elevation_product(bounds, elevation_output)

    print("\nFigures saved:")
    print(f"- {survey_output}")
    if model_has_data:
        print(f"- {model_output}")
    else:
        print(f"- {model_output} (model contour unavailable; contains placeholder message)")
    if density_generated:
        print(f"- {density_output}")
    else:
        if density_message:
            print(f"- Building density map not generated ({density_message})")
        else:
            print("- Building density map not generated")
    if elevation_generated:
        print(f"- {elevation_output}")
    else:
        if elevation_message:
            print(f"- Elevation map not generated ({elevation_message})")
        else:
            print("- Elevation map not generated")


if __name__ == '__main__':
    main()