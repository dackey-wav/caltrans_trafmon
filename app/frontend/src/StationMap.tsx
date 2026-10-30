import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import maplibregl, { type Map } from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import type { MapStation } from './types';

/** Rough Poland bounds — pan/zoom limited to this box (lng/lat). */
const POLAND_BOUNDS: maplibregl.LngLatBoundsLike = [
  [14.05, 49.0],
  [24.25, 54.95],
];

const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';

type Props = {
  stations: MapStation[];
  selectedStationId: string | null;
  onSelectStation: (stationId: string | null) => void;
};

function pinHtml(count: number, selected: boolean): string {
  const badge =
    count > 0
      ? `<span class="map-pin-badge">${count > 99 ? '99+' : count}</span>`
      : `<span class="map-pin-badge map-pin-badge-zero">0</span>`;
  return `<div class="map-pin ${selected ? 'map-pin-selected' : ''}"><span class="map-pin-dot"></span>${badge}</div>`;
}

function filterWithCoords(stations: MapStation[]): (MapStation & { lat: number; lon: number })[] {
  return stations.filter(
    (s): s is MapStation & { lat: number; lon: number } =>
      s.lat != null && s.lon != null && Number.isFinite(s.lat) && Number.isFinite(s.lon),
  );
}

function stationBounds(valid: (MapStation & { lat: number; lon: number })[]) {
  const lngs = valid.map((s) => s.lon);
  const lats = valid.map((s) => s.lat);
  return new maplibregl.LngLatBounds(
    [Math.min(...lngs), Math.min(...lats)],
    [Math.max(...lngs), Math.max(...lats)],
  );
}

/** Dark Matter is very dark; lift land + roads so the map is visible on our UI. */
function tuneMapStyle(map: Map) {
  try {
    if (map.getLayer('background')) {
      map.setPaintProperty('background', 'background-color', '#1c2129');
    }
    if (map.getLayer('water')) {
      map.setPaintProperty('water', 'fill-color', '#151a22');
    }
  } catch {
    /* ignore */
  }

  const roadPaint: [RegExp, string][] = [
    [/road_mot_fill/, '#8fa8c8'],
    [/road_trunk_fill/, '#7d96b5'],
    [/road_pri_fill/, '#6a84a3'],
    [/road_sec_fill/, '#556b85'],
    [/road_minor_fill/, '#3d4d60'],
  ];

  for (const layer of map.getStyle()?.layers ?? []) {
    if (layer.type !== 'fill' || !layer.id.startsWith('road_')) continue;
    const rule = roadPaint.find(([re]) => re.test(layer.id));
    if (!rule) continue;
    try {
      map.setPaintProperty(layer.id, 'fill-color', rule[1]);
      map.setPaintProperty(layer.id, 'fill-opacity', 0.95);
    } catch {
      /* expression-based layers */
    }
  }
}

export function StationMap({ stations, selectedStationId, onSelectStation }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<Map | null>(null);
  const markersRef = useRef<maplibregl.Marker[]>([]);
  const onSelectRef = useRef(onSelectStation);
  const selectedStationRef = useRef(selectedStationId);
  const [mapError, setMapError] = useState<string | null>(null);

  useEffect(() => {
    onSelectRef.current = onSelectStation;
    selectedStationRef.current = selectedStationId;
  }, [onSelectStation, selectedStationId]);

  const valid = useMemo(() => filterWithCoords(stations), [stations]);

  const dataKey = useMemo(
    () => valid.map((s) => `${s.station_id}:${s.lat}:${s.lon}:${s.anomaly_count}`).sort().join('|'),
    [valid],
  );

  const zoomIn = useCallback(() => {
    mapRef.current?.zoomIn({ duration: 200 });
  }, []);

  const zoomOut = useCallback(() => {
    mapRef.current?.zoomOut({ duration: 200 });
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (!el || valid.length === 0) {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
      return;
    }

    let cancelled = false;
    let map: Map | null = null;
    let ro: ResizeObserver | null = null;

    const mountMap = () => {
      if (cancelled || mapRef.current) return;
      if (el.offsetWidth < 2 || el.offsetHeight < 2) {
        requestAnimationFrame(mountMap);
        return;
      }

      setMapError(null);

      map = new maplibregl.Map({
        container: el,
        style: MAP_STYLE,
        attributionControl: false,
        maxBounds: POLAND_BOUNDS,
        minZoom: 7,
        maxZoom: 14,
        center: [19.0, 50.25],
        zoom: 8,
      });

      map.on('error', (e) => {
        const msg = e.error?.message ?? 'Map tile error';
        setMapError(msg);
        console.error('[StationMap]', e);
      });

      map.on('load', () => {
        if (!map || cancelled) return;
        tuneMapStyle(map);
        map.fitBounds(stationBounds(valid), { padding: 56, maxZoom: 11, duration: 0 });
        map.resize();
        syncMarkers(map);
      });

      map.on('click', () => {
        onSelectRef.current(null);
      });

      mapRef.current = map;

      ro = new ResizeObserver(() => {
        mapRef.current?.resize();
      });
      ro.observe(el);
    };

    const syncMarkers = (target: Map) => {
      markersRef.current.forEach((m) => m.remove());
      markersRef.current = [];

      for (const s of valid) {
        const selected = s.station_id === selectedStationRef.current;
        const pin = document.createElement('div');
        pin.className = 'map-pin-wrap';
        pin.innerHTML = pinHtml(s.anomaly_count, selected);
        pin.addEventListener('click', (e) => {
          e.stopPropagation();
          onSelectRef.current(
            s.station_id === selectedStationRef.current ? null : s.station_id,
          );
        });

        const marker = new maplibregl.Marker({ element: pin, anchor: 'bottom' })
          .setLngLat([s.lon, s.lat])
          .addTo(target);
        markersRef.current.push(marker);
      }
    };

    if (mapRef.current) {
      mapRef.current.remove();
      mapRef.current = null;
    }
    mountMap();

    return () => {
      cancelled = true;
      ro?.disconnect();
      markersRef.current.forEach((m) => m.remove());
      markersRef.current = [];
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, [dataKey, valid]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded() || valid.length === 0) return;

    markersRef.current.forEach((m) => m.remove());
    markersRef.current = [];

    for (const s of valid) {
      const selected = s.station_id === selectedStationId;
      const pin = document.createElement('div');
      pin.className = 'map-pin-wrap';
      pin.innerHTML = pinHtml(s.anomaly_count, selected);
      pin.addEventListener('click', (e) => {
        e.stopPropagation();
        onSelectStation(s.station_id === selectedStationId ? null : s.station_id);
      });

      const marker = new maplibregl.Marker({ element: pin, anchor: 'bottom' })
        .setLngLat([s.lon, s.lat])
        .addTo(map);
      markersRef.current.push(marker);
    }
  }, [dataKey, selectedStationId, valid, onSelectStation]);

  if (valid.length === 0) {
    return (
      <div className="map-empty">
        No coordinates for this region. Add static metadata under <code>app/core/data/</code>.
      </div>
    );
  }

  return (
    <div className="map-host-wrap">
      <div ref={containerRef} className="map-host" role="presentation" />
      {mapError && <div className="map-error-banner">{mapError}</div>}
      <div className="map-zoom-stack">
        <button type="button" className="map-zoom-btn" onClick={zoomIn} aria-label="Zoom in">
          +
        </button>
        <button type="button" className="map-zoom-btn" onClick={zoomOut} aria-label="Zoom out">
          −
        </button>
      </div>
    </div>
  );
}
