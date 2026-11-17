import { useEffect, useRef, useState } from 'react';
import maplibregl, { type Map } from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';

/** Dark Matter is very dark; lift land and roads so the freeway stands out. */
function tuneMapStyle(map: Map) {
  const paints: [RegExp, string][] = [
    [/road_mot_fill/, '#8fa8c8'],
    [/road_trunk_fill/, '#7d96b5'],
    [/road_pri_fill/, '#6a84a3'],
    [/road_sec_fill/, '#556b85'],
    [/road_minor_fill/, '#3d4d60'],
  ];
  try {
    if (map.getLayer('background')) map.setPaintProperty('background', 'background-color', '#1c2129');
    if (map.getLayer('water')) map.setPaintProperty('water', 'fill-color', '#151a22');
  } catch {
    /* style variant without these layers */
  }
  for (const layer of map.getStyle()?.layers ?? []) {
    if (layer.type !== 'fill' || !layer.id.startsWith('road_')) continue;
    const rule = paints.find(([re]) => re.test(layer.id));
    if (!rule) continue;
    try {
      map.setPaintProperty(layer.id, 'fill-color', rule[1]);
    } catch {
      /* expression-based layer */
    }
  }
}

export function StationMap({ lat, lon, label }: { lat: number; lon: number; label: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    const map = new maplibregl.Map({
      container: ref.current,
      style: MAP_STYLE,
      center: [lon, lat],
      zoom: 13.5,
      attributionControl: { compact: true },
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');
    map.on('load', () => tuneMapStyle(map));
    map.on('error', (e) => setError(e.error?.message ?? 'map error'));
    const el = document.createElement('div');
    el.className = 'map-pin';
    el.innerHTML = '<span class="map-pin-dot"></span>';
    new maplibregl.Marker({ element: el })
      .setLngLat([lon, lat])
      .setPopup(new maplibregl.Popup({ offset: 12, closeButton: false }).setText(label))
      .addTo(map);
    return () => map.remove();
  }, [lat, lon, label]);

  return (
    <div className="map-box">
      <div ref={ref} className="map-host" />
      {error && <div className="map-error">Map tiles unavailable: {error}</div>}
    </div>
  );
}
