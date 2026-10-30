import { useEffect, useRef } from 'react';
import {
  ArrowLeft,
  BarChart3,
  Brain,
  ChevronDown,
  ChevronRight,
  LineChart,
  Map,
} from 'lucide-react';
import type { ChartMode, ClassStationSelection, DatasetInfo, StationGroup } from '../types';

const MODE_LABELS: Record<ChartMode, string> = {
  count: 'Vehicles by count',
  classes: 'Vehicles by classes',
  map: 'Station map',
};

type IndeterminateCheckboxProps = {
  checked: boolean;
  indeterminate: boolean;
  onChange: (checked: boolean) => void;
};

function IndeterminateCheckbox({
  checked,
  indeterminate,
  onChange,
}: IndeterminateCheckboxProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (inputRef.current) inputRef.current.indeterminate = indeterminate;
  }, [indeterminate]);

  return (
    <input
      ref={inputRef}
      type="checkbox"
      checked={checked}
      onChange={(event) => onChange(event.target.checked)}
    />
  );
}

export type TrafficSidebarProps = {
  chartMode: ChartMode | null;
  panel: 'menu' | 'locations';
  datasets: DatasetInfo[];
  expandedDatasets: Record<string, boolean>;
  expandedBases: Record<string, boolean>;
  selectedLocations: Set<string>;
  classSelection: ClassStationSelection | null;
  onOpenMode: (mode: ChartMode) => void;
  onOpenModels: () => void;
  onBack: () => void;
  onToggleDataset: (datasetId: string) => void;
  onToggleBase: (baseId: string) => void;
  onSetBaseSelection: (
    datasetId: string,
    laneIds: string[],
    checked: boolean,
  ) => void;
  onToggleLane: (datasetId: string, locationId: string) => void;
  onSelectClassStation: (datasetId: string, station: StationGroup) => void;
  onSetClassLaneScope: (scope: 'all' | string) => void;
};

export function TrafficSidebar({
  chartMode,
  panel,
  datasets,
  expandedDatasets,
  expandedBases,
  selectedLocations,
  classSelection,
  onOpenMode,
  onOpenModels,
  onBack,
  onToggleDataset,
  onToggleBase,
  onSetBaseSelection,
  onToggleLane,
  onSelectClassStation,
  onSetClassLaneScope,
}: TrafficSidebarProps) {
  const renderCountStation = (dataset: DatasetInfo, station: StationGroup) => {
    const laneIds = station.lanes.map((lane) => lane.location_id);
    const allSelected = laneIds.every((id) => selectedLocations.has(id));
    const someSelected = laneIds.some((id) => selectedLocations.has(id));
    const multiLane = station.lanes.length > 1;

    return (
      <div key={station.base_id} className="station-group">
        <div className="station-row">
          <IndeterminateCheckbox
            checked={allSelected}
            indeterminate={someSelected && !allSelected}
            onChange={(checked) => onSetBaseSelection(dataset.id, laneIds, checked)}
          />
          <button
            type="button"
            className="station-label-btn"
            onClick={() => multiLane && onToggleBase(station.base_id)}
          >
            {multiLane &&
              (expandedBases[station.base_id] ? (
                <ChevronDown size={12} />
              ) : (
                <ChevronRight size={12} />
              ))}
            <span>{station.label}</span>
          </button>
          {multiLane && <span className="lane-count">{station.lanes.length} lanes</span>}
        </div>
        {multiLane && expandedBases[station.base_id] && (
          <div className="lane-list">
            {station.lanes.map((lane) => (
              <label
                key={lane.location_id}
                className={`lane-chip ${selectedLocations.has(lane.location_id) ? 'active' : ''}`}
              >
                <input
                  type="checkbox"
                  checked={selectedLocations.has(lane.location_id)}
                  onChange={() => onToggleLane(dataset.id, lane.location_id)}
                />
                {lane.lane || 'main'}
              </label>
            ))}
          </div>
        )}
      </div>
    );
  };

  const renderClassStation = (dataset: DatasetInfo, station: StationGroup) => {
    const selected =
      classSelection?.datasetId === dataset.id && classSelection.baseId === station.base_id;
    const multiLane = station.lanes.length > 1;

    return (
      <div key={station.base_id} className="station-group">
        <button
          type="button"
          className={`station-row station-selectable ${selected ? 'selected' : ''}`}
          onClick={() => onSelectClassStation(dataset.id, station)}
        >
          <span className="station-radio" aria-hidden />
          <span className="station-label-text">{station.label}</span>
          {multiLane && <span className="lane-count">{station.lanes.length} lanes</span>}
        </button>
        {selected && multiLane && classSelection && (
          <div className="lane-scope">
            <span className="lane-scope-label">Lane</span>
            <div className="lane-scope-options">
              <button
                type="button"
                className={`lane-scope-btn ${classSelection.laneScope === 'all' ? 'active' : ''}`}
                onClick={() => onSetClassLaneScope('all')}
              >
                All
              </button>
              {station.lanes.map((lane) => (
                <button
                  key={lane.location_id}
                  type="button"
                  className={`lane-scope-btn ${
                    classSelection.laneScope === lane.location_id ? 'active' : ''
                  }`}
                  onClick={() => onSetClassLaneScope(lane.location_id)}
                >
                  {lane.lane || 'main'}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className={`sidebar-track ${panel === 'locations' ? 'show-locations' : ''}`}>
      <div className="sidebar-pane">
        <div className="sidebar-section-title">View</div>
        <button
          type="button"
          className={`mode-card ${chartMode === 'count' ? 'active' : ''}`}
          onClick={() => onOpenMode('count')}
        >
          <LineChart size={20} className="mode-card-icon" />
          <span className="mode-card-title">Vehicles by count</span>
          <span className="mode-card-desc">Line chart · multiple locations</span>
        </button>
        <button
          type="button"
          className={`mode-card ${chartMode === 'classes' ? 'active' : ''}`}
          onClick={() => onOpenMode('classes')}
        >
          <BarChart3 size={20} className="mode-card-icon" />
          <span className="mode-card-title">Vehicles by classes</span>
          <span className="mode-card-desc">Stacked bars · one location</span>
        </button>
        <button
          type="button"
          className={`mode-card ${chartMode === 'map' ? 'active' : ''}`}
          onClick={() => onOpenMode('map')}
        >
          <Map size={20} className="mode-card-icon" />
          <span className="mode-card-title">Station map</span>
          <span className="mode-card-desc">Anomaly count per pin · click for details</span>
        </button>
        <div className="sidebar-section-title sidebar-section-spaced">Models</div>
        <button type="button" className="mode-card" onClick={onOpenModels}>
          <Brain size={20} className="mode-card-icon" />
          <span className="mode-card-title">Local runs</span>
          <span className="mode-card-desc">Browse model eval artifacts</span>
        </button>
        {chartMode === 'map' && (
          <p className="sidebar-map-hint">
            Pins and details use the main panel. Populate{' '}
            <code style={{ fontSize: '0.72rem' }}>dataset_a_map_anomalies.json</code> for counts;
            geometry is static under{' '}
            <code style={{ fontSize: '0.72rem' }}>dataset_a_stations_geo.json</code>.
          </p>
        )}
      </div>
      <div className="sidebar-pane">
        {chartMode && chartMode !== 'map' && (
          <>
            <button type="button" className="sidebar-back" onClick={onBack}>
              <ArrowLeft size={14} />
              Back
            </button>
            <div className="sidebar-section-title">{MODE_LABELS[chartMode]}</div>
            {datasets.map((dataset) => (
              <div key={dataset.id} className="region-block">
                <button
                  type="button"
                  className="region-header"
                  onClick={() => onToggleDataset(dataset.id)}
                >
                  {expandedDatasets[dataset.id] ? (
                    <ChevronDown size={14} />
                  ) : (
                    <ChevronRight size={14} />
                  )}
                  <span className="region-name">{dataset.label}</span>
                  <span className="region-badge">{dataset.station_count}</span>
                </button>
                {expandedDatasets[dataset.id] && (
                  <div className="station-tree">
                    {dataset.stations.map((station) =>
                      chartMode === 'count'
                        ? renderCountStation(dataset, station)
                        : renderClassStation(dataset, station),
                    )}
                  </div>
                )}
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}