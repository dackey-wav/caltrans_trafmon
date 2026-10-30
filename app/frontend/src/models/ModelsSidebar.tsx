import { ArrowLeft } from 'lucide-react';
import type { LocalModelInfo } from '../types';

export type ModelsSidebarProps = {
  models: LocalModelInfo[];
  selectedModel: LocalModelInfo | null;
  selectedRunId: string | null;
  loading: boolean;
  onBackToTraffic: () => void;
  onBackToModels: () => void;
  onSelectModel: (modelId: string) => void;
  onSelectRun: (runId: string) => void;
};

export function ModelsSidebar({
  models,
  selectedModel,
  selectedRunId,
  loading,
  onBackToTraffic,
  onBackToModels,
  onSelectModel,
  onSelectRun,
}: ModelsSidebarProps) {
  const runs = selectedModel?.runs ?? [];

  return (
    <div className="sidebar-pane sidebar-pane-full">
      {selectedModel ? (
        <>
          <button type="button" className="sidebar-back" onClick={onBackToModels}>
            <ArrowLeft size={14} />
            All models
          </button>
          <div className="sidebar-section-title">{selectedModel.label}</div>
          <div className="sidebar-subsection">
            {selectedModel.family === 'anomaly' ? 'Anomaly detection' : 'Forecasting'}
          </div>
          {loading && <p className="sidebar-map-hint">Loading runs…</p>}
          {!loading && runs.length === 0 && (
            <p className="sidebar-map-hint">
              No runs under{' '}
              <code style={{ fontSize: '0.72rem' }}>{selectedModel.artifacts_root}</code>.
            </p>
          )}
          <div className="models-run-list">
            {runs.map((run) => (
              <button
                key={run.run_id}
                type="button"
                className={`models-run-item ${selectedRunId === run.run_id ? 'active' : ''}`}
                onClick={() => onSelectRun(run.run_id)}
              >
                <span className="models-run-item-id">{run.run_id}</span>
                {selectedModel.family === 'anomaly' && run.anomaly_rate != null && (
                  <span className="models-run-item-rate">
                    {(run.anomaly_rate * 100).toFixed(1)}%
                  </span>
                )}
                {selectedModel.family === 'forecasting' && run.smape != null && (
                  <span className="models-run-item-rate">{run.smape.toFixed(1)}%</span>
                )}
              </button>
            ))}
          </div>
        </>
      ) : (
        <>
          <button type="button" className="sidebar-back" onClick={onBackToTraffic}>
            <ArrowLeft size={14} />
            Traffic views
          </button>
          <div className="sidebar-section-title">Models</div>
          <div className="sidebar-subsection">Local artifacts</div>
          {loading && <p className="sidebar-map-hint">Loading models…</p>}
          <div className="models-run-list">
            {models.map((model) => (
              <button
                key={model.id}
                type="button"
                className="models-run-item"
                onClick={() => onSelectModel(model.id)}
              >
                <span className="models-run-item-id">{model.label}</span>
                <span className="models-run-item-rate">
                  {model.run_count ?? model.runs.length}
                </span>
              </button>
            ))}
          </div>
          <p className="sidebar-map-hint">Pick a model to browse its runs.</p>
        </>
      )}
    </div>
  );
}
