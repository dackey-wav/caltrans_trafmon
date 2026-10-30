import type { ComponentType } from 'react';
import { Boxes, Waves } from 'lucide-react';
import type { LocalModelInfo } from './types';

type IconType = ComponentType<{ size?: number; className?: string }>;

const MODEL_ICONS: Record<string, IconType> = {
  timesnet: Waves,
};

type Props = {
  models: LocalModelInfo[];
  loading: boolean;
  onSelect: (modelId: string) => void;
};

export function ModelPicker({ models, loading, onSelect }: Props) {
  if (loading && models.length === 0) {
    return <div className="empty-state">Loading models…</div>;
  }
  if (models.length === 0) {
    return <div className="empty-state">No models registered.</div>;
  }

  return (
    <div className="model-picker">
      <div className="model-picker-head">
        <h2 className="model-picker-title">Local model runs</h2>
        <p className="model-picker-sub">
          Pick a model to browse its evaluation artifacts. Anomaly models flag unusual
          points; forecasting models compare actual vs predicted traffic.
        </p>
      </div>
      <div className="model-tiles">
        {models.map((model) => {
          const Icon = MODEL_ICONS[model.id] ?? Boxes;
          const count = model.run_count ?? model.runs.length;
          const disabled = count === 0;
          return (
            <button
              key={model.id}
              type="button"
              className={`model-tile model-tile--${model.family}${disabled ? ' model-tile--empty' : ''}`}
              onClick={() => onSelect(model.id)}
            >
              <div className="model-tile-top">
                <span className="model-tile-icon">
                  <Icon size={22} />
                </span>
                <span className={`model-tile-badge model-tile-badge--${model.family}`}>
                  {model.family === 'anomaly' ? 'anomaly' : 'forecast'}
                </span>
              </div>
              <span className="model-tile-name">{model.label}</span>
              {model.tagline && <span className="model-tile-tagline">{model.tagline}</span>}
              {model.focus && <span className="model-tile-focus">{model.focus}</span>}
              <span className="model-tile-runs">
                {disabled ? 'no runs yet' : `${count} run${count === 1 ? '' : 's'}`}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
