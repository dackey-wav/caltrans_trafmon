import { useEffect, useState } from 'react';
import { Activity, Cpu, MapPin } from 'lucide-react';
import { fetchConfig, type AppConfig } from './api';
import { LivePage } from './live/LivePage';
import { ModelPage } from './model/ModelPage';
import { defaultMqttUrl, useLiveSystem } from './live/useLiveSystem';
import './index.css';

type Page = 'live' | 'model';

function App() {
  const [page, setPage] = useState<Page>('live');
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);

  useEffect(() => {
    fetchConfig().then(setConfig).catch((e: Error) => setConfigError(e.message));
  }, []);

  const mqttUrl = config ? (config.mqtt_ws_url ?? defaultMqttUrl()) : null;
  // One MQTT connection for the whole app, so switching pages keeps the stream.
  const { state, send } = useLiveSystem(mqttUrl, config?.station_id ?? null);

  return (
    <div className="dashboard-container">
      <aside className="sidebar">
        <div className="sidebar-header">
          <Activity size={18} color="var(--accent)" />
          <span>Caltrans Trafmon</span>
        </div>
        <nav className="nav">
          <button className={page === 'live' ? 'active' : ''} onClick={() => setPage('live')}>
            <Cpu size={16} />
            <span>
              <b>Live IoT system</b>
              <small>Sensor → MQTT → predictor</small>
            </span>
          </button>
          <button className={page === 'model' ? 'active' : ''} onClick={() => setPage('model')}>
            <MapPin size={16} />
            <span>
              <b>Station &amp; model</b>
              <small>Data quality, offline test</small>
            </span>
          </button>
        </nav>
        <div className="sidebar-foot">
          <div>
            <span className={`status-dot status-${state.connection === 'online' ? 'online' : 'offline'}`} />
            MQTT {state.connection}
          </div>
          <code>{mqttUrl ?? '…'}</code>
          {config && <small>station {config.station_id}</small>}
        </div>
      </aside>
      <main className="main">
        {configError && <div className="error-banner">API unreachable: {configError}</div>}
        {page === 'live' ? <LivePage state={state} send={send} /> : <ModelPage />}
      </main>
    </div>
  );
}

export default App;
