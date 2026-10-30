CREATE DATABASE IF NOT EXISTS traffic;

CREATE TABLE IF NOT EXISTS traffic.traffic_5min
(
    ts                   DateTime64(0, 'America/Los_Angeles'),
    speed_mph            Nullable(Float32),
    flow_veh_5min        Nullable(Float32),
    occupancy_fraction   Nullable(Float32),
    observed_percent     Nullable(Float32),
    ingest_version       DateTime64(3, 'UTC')
)
ENGINE = ReplacingMergeTree(ingest_version)
PARTITION BY toYYYYMM(ts)
ORDER BY (ts)
SETTINGS index_granularity = 8192;
