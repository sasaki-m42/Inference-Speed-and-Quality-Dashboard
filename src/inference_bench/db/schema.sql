PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS samples (
  id INTEGER PRIMARY KEY,
  flight_id TEXT NOT NULL,
  ts INTEGER NOT NULL,
  source TEXT NOT NULL,
  split TEXT NOT NULL,
  n_features INTEGER NOT NULL,
  features BLOB NOT NULL,
  label INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_samples_split ON samples(split);
CREATE INDEX IF NOT EXISTS idx_samples_flight_ts ON samples(flight_id, ts);

CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY,
  sample_id INTEGER NOT NULL,
  model_name TEXT NOT NULL,
  backend TEXT NOT NULL,
  score REAL NOT NULL,
  pred INTEGER NOT NULL,
  created_at INTEGER NOT NULL,
  FOREIGN KEY(sample_id) REFERENCES samples(id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_sample_backend
ON predictions(sample_id, backend);

CREATE TABLE IF NOT EXISTS benchmark (
  id INTEGER PRIMARY KEY,
  model_name TEXT NOT NULL,
  backend TEXT NOT NULL,
  n_samples INTEGER NOT NULL,
  batch_size INTEGER NOT NULL,
  num_threads INTEGER NOT NULL,
  warmup_iters INTEGER NOT NULL,
  measure_iters INTEGER NOT NULL,
  p50_ms REAL NOT NULL,
  p95_ms REAL NOT NULL,
  throughput_sps REAL NOT NULL,
  created_at INTEGER NOT NULL,
  notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_benchmark_backend_created
ON benchmark(backend, created_at);
