-- London Office Monitor schema. Idempotent: every statement is IF NOT EXISTS.
-- {{BROKERS}}, {{METRICS}} and {{SUBMARKETS}} are filled from london_office_monitor.vocab by apply_schema(), so the
-- database refuses any value the code does not know.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS sources (
  id bigserial PRIMARY KEY,
  source_key text NOT NULL UNIQUE,
  publisher text NOT NULL,
  title text NOT NULL,
  url text NOT NULL,
  published date,
  licence_note text NOT NULL
);

CREATE TABLE IF NOT EXISTS broker_metrics (
  id bigserial PRIMARY KEY,
  broker text NOT NULL CHECK (broker IN ({{BROKERS}}) AND broker <> 'Knight Frank'),
  metric text NOT NULL CHECK (metric IN ({{METRICS}})),
  submarket text NOT NULL CHECK (submarket IN ({{SUBMARKETS}})),
  broker_label text NOT NULL,
  period text NOT NULL CHECK (period ~ '^[0-9]{4}(Q[1-4]|-(0[1-9]|1[0-2]))$'),
  value numeric NOT NULL,
  unit text NOT NULL,
  measure_note text NOT NULL CHECK (measure_note <> ''),
  source_id bigint NOT NULL REFERENCES sources (id),
  locator text,
  checked_at date NOT NULL,
  check_method text NOT NULL CHECK (check_method IN ('automatic', 'manual')),
  UNIQUE (broker, metric, submarket, period)
);

CREATE TABLE IF NOT EXISTS metric_definitions (
  id bigserial PRIMARY KEY,
  broker text NOT NULL CHECK (broker IN ({{BROKERS}})),
  metric text NOT NULL CHECK (metric IN ({{METRICS}})),
  definition text NOT NULL,
  source_id bigint NOT NULL REFERENCES sources (id),
  UNIQUE (broker, metric)
);

CREATE TABLE IF NOT EXISTS broker_area_map (
  id bigserial PRIMARY KEY,
  broker text NOT NULL CHECK (broker IN ({{BROKERS}})),
  broker_label text NOT NULL,
  submarket text NOT NULL CHECK (submarket IN ({{SUBMARKETS}})),
  fit text NOT NULL CHECK (fit IN ('exact', 'approximate')),
  note text NOT NULL DEFAULT '',
  UNIQUE (broker, broker_label)
);

CREATE TABLE IF NOT EXISTS macro_series (
  series text NOT NULL,
  day date NOT NULL,
  value numeric NOT NULL,
  PRIMARY KEY (series, day)
);

CREATE TABLE IF NOT EXISTS portfolio_assets (
  id bigserial PRIMARY KEY,
  name text NOT NULL UNIQUE,
  address text NOT NULL,
  submarket text NOT NULL CHECK (submarket IN ({{SUBMARKETS}})),
  size_sqft integer,
  status text NOT NULL,
  notes text NOT NULL DEFAULT '',
  source_keys text[] NOT NULL DEFAULT '{}',
  embedding vector(1024)
);

CREATE TABLE IF NOT EXISTS mandate (
  id bigserial PRIMARY KEY,
  label text NOT NULL CHECK (label = 'illustrative'),
  body jsonb NOT NULL
);

CREATE TABLE IF NOT EXISTS mandate_statements (
  id bigserial PRIMARY KEY,
  text text NOT NULL UNIQUE,
  embedding vector(1024)
);

-- News items, Gazette notices and internal documents. From external sources only what the feed publishes is kept:
-- title, summary, link and dates. The checks on source_kind and summary are set at the end of this file, so that
-- databases created earlier get them too.
CREATE TABLE IF NOT EXISTS documents (
  id bigserial PRIMARY KEY,
  source_kind text NOT NULL,
  publisher text NOT NULL,
  title text NOT NULL,
  subtitle text,
  summary text NOT NULL,
  url text NOT NULL,
  published_at timestamptz NOT NULL,
  fetched_at timestamptz NOT NULL,
  content_hash text NOT NULL UNIQUE,
  story_id bigint NOT NULL,
  synthetic boolean NOT NULL DEFAULT false,
  fixture_role text,
  meta jsonb NOT NULL DEFAULT '{}',
  embedding vector(1024),
  tsv tsvector GENERATED ALWAYS AS (
    to_tsvector('english', title || ' ' || coalesce(subtitle, '') || ' ' || summary)) STORED
);
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw ON documents USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS documents_tsv_gin ON documents USING gin (tsv);
CREATE INDEX IF NOT EXISTS documents_published_at ON documents (published_at);
CREATE INDEX IF NOT EXISTS documents_url ON documents (url);
CREATE INDEX IF NOT EXISTS documents_story_id ON documents (story_id);

CREATE TABLE IF NOT EXISTS chunks (
  id bigserial PRIMARY KEY,
  document_id bigint NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
  chunk_index integer NOT NULL,
  heading_path text[] NOT NULL DEFAULT '{}',
  text text NOT NULL,
  char_start integer NOT NULL,
  char_end integer NOT NULL,
  is_table boolean NOT NULL DEFAULT false,
  embedding vector(1024),
  tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
  UNIQUE (document_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);

CREATE TABLE IF NOT EXISTS ingest_runs (
  id bigserial PRIMARY KEY,
  source_name text NOT NULL,
  started_at timestamptz NOT NULL,
  finished_at timestamptz,
  fetched integer NOT NULL DEFAULT 0,
  new integer NOT NULL DEFAULT 0,
  error text
);
CREATE INDEX IF NOT EXISTS ingest_runs_source ON ingest_runs (source_name, started_at);

-- score_inputs is added at the end of this file, so that databases created earlier get it too.
CREATE TABLE IF NOT EXISTS opportunities (
  id bigserial PRIMARY KEY,
  thread_id text NOT NULL,
  approval_id text,
  status text NOT NULL CHECK (status IN ('draft', 'discarded', 'pending', 'approved', 'rejected')),
  card jsonb NOT NULL,
  score integer,
  score_components jsonb,
  review_notes text,
  decided_at timestamptz,
  decision_reason text,
  embedding vector(1024),
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS opportunities_thread ON opportunities (thread_id);

CREATE TABLE IF NOT EXISTS approvals (
  approval_id text PRIMARY KEY,
  thread_id text NOT NULL,
  tool_call_id text NOT NULL,
  card_ids bigint[] NOT NULL,
  status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'resolved')),
  created_at timestamptz NOT NULL DEFAULT now(),
  resolved_at timestamptz
);

CREATE TABLE IF NOT EXISTS model_calls (
  id bigserial PRIMARY KEY,
  thread_id text NOT NULL,
  turn_id text NOT NULL,
  mode text NOT NULL,
  role text NOT NULL,
  model text NOT NULL,
  provider text,
  input_tokens integer,
  output_tokens integer,
  cost_usd numeric,
  latency_ms integer NOT NULL,
  ok boolean NOT NULL,
  error text,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS model_calls_turn ON model_calls (turn_id);

CREATE TABLE IF NOT EXISTS tool_calls (
  id bigserial PRIMARY KEY,
  thread_id text NOT NULL,
  turn_id text NOT NULL,
  mode text NOT NULL,
  tool text NOT NULL,
  args jsonb NOT NULL DEFAULT '{}',
  outcome text,
  ms integer NOT NULL,
  ok boolean NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS tool_calls_turn ON tool_calls (turn_id);

-- Internal documents (25 Oct 2026): a third source kind, and room for whole documents rather than feed summaries.
-- Dropped and re-added so that every database, new or created before, ends with these exact checks.
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_source_kind_check;
ALTER TABLE documents ADD CONSTRAINT documents_source_kind_check
  CHECK (source_kind IN ('news', 'gazette', 'internal'));
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_summary_check;
ALTER TABLE documents ADD CONSTRAINT documents_summary_check
  CHECK (char_length(summary) <= CASE WHEN source_kind = 'internal' THEN 50000 ELSE 1000 END);

-- Score inputs (25 Oct 2026): what each card's score was computed from (the closest portfolio asset or mandate
-- statement, its similarity, and the age of the newest evidence), so every part of the score can be traced. Cards
-- scored earlier keep NULL.
ALTER TABLE opportunities ADD COLUMN IF NOT EXISTS score_inputs jsonb;
