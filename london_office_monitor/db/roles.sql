-- Least-privilege grants. {{DB}} is the database name, quoted by apply_roles().
-- lom_mcp (the MCP server) reads the market data and may only add documents; it cannot touch opportunities,
-- approvals, telemetry or conversation checkpoints. lom_app (the agent) reads everything it serves and writes only
-- opportunities, approvals and telemetry. Checkpoint grants are added by setup_checkpointer().

GRANT CONNECT ON DATABASE {{DB}} TO lom_app, lom_mcp;
GRANT USAGE ON SCHEMA public TO lom_app, lom_mcp;

GRANT SELECT ON sources, broker_metrics, metric_definitions, broker_area_map, macro_series, portfolio_assets, mandate,
  mandate_statements, documents, chunks, ingest_runs, opportunities TO lom_mcp;
GRANT INSERT ON documents, chunks, sources, ingest_runs, macro_series TO lom_mcp;
GRANT USAGE, SELECT ON SEQUENCE documents_id_seq, chunks_id_seq, sources_id_seq, ingest_runs_id_seq TO lom_mcp;

GRANT SELECT ON sources, broker_metrics, metric_definitions, broker_area_map, macro_series, portfolio_assets, mandate,
  mandate_statements, documents, chunks, ingest_runs, opportunities, approvals, model_calls, tool_calls TO lom_app;
GRANT INSERT, UPDATE ON opportunities, approvals, model_calls, tool_calls TO lom_app;
GRANT USAGE, SELECT ON SEQUENCE opportunities_id_seq, model_calls_id_seq, tool_calls_id_seq TO lom_app;
