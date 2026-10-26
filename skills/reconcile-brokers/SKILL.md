---
name: reconcile-brokers
description: Explain why brokers report different figures for the same metric, and give a house-view range with reasons.
allowed-tools: [compare_brokers, get_metric, get_citation, read_skill_file]
---
# Reconcile brokers

Use this when the brokers' figures differ, or when someone asks whether the brokers agree.

1. Gather every broker's figure. For a named period, call compare_brokers for that period. Otherwise call get_metric and take each broker's newest figure, even when their latest periods differ: never drop a broker because its newest figure is from an earlier quarter (compare_brokers for the latest quarter alone would). The period column says which is which.
2. Read how each broker measures: read_skill_file("reconcile-brokers", "definitions.md"), plus the definition and measure_note on each row.
3. Explain the spread only from what the data shows:
   - **Definitions**: headline prime rent against average achieved rent; how far ahead availability counts space; the deal-size threshold for take-up.
   - **Area boundaries**: a row whose area_fit is "approximate" covers a wider or narrower area than the submarket.
   - **Periods**: a monthly reading set against quarterly figures (see period_note).
4. Show a table: broker | figure | period | what is measured | citation.
5. Finish with a line that starts "House view:" giving the range from the minimum to the maximum figure as the tool returned them, and one or two sentences on which figure suits which purpose (for example headline prime for asking rents on the best space, average achieved for underwriting). Never state an average of the brokers.

Cite every figure with its [F...] citation from the tool result.
