# Live smoke checks

The tests under `tests/` are hermetic: they do not contact the network or real LLM harnesses. Scripts named `smoke_*.py` in this directory are manual checks for installed CLIs, authenticated subscriptions, and live sources. They are not part of the CI gate.

Exit code 77 means skip because the environment is unavailable, such as a missing binary, login, or network. Any other nonzero code is a failure.

A live-check report must record the step, expected result, observed result, and verdict, followed by an explicit list of what was not exercised and why. Before trusting a green probe, point it at a known-dead target. The probe must fail or explicitly report a skip; a green result against a dead target is a defect in the probe.
