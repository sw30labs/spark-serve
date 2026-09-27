# Native protocol smoke

Run `make -f gui/Makefile smoke` on macOS with CommandLineTools. This compiles
the production Swift models and stores, then checks the Python-to-Swift contract
without launching the app or contacting a Spark.

`fixtures.json` contains synthetic status/telemetry and benchmark records produced
by the Python backend with injected transport. It has no live host data. The
test checks ISO timestamps, optional metrics, started/result events, retained
history bounds, and stale state after disconnect. Use `GUI_FIXTURE=/path/to/fixtures.json`
to check newly generated backend fixtures after a protocol change.
