# Build the native spark-serve macOS app with CommandLineTools only (no Xcode).
APP     := gui/SparkServeApp.app
BIN     := SparkServeApp
TARGET  := arm64-apple-macosx13.0
GUI_FIXTURE ?= gui/tests/fixtures.json

all:
	@set -e; cd "$(dir $(lastword $(MAKEFILE_LIST)))"/.. ; \
	rm -rf "$(APP)" ; \
	mkdir -p "$(APP)/Contents/MacOS" "$(APP)/Contents/Resources" ; \
	cp gui/Info.plist "$(APP)/Contents/Info.plist" ; \
	cp gui/Resources/AppIcon.icns "$(APP)/Contents/Resources/AppIcon.icns" ; \
	echo "==> compiling (swiftc, $(TARGET))" ; \
	xcrun -sdk macosx swiftc -O \
	  -target "$(TARGET)" \
	  -framework SwiftUI -framework AppKit -framework Charts \
	  gui/*.swift \
	  -o "$(APP)/Contents/MacOS/$(BIN)" ; \
	codesign --force --deep --sign - "$(APP)" >/dev/null 2>&1 || echo "   (codesign skipped)" ; \
	echo "==> done: $(APP)" ; \
	echo "    run:  open $(APP)"

clean:
	rm -rf gui/SparkServeApp.app

smoke:
	@set -e; cd "$(dir $(lastword $(MAKEFILE_LIST)))"/.. ; \
	test -n "$(GUI_FIXTURE)" || { echo "Set GUI_FIXTURE to backend-generated fixtures.json"; exit 1; } ; \
	smoke_dir=$$(mktemp -d /tmp/spark-serve-gui-smoke.XXXXXX) ; \
	trap 'rm -rf "$$smoke_dir"' EXIT ; \
	xcrun -sdk macosx swiftc -O -target "$(TARGET)" \
	  -framework SwiftUI -framework AppKit -framework Charts \
	  $(filter-out gui/main.swift,$(wildcard gui/*.swift)) gui/tests/main.swift \
	  -o "$$smoke_dir/gui-protocol-smoke" ; \
	"$$smoke_dir/gui-protocol-smoke" "$(GUI_FIXTURE)"

.PHONY: all clean smoke
