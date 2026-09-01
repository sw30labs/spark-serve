# Build the native spark-serve macOS app with CommandLineTools only (no Xcode).
APP     := gui/SparkServeApp.app
BIN     := SparkServeApp
TARGET  := arm64-apple-macosx13.0

all:
	@cd "$(dir $(lastword $(MAKEFILE_LIST)))"/.. ; \
	rm -rf "$(APP)" ; \
	mkdir -p "$(APP)/Contents/MacOS" "$(APP)/Contents/Resources" ; \
	cp gui/Info.plist "$(APP)/Contents/Info.plist" ; \
	echo "==> compiling (swiftc, $(TARGET))" ; \
	xcrun -sdk macosx swiftc -O \
	  -target "$(TARGET)" \
	  -framework SwiftUI -framework AppKit \
	  gui/SparkServeApp.swift \
	  -o "$(APP)/Contents/MacOS/$(BIN)" ; \
	codesign --force --deep --sign - "$(APP)" >/dev/null 2>&1 || echo "   (codesign skipped)" ; \
	echo "==> done: $(APP)" ; \
	echo "    run:  open $(APP)"

clean:
	rm -rf gui/SparkServeApp.app

.PHONY: all clean
