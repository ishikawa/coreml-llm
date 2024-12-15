.PHONY: build

build: models/GPT2Model.mlpackage
	xcrun coremlcompiler compile models/gpt2-baseline.mlpackage ./CoreMLRunner/Sources/CoreMLRunner/
	swift build --package-path ./CoreMLRunner
#	xcrun coremlcompiler generate models/GPT2Model.mlpackage ./MyCoreMLRunner/Sources/ \
#		--language Swift \
#		--swift-version 6.0
# coremldata.bin is also output to the analytics folder, so delete it.
#	rm -rf ./MyCoreMLRunner/Sources/GPT2Model.mlmodelc/analytics/	
