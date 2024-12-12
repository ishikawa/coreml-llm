.PHONY: build

build: models/GPT2Model.mlpackage
	xcrun coremlcompiler compile models/GPT2Model.mlpackage ./SwiftCoreMLRunner/Sources/
	xcrun coremlcompiler generate models/GPT2Model.mlpackage ./SwiftCoreMLRunner/Sources/ --language Swift
	
