import CoreML
import MyCoreMLModels

if #available(macOS 13.0, *) {
    llm_predict()
} else {
    // Fail
    print("This version of macOS does not support CoreML")
}