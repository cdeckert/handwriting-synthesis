# Handwriting Studio for iPhone

This is the native iOS client for the handwriting synthesis model.

## Native stack

- SwiftUI for the complete interface
- Core ML for local recurrent neural-network inference
- SwiftUI `Canvas` and Core Graphics paths for handwriting rendering
- A live progress bar driven by completed Core ML inference steps
- Named writing styles with previews drawn from the bundled style primers
- UIKit's native share sheet for SVG export
- XCTest for client model and configuration tests

The app does not embed the React site, use a web view, or contact the Python service. The converted `HandwritingStep.mlpackage`, all 13 style primers, the recurrent sampling loop, GMM sampling, stroke cleanup, layout, and SVG export run locally on the device. The style selector shows a real sample for every bundled style, and generation reports actual model progress rather than displaying a timer-based animation.

## Run in the simulator

Open `HandwritingStudio.xcodeproj`, select an iPhone simulator, and run the `HandwritingStudio` scheme. No service or network connection is required.

## Run on an iPhone

Select your development team in Xcode and run the app on an iPhone with iOS 17 or newer. The model and style data are part of the application bundle, so generation also works offline.

## Regenerate the project

The checked-in Xcode project is generated from `project.yml`:

```bash
xcodegen generate
```
