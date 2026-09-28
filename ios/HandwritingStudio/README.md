# Handwriting Studio for iPhone and iPad

This is the native iOS client for the handwriting synthesis model.

## Native stack

- SwiftUI for the complete interface
- Core ML for local recurrent neural-network inference
- SwiftUI `Canvas` and Core Graphics paths for handwriting rendering
- A live progress bar driven by completed Core ML inference steps
- Named writing styles with previews drawn from the bundled style primers
- Selectable A5, A4, A3, Letter, Legal, phone, tablet, desktop HD, and square canvases
- Portrait/landscape orientation, adjustable writing size, and automatic word wrapping
- German umlauts, sharp s, and French accented-letter vector marks
- Personal handwriting primers captured locally with Apple Pencil on iPad
- UIKit's native share sheet for SVG export
- XCTest for client model and configuration tests

The app does not embed the React site, use a web view, or contact the Python service. The converted `HandwritingStep.mlpackage`, all 13 style primers, the recurrent sampling loop, GMM sampling, stroke cleanup, layout, and SVG export run locally on the device. The style selector shows a real sample for every bundled style, and generation reports actual model progress rather than displaying a timer-based animation. Text wraps at word boundaries according to the selected canvas, orientation, margins, and writing size; explicit line breaks are preserved.

## Run in the simulator

Open `HandwritingStudio.xcodeproj`, select an iPhone or iPad simulator, and run the `HandwritingStudio` scheme. No service or network connection is required.

## Run on an iPhone or iPad

Select your development team in Xcode and run the app on a device with iOS 17 or newer. On iPad, use **Add my handwriting** to write the prompted sample with Apple Pencil. The normalized sample is stored only in the app's Application Support directory and appears as a personal style in the selector. The model and style data are local, so generation and personal styles work offline.

## Regenerate the project

The checked-in Xcode project is generated from `project.yml`:

```bash
xcodegen generate
```
