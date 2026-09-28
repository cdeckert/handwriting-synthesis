//
// HandwritingStep.swift
//
// This file was automatically generated and should not be edited.
//

import CoreML


/// Model Prediction Input Type
@available(macOS 14.0, iOS 17.0, tvOS 17.0, watchOS 10.0, visionOS 1.0, *)
class HandwritingStepInput : MLFeatureProvider {

    /// Previous sampled x/y offset and pen state. as 1 by 3 matrix of floats
    var stroke: MLMultiArray

    /// Encoded text padded to 120 characters. as 1 by 120 matrix of 32-bit integers
    var chars: MLMultiArray

    /// Encoded text length including terminator. as 1 element vector of 32-bit integers
    var chars_len: MLMultiArray

    /// h1 as 1 by 400 matrix of floats
    var h1: MLMultiArray

    /// c1 as 1 by 400 matrix of floats
    var c1: MLMultiArray

    /// h2 as 1 by 400 matrix of floats
    var h2: MLMultiArray

    /// c2 as 1 by 400 matrix of floats
    var c2: MLMultiArray

    /// h3 as 1 by 400 matrix of floats
    var h3: MLMultiArray

    /// c3 as 1 by 400 matrix of floats
    var c3: MLMultiArray

    /// kappa as 1 by 10 matrix of floats
    var kappa: MLMultiArray

    /// window as 1 by 73 matrix of floats
    var window: MLMultiArray

    var featureNames: Set<String> { ["stroke", "chars", "chars_len", "h1", "c1", "h2", "c2", "h3", "c3", "kappa", "window"] }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        if featureName == "stroke" {
            return MLFeatureValue(multiArray: stroke)
        }
        if featureName == "chars" {
            return MLFeatureValue(multiArray: chars)
        }
        if featureName == "chars_len" {
            return MLFeatureValue(multiArray: chars_len)
        }
        if featureName == "h1" {
            return MLFeatureValue(multiArray: h1)
        }
        if featureName == "c1" {
            return MLFeatureValue(multiArray: c1)
        }
        if featureName == "h2" {
            return MLFeatureValue(multiArray: h2)
        }
        if featureName == "c2" {
            return MLFeatureValue(multiArray: c2)
        }
        if featureName == "h3" {
            return MLFeatureValue(multiArray: h3)
        }
        if featureName == "c3" {
            return MLFeatureValue(multiArray: c3)
        }
        if featureName == "kappa" {
            return MLFeatureValue(multiArray: kappa)
        }
        if featureName == "window" {
            return MLFeatureValue(multiArray: window)
        }
        return nil
    }

    init(stroke: MLMultiArray, chars: MLMultiArray, chars_len: MLMultiArray, h1: MLMultiArray, c1: MLMultiArray, h2: MLMultiArray, c2: MLMultiArray, h3: MLMultiArray, c3: MLMultiArray, kappa: MLMultiArray, window: MLMultiArray) {
        self.stroke = stroke
        self.chars = chars
        self.chars_len = chars_len
        self.h1 = h1
        self.c1 = c1
        self.h2 = h2
        self.c2 = c2
        self.h3 = h3
        self.c3 = c3
        self.kappa = kappa
        self.window = window
    }

    convenience init(stroke: MLShapedArray<Float>, chars: MLShapedArray<Int32>, chars_len: MLShapedArray<Int32>, h1: MLShapedArray<Float>, c1: MLShapedArray<Float>, h2: MLShapedArray<Float>, c2: MLShapedArray<Float>, h3: MLShapedArray<Float>, c3: MLShapedArray<Float>, kappa: MLShapedArray<Float>, window: MLShapedArray<Float>) {
        self.init(stroke: MLMultiArray(stroke), chars: MLMultiArray(chars), chars_len: MLMultiArray(chars_len), h1: MLMultiArray(h1), c1: MLMultiArray(c1), h2: MLMultiArray(h2), c2: MLMultiArray(c2), h3: MLMultiArray(h3), c3: MLMultiArray(c3), kappa: MLMultiArray(kappa), window: MLMultiArray(window))
    }

}


/// Model Prediction Output Type
@available(macOS 14.0, iOS 17.0, tvOS 17.0, watchOS 10.0, visionOS 1.0, *)
class HandwritingStepOutput : MLFeatureProvider {

    /// Source provided by CoreML
    private let provider : MLFeatureProvider

    /// Raw 20-component GMM parameters. as 1 by 121 matrix of floats
    var gmm_params: MLMultiArray {
        provider.featureValue(for: "gmm_params")!.multiArrayValue!
    }

    /// Raw 20-component GMM parameters. as 1 by 121 matrix of floats
    var gmm_paramsShapedArray: MLShapedArray<Float> {
        MLShapedArray<Float>(gmm_params)
    }

    /// next_h1 as 1 by 400 matrix of floats
    var next_h1: MLMultiArray {
        provider.featureValue(for: "next_h1")!.multiArrayValue!
    }

    /// next_h1 as 1 by 400 matrix of floats
    var next_h1ShapedArray: MLShapedArray<Float> {
        MLShapedArray<Float>(next_h1)
    }

    /// next_c1 as 1 by 400 matrix of floats
    var next_c1: MLMultiArray {
        provider.featureValue(for: "next_c1")!.multiArrayValue!
    }

    /// next_c1 as 1 by 400 matrix of floats
    var next_c1ShapedArray: MLShapedArray<Float> {
        MLShapedArray<Float>(next_c1)
    }

    /// next_h2 as 1 by 400 matrix of floats
    var next_h2: MLMultiArray {
        provider.featureValue(for: "next_h2")!.multiArrayValue!
    }

    /// next_h2 as 1 by 400 matrix of floats
    var next_h2ShapedArray: MLShapedArray<Float> {
        MLShapedArray<Float>(next_h2)
    }

    /// next_c2 as 1 by 400 matrix of floats
    var next_c2: MLMultiArray {
        provider.featureValue(for: "next_c2")!.multiArrayValue!
    }

    /// next_c2 as 1 by 400 matrix of floats
    var next_c2ShapedArray: MLShapedArray<Float> {
        MLShapedArray<Float>(next_c2)
    }

    /// next_h3 as 1 by 400 matrix of floats
    var next_h3: MLMultiArray {
        provider.featureValue(for: "next_h3")!.multiArrayValue!
    }

    /// next_h3 as 1 by 400 matrix of floats
    var next_h3ShapedArray: MLShapedArray<Float> {
        MLShapedArray<Float>(next_h3)
    }

    /// next_c3 as 1 by 400 matrix of floats
    var next_c3: MLMultiArray {
        provider.featureValue(for: "next_c3")!.multiArrayValue!
    }

    /// next_c3 as 1 by 400 matrix of floats
    var next_c3ShapedArray: MLShapedArray<Float> {
        MLShapedArray<Float>(next_c3)
    }

    /// next_kappa as 1 by 10 matrix of floats
    var next_kappa: MLMultiArray {
        provider.featureValue(for: "next_kappa")!.multiArrayValue!
    }

    /// next_kappa as 1 by 10 matrix of floats
    var next_kappaShapedArray: MLShapedArray<Float> {
        MLShapedArray<Float>(next_kappa)
    }

    /// next_window as 1 by 73 matrix of floats
    var next_window: MLMultiArray {
        provider.featureValue(for: "next_window")!.multiArrayValue!
    }

    /// next_window as 1 by 73 matrix of floats
    var next_windowShapedArray: MLShapedArray<Float> {
        MLShapedArray<Float>(next_window)
    }

    /// next_phi as 1 by 120 matrix of floats
    var next_phi: MLMultiArray {
        provider.featureValue(for: "next_phi")!.multiArrayValue!
    }

    /// next_phi as 1 by 120 matrix of floats
    var next_phiShapedArray: MLShapedArray<Float> {
        MLShapedArray<Float>(next_phi)
    }

    var featureNames: Set<String> {
        provider.featureNames
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        provider.featureValue(for: featureName)
    }

    init(gmm_params: MLMultiArray, next_h1: MLMultiArray, next_c1: MLMultiArray, next_h2: MLMultiArray, next_c2: MLMultiArray, next_h3: MLMultiArray, next_c3: MLMultiArray, next_kappa: MLMultiArray, next_window: MLMultiArray, next_phi: MLMultiArray) {
        self.provider = try! MLDictionaryFeatureProvider(dictionary: ["gmm_params" : MLFeatureValue(multiArray: gmm_params), "next_h1" : MLFeatureValue(multiArray: next_h1), "next_c1" : MLFeatureValue(multiArray: next_c1), "next_h2" : MLFeatureValue(multiArray: next_h2), "next_c2" : MLFeatureValue(multiArray: next_c2), "next_h3" : MLFeatureValue(multiArray: next_h3), "next_c3" : MLFeatureValue(multiArray: next_c3), "next_kappa" : MLFeatureValue(multiArray: next_kappa), "next_window" : MLFeatureValue(multiArray: next_window), "next_phi" : MLFeatureValue(multiArray: next_phi)])
    }

    init(features: MLFeatureProvider) {
        self.provider = features
    }
}


/// Class for model loading and prediction
@available(macOS 14.0, iOS 17.0, tvOS 17.0, watchOS 10.0, visionOS 1.0, *)
class HandwritingStep {
    let model: MLModel

    /// URL of model assuming it was installed in the same bundle as this class
    class var urlOfModelInThisBundle : URL {
        return Bundle.module.url(
            forResource: "HandwritingStep",
            withExtension: "mlmodelc",
            subdirectory: "Models"
        ) ?? Bundle.module.url(forResource: "HandwritingStep", withExtension: "mlmodelc")!
    }

    /**
        Construct HandwritingStep instance with an existing MLModel object.

        Usually the application does not use this initializer unless it makes a subclass of HandwritingStep.
        Such application may want to use `MLModel(contentsOfURL:configuration:)` and `HandwritingStep.urlOfModelInThisBundle` to create a MLModel object to pass-in.

        - parameters:
          - model: MLModel object
    */
    init(model: MLModel) {
        self.model = model
    }

    /**
        Construct a model with configuration

        - parameters:
           - configuration: the desired model configuration

        - throws: an NSError object that describes the problem
    */
    convenience init(configuration: MLModelConfiguration = MLModelConfiguration()) throws {
        try self.init(contentsOf: type(of:self).urlOfModelInThisBundle, configuration: configuration)
    }

    /**
        Construct HandwritingStep instance with explicit path to mlmodelc file
        - parameters:
           - modelURL: the file url of the model

        - throws: an NSError object that describes the problem
    */
    convenience init(contentsOf modelURL: URL) throws {
        try self.init(model: MLModel(contentsOf: modelURL))
    }

    /**
        Construct a model with URL of the .mlmodelc directory and configuration

        - parameters:
           - modelURL: the file url of the model
           - configuration: the desired model configuration

        - throws: an NSError object that describes the problem
    */
    convenience init(contentsOf modelURL: URL, configuration: MLModelConfiguration) throws {
        try self.init(model: MLModel(contentsOf: modelURL, configuration: configuration))
    }

    /**
        Construct HandwritingStep instance asynchronously with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - configuration: the desired model configuration
          - handler: the completion handler to be called when the model loading completes successfully or unsuccessfully
    */
    class func load(configuration: MLModelConfiguration = MLModelConfiguration(), completionHandler handler: @escaping (Swift.Result<HandwritingStep, Error>) -> Void) {
        load(contentsOf: self.urlOfModelInThisBundle, configuration: configuration, completionHandler: handler)
    }

    /**
        Construct HandwritingStep instance asynchronously with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - configuration: the desired model configuration
    */
    class func load(configuration: MLModelConfiguration = MLModelConfiguration()) async throws -> HandwritingStep {
        try await load(contentsOf: self.urlOfModelInThisBundle, configuration: configuration)
    }

    /**
        Construct HandwritingStep instance asynchronously with URL of the .mlmodelc directory with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - modelURL: the URL to the model
          - configuration: the desired model configuration
          - handler: the completion handler to be called when the model loading completes successfully or unsuccessfully
    */
    class func load(contentsOf modelURL: URL, configuration: MLModelConfiguration = MLModelConfiguration(), completionHandler handler: @escaping (Swift.Result<HandwritingStep, Error>) -> Void) {
        MLModel.load(contentsOf: modelURL, configuration: configuration) { result in
            switch result {
            case .failure(let error):
                handler(.failure(error))
            case .success(let model):
                handler(.success(HandwritingStep(model: model)))
            }
        }
    }

    /**
        Construct HandwritingStep instance asynchronously with URL of the .mlmodelc directory with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - modelURL: the URL to the model
          - configuration: the desired model configuration
    */
    class func load(contentsOf modelURL: URL, configuration: MLModelConfiguration = MLModelConfiguration()) async throws -> HandwritingStep {
        let model = try await MLModel.load(contentsOf: modelURL, configuration: configuration)
        return HandwritingStep(model: model)
    }

    /**
        Make a prediction using the structured interface

        It uses the default function if the model has multiple functions.

        - parameters:
           - input: the input to the prediction as HandwritingStepInput

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as HandwritingStepOutput
    */
    func prediction(input: HandwritingStepInput) throws -> HandwritingStepOutput {
        try prediction(input: input, options: MLPredictionOptions())
    }

    /**
        Make a prediction using the structured interface

        It uses the default function if the model has multiple functions.

        - parameters:
           - input: the input to the prediction as HandwritingStepInput
           - options: prediction options

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as HandwritingStepOutput
    */
    func prediction(input: HandwritingStepInput, options: MLPredictionOptions) throws -> HandwritingStepOutput {
        let outFeatures = try model.prediction(from: input, options: options)
        return HandwritingStepOutput(features: outFeatures)
    }

    /**
        Make an asynchronous prediction using the structured interface

        It uses the default function if the model has multiple functions.

        - parameters:
           - input: the input to the prediction as HandwritingStepInput
           - options: prediction options

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as HandwritingStepOutput
    */
    func prediction(input: HandwritingStepInput, options: MLPredictionOptions = MLPredictionOptions()) async throws -> HandwritingStepOutput {
        let outFeatures = try await model.prediction(from: input, options: options)
        return HandwritingStepOutput(features: outFeatures)
    }

    /**
        Make a prediction using the convenience interface

        It uses the default function if the model has multiple functions.

        - parameters:
            - stroke: Previous sampled x/y offset and pen state. as 1 by 3 matrix of floats
            - chars: Encoded text padded to 120 characters. as 1 by 120 matrix of 32-bit integers
            - chars_len: Encoded text length including terminator. as 1 element vector of 32-bit integers
            - h1: 1 by 400 matrix of floats
            - c1: 1 by 400 matrix of floats
            - h2: 1 by 400 matrix of floats
            - c2: 1 by 400 matrix of floats
            - h3: 1 by 400 matrix of floats
            - c3: 1 by 400 matrix of floats
            - kappa: 1 by 10 matrix of floats
            - window: 1 by 73 matrix of floats

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as HandwritingStepOutput
    */
    func prediction(stroke: MLMultiArray, chars: MLMultiArray, chars_len: MLMultiArray, h1: MLMultiArray, c1: MLMultiArray, h2: MLMultiArray, c2: MLMultiArray, h3: MLMultiArray, c3: MLMultiArray, kappa: MLMultiArray, window: MLMultiArray) throws -> HandwritingStepOutput {
        let input_ = HandwritingStepInput(stroke: stroke, chars: chars, chars_len: chars_len, h1: h1, c1: c1, h2: h2, c2: c2, h3: h3, c3: c3, kappa: kappa, window: window)
        return try prediction(input: input_)
    }

    /**
        Make a prediction using the convenience interface

        It uses the default function if the model has multiple functions.

        - parameters:
            - stroke: Previous sampled x/y offset and pen state. as 1 by 3 matrix of floats
            - chars: Encoded text padded to 120 characters. as 1 by 120 matrix of 32-bit integers
            - chars_len: Encoded text length including terminator. as 1 element vector of 32-bit integers
            - h1: 1 by 400 matrix of floats
            - c1: 1 by 400 matrix of floats
            - h2: 1 by 400 matrix of floats
            - c2: 1 by 400 matrix of floats
            - h3: 1 by 400 matrix of floats
            - c3: 1 by 400 matrix of floats
            - kappa: 1 by 10 matrix of floats
            - window: 1 by 73 matrix of floats

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as HandwritingStepOutput
    */

    func prediction(stroke: MLShapedArray<Float>, chars: MLShapedArray<Int32>, chars_len: MLShapedArray<Int32>, h1: MLShapedArray<Float>, c1: MLShapedArray<Float>, h2: MLShapedArray<Float>, c2: MLShapedArray<Float>, h3: MLShapedArray<Float>, c3: MLShapedArray<Float>, kappa: MLShapedArray<Float>, window: MLShapedArray<Float>) throws -> HandwritingStepOutput {
        let input_ = HandwritingStepInput(stroke: stroke, chars: chars, chars_len: chars_len, h1: h1, c1: c1, h2: h2, c2: c2, h3: h3, c3: c3, kappa: kappa, window: window)
        return try prediction(input: input_)
    }

    /**
        Make a batch prediction using the structured interface

        It uses the default function if the model has multiple functions.

        - parameters:
           - inputs: the inputs to the prediction as [HandwritingStepInput]
           - options: prediction options

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as [HandwritingStepOutput]
    */
    func predictions(inputs: [HandwritingStepInput], options: MLPredictionOptions = MLPredictionOptions()) throws -> [HandwritingStepOutput] {
        let batchIn = MLArrayBatchProvider(array: inputs)
        let batchOut = try model.predictions(from: batchIn, options: options)
        var results : [HandwritingStepOutput] = []
        results.reserveCapacity(inputs.count)
        for i in 0..<batchOut.count {
            let outProvider = batchOut.features(at: i)
            let result =  HandwritingStepOutput(features: outProvider)
            results.append(result)
        }
        return results
    }
}
