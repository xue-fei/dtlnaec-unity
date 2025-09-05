using System;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MathNet.Numerics.IntegralTransforms;
using UnityEngine;

public class DtlnaecProcessor
{
    // Constants from the Python script
    private const int BlockLen = 512;
    private const int BlockShift = 128;
    private const int FftSize = BlockLen;
    private const int RequiredSampleRate = 16000;
    // RFFT returns (N/2)+1 complex numbers
    private const int FftHalfSize = (FftSize / 2) + 1;

    // ONNX session instances
    private InferenceSession _session1;
    private InferenceSession _session2;

    // State tensors
    private DenseTensor<float> _states1;
    private DenseTensor<float> _states2;

    // Input/output names
    private List<string> _inputNames1;
    private List<string> _outputNames1;
    private List<string> _inputNames2;
    private List<string> _outputNames2;

    // Buffers
    private float[] _inBuffer = new float[BlockLen];
    private float[] _inBufferLpb = new float[BlockLen];
    private float[] _outBuffer = new float[BlockLen];

    public bool Initialize(string model1Path, string model2Path)
    {
        try
        {
            // Use recommended session options for performance
            var sessionOptions = new SessionOptions();
            sessionOptions.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            sessionOptions.InterOpNumThreads = 1;
            sessionOptions.IntraOpNumThreads = 1;

            // Load ONNX models
            _session1 = new InferenceSession(model1Path, sessionOptions);
            _session2 = new InferenceSession(model2Path, sessionOptions);

            // Get input/output names
            _inputNames1 = _session1.InputMetadata.Keys.ToList();
            _outputNames1 = _session1.OutputMetadata.Keys.ToList();
            _inputNames2 = _session2.InputMetadata.Keys.ToList();
            _outputNames2 = _session2.OutputMetadata.Keys.ToList();

            // Initialize state tensors
            var stateShape1 = _session1.InputMetadata[_inputNames1[1]].Dimensions;
            var stateShape2 = _session2.InputMetadata[_inputNames2[1]].Dimensions;

            _states1 = new DenseTensor<float>(stateShape1);
            _states2 = new DenseTensor<float>(stateShape2);

            // Reset states
            ResetStates();

            Debug.Log("DTLN-AEC processor initialized successfully");
            return true;
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to initialize DTLN-AEC processor: {ex.Message}");
            return false;
        }
    }

    public void ResetStates()
    {
        // Reset state tensors to zeros
        if (_states1 != null)
        {
            _states1.Buffer.Span.Clear();
        }

        if (_states2 != null)
        {
            _states2.Buffer.Span.Clear();
        }

        // Reset buffers
        Array.Clear(_inBuffer, 0, _inBuffer.Length);
        Array.Clear(_inBufferLpb, 0, _inBufferLpb.Length);
        Array.Clear(_outBuffer, 0, _outBuffer.Length);
    }

    public float[] ProcessAudio(float[] micAudio, float[] lpbAudio)
    {
        if (_session1 == null || _session2 == null)
        {
            Debug.LogError("DTLN-AEC processor not initialized");
            return null;
        }

        // Ensure audio lengths are the same
        int minLen = Math.Min(micAudio.Length, lpbAudio.Length);
        var micAudioTrimmed = new float[minLen];
        var lpbAudioTrimmed = new float[minLen];
        Array.Copy(micAudio, micAudioTrimmed, minLen);
        Array.Copy(lpbAudio, lpbAudioTrimmed, minLen);

        int originalLen = minLen;

        // Pad audio
        var padding = new float[BlockLen - BlockShift];
        var micPadded = new float[padding.Length * 2 + micAudioTrimmed.Length];
        var lpbPadded = new float[padding.Length * 2 + lpbAudioTrimmed.Length];

        Array.Copy(padding, 0, micPadded, 0, padding.Length);
        Array.Copy(micAudioTrimmed, 0, micPadded, padding.Length, micAudioTrimmed.Length);
        Array.Copy(padding, 0, micPadded, padding.Length + micAudioTrimmed.Length, padding.Length);

        Array.Copy(padding, 0, lpbPadded, 0, padding.Length);
        Array.Copy(lpbAudioTrimmed, 0, lpbPadded, padding.Length, lpbAudioTrimmed.Length);
        Array.Copy(padding, 0, lpbPadded, padding.Length + lpbAudioTrimmed.Length, padding.Length);

        // Preallocate output file
        var outFile = new float[micPadded.Length];

        // Calculate number of blocks
        int numBlocks = (micPadded.Length - (BlockLen - BlockShift)) / BlockShift;

        // Process each block
        for (int idx = 0; idx < numBlocks; idx++)
        {
            int start = idx * BlockShift;

            // Shift and update buffers
            Array.Copy(_inBuffer, BlockShift, _inBuffer, 0, BlockLen - BlockShift);
            Array.Copy(micPadded, start, _inBuffer, BlockLen - BlockShift, BlockShift);

            Array.Copy(_inBufferLpb, BlockShift, _inBufferLpb, 0, BlockLen - BlockShift);
            Array.Copy(lpbPadded, start, _inBufferLpb, BlockLen - BlockShift, BlockShift);

            // Process the current block
            ProcessBlock(outFile, start);
        }

        // Trim to original length
        var predictedSpeech = new float[originalLen];
        Array.Copy(outFile, BlockLen - BlockShift, predictedSpeech, 0, originalLen);

        // Normalize if clipping occurs
        float maxVal = predictedSpeech.Max(x => Math.Abs(x));
        if (maxVal > 1.0f)
        {
            for (int i = 0; i < predictedSpeech.Length; i++)
            {
                predictedSpeech[i] = (predictedSpeech[i] / maxVal) * 0.99f;
            }
        }

        return predictedSpeech;
    }

    private void ProcessBlock(float[] outFile, int startIndex)
    {
        // --- FFT ---
        var inBlockFft = PerformRfft(_inBuffer);
        var lpbBlockFft = PerformRfft(_inBufferLpb);

        // Calculate magnitude for model 1 input
        var inMag = new DenseTensor<float>(dimensions: new[] { 1, 1, FftHalfSize });
        var lpbMag = new DenseTensor<float>(dimensions: new[] { 1, 1, FftHalfSize });

        for (int i = 0; i < FftHalfSize; i++)
        {
            inMag[0, 0, i] = (float)inBlockFft[i].Magnitude;
            lpbMag[0, 0, i] = (float)lpbBlockFft[i].Magnitude;
        }

        // --- Run Model 1 ---
        var inputs1 = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputNames1[0], inMag),
            NamedOnnxValue.CreateFromTensor(_inputNames1[2], lpbMag),
            NamedOnnxValue.CreateFromTensor(_inputNames1[1], _states1)
        };

        using var outputs1 = _session1.Run(inputs1);
        var outMask = outputs1.First(v => v.Name == _outputNames1[0]).AsTensor<float>();
        _states1 = outputs1.First(v => v.Name == _outputNames1[1]).AsTensor<float>().ToDenseTensor();

        // --- Apply Mask and IFFT ---
        for (int i = 0; i < FftHalfSize; i++)
        {
            inBlockFft[i] = new Complex(
                inBlockFft[i].Real * outMask[0, 0, i],
                inBlockFft[i].Imaginary * outMask[0, 0, i]
            );
        }

        var estimatedBlockTime = PerformIrfft(inBlockFft);

        // --- Run Model 2 ---
        var estimatedBlockTensor = new DenseTensor<float>(dimensions: new[] { 1, 1, BlockLen });
        var inLpbTensor = new DenseTensor<float>(dimensions: new[] { 1, 1, BlockLen });

        for (int i = 0; i < BlockLen; i++)
        {
            estimatedBlockTensor[0, 0, i] = estimatedBlockTime[i];
            inLpbTensor[0, 0, i] = _inBufferLpb[i];
        }

        var inputs2 = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputNames2[0], estimatedBlockTensor),
            NamedOnnxValue.CreateFromTensor(_inputNames2[2], inLpbTensor),
            NamedOnnxValue.CreateFromTensor(_inputNames2[1], _states2)
        };

        using var outputs2 = _session2.Run(inputs2);
        var outBlock = outputs2.First(v => v.Name == _outputNames2[0]).AsTensor<float>();
        _states2 = outputs2.First(v => v.Name == _outputNames2[1]).AsTensor<float>().ToDenseTensor();

        // --- Overlap-Add ---
        Array.Copy(_outBuffer, BlockShift, _outBuffer, 0, BlockLen - BlockShift);
        Array.Clear(_outBuffer, BlockLen - BlockShift, BlockShift);

        var outBlockSpan = outBlock.Buffer.Span;
        for (int i = 0; i < BlockLen; i++)
        {
            _outBuffer[i] += outBlockSpan[i];
        }

        // Write to final output array
        Array.Copy(_outBuffer, 0, outFile, startIndex, BlockShift);
    }

    private Complex[] PerformRfft(float[] input)
    {
        var complexInput = new Complex[FftSize];
        for (int i = 0; i < FftSize; i++)
        {
            complexInput[i] = new Complex(input[i], 0);
        }

        Fourier.Forward(complexInput, FourierOptions.Matlab);

        // Return only the first half (N/2 + 1)
        var result = new Complex[FftHalfSize];
        Array.Copy(complexInput, result, FftHalfSize);

        return result;
    }

    private float[] PerformIrfft(Complex[] input)
    {
        // Reconstruct the full spectrum for IFFT
        var fullSpectrum = new Complex[FftSize];
        Array.Copy(input, fullSpectrum, FftHalfSize);

        // Fill the second half with complex conjugates
        for (int i = 1; i < FftHalfSize - 1; i++)
        {
            fullSpectrum[FftSize - i] = Complex.Conjugate(input[i]);
        }

        Fourier.Inverse(fullSpectrum, FourierOptions.Matlab);

        // Return the real part of the result
        var result = new float[FftSize];
        for (int i = 0; i < FftSize; i++)
        {
            result[i] = (float)fullSpectrum[i].Real;
        }

        return result;
    }

    public void Dispose()
    {
        _session1?.Dispose();
        _session2?.Dispose();
        _session1 = null;
        _session2 = null;
    }
}