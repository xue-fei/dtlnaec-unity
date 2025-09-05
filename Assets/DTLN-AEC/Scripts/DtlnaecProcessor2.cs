using System;
using System.Linq;
using System.Numerics;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MathNet.Numerics.IntegralTransforms;
using UnityEngine;

public class DtlnaecProcessor2
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

    // Buffers for real-time processing
    private float[] _inputBuffer = new float[BlockLen];
    private float[] _lpbBuffer = new float[BlockLen];
    private float[] _outputBuffer = new float[BlockLen];
    private int _bufferPosition = 0;

    // Frame counter for tracking processing state
    private int _framesProcessed = 0;

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

            _states1 = new DenseTensor<float>(new ReadOnlySpan<int>(stateShape1.ToArray()), false);
            _states2 = new DenseTensor<float>(new ReadOnlySpan<int>(stateShape2.ToArray()), false);

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
        Array.Clear(_inputBuffer, 0, _inputBuffer.Length);
        Array.Clear(_lpbBuffer, 0, _lpbBuffer.Length);
        Array.Clear(_outputBuffer, 0, _outputBuffer.Length);
        _bufferPosition = 0;
        _framesProcessed = 0;
    }

    /// <summary>
    /// Process a frame of audio data for real-time streaming
    /// </summary>
    /// <param name="micFrame">Microphone audio frame (must be BlockShift samples)</param>
    /// <param name="lpbFrame">Loudspeaker audio frame (must be BlockShift samples)</param>
    /// <returns>Processed audio frame (BlockShift samples) or null if not enough data</returns>
    public float[] ProcessFrame(float[] micFrame, float[] lpbFrame)
    {
        if (micFrame.Length != BlockShift || lpbFrame.Length != BlockShift)
        {
            Debug.LogError($"Input frames must be exactly {BlockShift} samples");
            return null;
        }

        if (_session1 == null || _session2 == null)
        {
            Debug.LogError("DTLN-AEC processor not initialized");
            return null;
        }

        // Add new data to buffers
        Array.Copy(micFrame, 0, _inputBuffer, _bufferPosition, BlockShift);
        Array.Copy(lpbFrame, 0, _lpbBuffer, _bufferPosition, BlockShift);
        _bufferPosition += BlockShift;

        // Check if we have enough data for processing
        if (_bufferPosition < BlockLen)
        {
            // Not enough data yet, return silent frame
            return new float[BlockShift];
        }

        // Process the block
        float[] processedBlock = ProcessBlock(_inputBuffer, _lpbBuffer);

        // Shift buffers for next iteration
        Array.Copy(_inputBuffer, BlockShift, _inputBuffer, 0, BlockLen - BlockShift);
        Array.Copy(_lpbBuffer, BlockShift, _lpbBuffer, 0, BlockLen - BlockShift);
        _bufferPosition = BlockLen - BlockShift;

        // Extract the output frame (first BlockShift samples)
        float[] outputFrame = new float[BlockShift];
        Array.Copy(processedBlock, 0, outputFrame, 0, BlockShift);

        _framesProcessed++;
        return outputFrame;
    }

    /// <summary>
    /// Process any remaining audio in the buffers (for end of stream)
    /// </summary>
    /// <returns>Processed audio frames or null if no data</returns>
    public float[] Flush()
    {
        if (_bufferPosition == 0)
            return null;

        // Pad with zeros to complete the block
        Array.Clear(_inputBuffer, _bufferPosition, BlockLen - _bufferPosition);
        Array.Clear(_lpbBuffer, _bufferPosition, BlockLen - _bufferPosition);

        // Process the final block
        float[] processedBlock = ProcessBlock(_inputBuffer, _lpbBuffer);

        // Reset buffer position
        _bufferPosition = 0;

        // Return the valid portion of the processed block
        int validSamples = Math.Min(BlockShift, _bufferPosition);
        float[] output = new float[validSamples];
        Array.Copy(processedBlock, 0, output, 0, validSamples);

        return output;
    }

    private float[] ProcessBlock(float[] inputBlock, float[] lpbBlock)
    {
        // --- FFT ---
        var inBlockFft = PerformRfft(inputBlock);
        var lpbBlockFft = PerformRfft(lpbBlock);

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
            inLpbTensor[0, 0, i] = lpbBlock[i];
        }

        var inputs2 = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputNames2[0], estimatedBlockTensor),
            NamedOnnxValue.CreateFromTensor(_inputNames2[2], inLpbTensor),
            NamedOnnxValue.CreateFromTensor(_inputNames2[1], _states2)
        };

        using var outputs2 = _session2.Run(inputs2);
        var outBlock = outputs2.First(v => v.Name == _outputNames2[0]).AsTensor<float>() as DenseTensor<float>;
        _states2 = outputs2.First(v => v.Name == _outputNames2[1]).AsTensor<float>().ToDenseTensor();

        // Apply overlap-add to output buffer
        Array.Copy(_outputBuffer, BlockShift, _outputBuffer, 0, BlockLen - BlockShift);
        Array.Clear(_outputBuffer, BlockLen - BlockShift, BlockShift);

        var outBlockSpan = outBlock.Buffer.Span;
        for (int i = 0; i < BlockLen; i++)
        {
            _outputBuffer[i] += outBlockSpan[i];
        }

        // Return a copy of the output buffer
        float[] result = new float[BlockLen];
        Array.Copy(_outputBuffer, result, BlockLen);
        return result;
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

    // Properties for monitoring
    public int FramesProcessed => _framesProcessed;
    public int BufferFill => _bufferPosition;
}