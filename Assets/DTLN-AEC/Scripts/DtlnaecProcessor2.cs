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
    // Padding size (block_len - block_shift)
    private const int PaddingSize = BlockLen - BlockShift; // 384

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

    // Buffers for real-time processing (与Python一致的滑动窗口)
    private float[] _inputBuffer = new float[BlockLen];
    private float[] _lpbBuffer = new float[BlockLen];
    private float[] _outputBuffer = new float[BlockLen];

    // Frame counter for tracking processing state
    private int _framesProcessed = 0;

    // 用于累积padding帧
    private bool _isPaddingPhase = true;
    private int _paddingFramesReceived = 0;
    private const int PaddingFrames = PaddingSize / BlockShift; // 384/128 = 3 frames

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

        // Reset buffers - 初始化为全零（相当于Python的padding）
        Array.Clear(_inputBuffer, 0, _inputBuffer.Length);
        Array.Clear(_lpbBuffer, 0, _lpbBuffer.Length);
        Array.Clear(_outputBuffer, 0, _outputBuffer.Length);

        _framesProcessed = 0;
        _isPaddingPhase = true;
        _paddingFramesReceived = 0;
    }

    /// <summary>
    /// Process a frame of audio data for real-time streaming
    /// 完全按照Python逻辑实现的滑动窗口处理
    /// </summary>
    /// <param name="micFrame">Microphone audio frame (must be BlockShift samples)</param>
    /// <param name="lpbFrame">Loudspeaker audio frame (must be BlockShift samples)</param>
    /// <returns>Processed audio frame (BlockShift samples)</returns>
    public float[] ProcessFrame(float[] micFrame, float[] lpbFrame)
    {
        if (micFrame.Length != BlockShift || lpbFrame.Length != BlockShift)
        {
            Debug.LogError($"Input frames must be exactly {BlockShift} samples");
            return new float[BlockShift];
        }

        if (_session1 == null || _session2 == null)
        {
            Debug.LogError("DTLN-AEC processor not initialized");
            return new float[BlockShift];
        }

        // 前3帧作为padding（模拟Python的初始padding）
        if (_isPaddingPhase)
        {
            _paddingFramesReceived++;
            if (_paddingFramesReceived >= PaddingFrames)
            {
                _isPaddingPhase = false;
            }
            // padding阶段返回静音
            return new float[BlockShift];
        }

        // === 滑动窗口更新（与Python完全一致） ===
        // Python: in_buffer[:-block_shift] = in_buffer[block_shift:]
        Array.Copy(_inputBuffer, BlockShift, _inputBuffer, 0, BlockLen - BlockShift);
        Array.Copy(_lpbBuffer, BlockShift, _lpbBuffer, 0, BlockLen - BlockShift);

        // Python: in_buffer[-block_shift:] = audio[idx * block_shift : (idx * block_shift) + block_shift]
        Array.Copy(micFrame, 0, _inputBuffer, BlockLen - BlockShift, BlockShift);
        Array.Copy(lpbFrame, 0, _lpbBuffer, BlockLen - BlockShift, BlockShift);

        // === 处理完整的block ===
        ProcessBlock(_inputBuffer, _lpbBuffer);

        // === 提取输出（从overlap-add buffer的前BlockShift个样本） ===
        // Python: out_file[idx * block_shift : (idx * block_shift) + block_shift] = out_buffer[:block_shift]
        float[] outputFrame = new float[BlockShift];
        Array.Copy(_outputBuffer, 0, outputFrame, 0, BlockShift);

        _framesProcessed++;
        return outputFrame;
    }

    /// <summary>
    /// Process any remaining audio in the buffers (for end of stream)
    /// </summary>
    public float[] Flush()
    {
        // 处理最后的padding帧
        List<float> finalOutput = new List<float>();

        // 输出剩余的PaddingFrames帧
        for (int i = 0; i < PaddingFrames; i++)
        {
            float[] zeroFrame = new float[BlockShift];
            float[] output = ProcessFrame(zeroFrame, zeroFrame);
            finalOutput.AddRange(output);
        }

        return finalOutput.ToArray();
    }

    private void ProcessBlock(float[] inputBlock, float[] lpbBlock)
    {
        // === 1. FFT计算 ===
        var inBlockFft = PerformRfft(inputBlock);
        var lpbBlockFft = PerformRfft(lpbBlock);

        // === 2. 计算幅度谱 ===
        var inMag = new DenseTensor<float>(dimensions: new[] { 1, 1, FftHalfSize });
        var lpbMag = new DenseTensor<float>(dimensions: new[] { 1, 1, FftHalfSize });

        for (int i = 0; i < FftHalfSize; i++)
        {
            inMag[0, 0, i] = (float)inBlockFft[i].Magnitude;
            lpbMag[0, 0, i] = (float)lpbBlockFft[i].Magnitude;
        }

        // === 3. 运行Model 1 ===
        var inputs1 = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputNames1[0], inMag),
            NamedOnnxValue.CreateFromTensor(_inputNames1[2], lpbMag),
            NamedOnnxValue.CreateFromTensor(_inputNames1[1], _states1)
        };

        using (var outputs1 = _session1.Run(inputs1))
        {
            var outMask = outputs1.First(v => v.Name == _outputNames1[0]).AsTensor<float>();
            _states1 = outputs1.First(v => v.Name == _outputNames1[1]).AsTensor<float>().ToDenseTensor();

            // === 4. 应用mask并执行IFFT ===
            for (int i = 0; i < FftHalfSize; i++)
            {
                float maskValue = outMask[0, 0, i];
                inBlockFft[i] = new Complex(
                    inBlockFft[i].Real * maskValue,
                    inBlockFft[i].Imaginary * maskValue
                );
            }
        }

        var estimatedBlockTime = PerformIrfft(inBlockFft);

        // === 5. 准备Model 2的输入 ===
        var estimatedBlockTensor = new DenseTensor<float>(dimensions: new[] { 1, 1, BlockLen });
        var inLpbTensor = new DenseTensor<float>(dimensions: new[] { 1, 1, BlockLen });

        for (int i = 0; i < BlockLen; i++)
        {
            estimatedBlockTensor[0, 0, i] = estimatedBlockTime[i];
            inLpbTensor[0, 0, i] = lpbBlock[i];
        }

        // === 6. 运行Model 2 ===
        var inputs2 = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputNames2[0], estimatedBlockTensor),
            NamedOnnxValue.CreateFromTensor(_inputNames2[2], inLpbTensor),
            NamedOnnxValue.CreateFromTensor(_inputNames2[1], _states2)
        };

        using (var outputs2 = _session2.Run(inputs2))
        {
            var outBlock = outputs2.First(v => v.Name == _outputNames2[0]).AsTensor<float>() as DenseTensor<float>;
            _states2 = outputs2.First(v => v.Name == _outputNames2[1]).AsTensor<float>().ToDenseTensor();

            // === 7. Overlap-Add处理（与Python完全一致） ===
            // Python: out_buffer[:-block_shift] = out_buffer[block_shift:]
            Array.Copy(_outputBuffer, BlockShift, _outputBuffer, 0, BlockLen - BlockShift);
            // Python: out_buffer[-block_shift:] = np.zeros((block_shift))
            Array.Clear(_outputBuffer, BlockLen - BlockShift, BlockShift);

            // Python: out_buffer += np.squeeze(out_block)
            var outBlockSpan = outBlock.Buffer.Span;
            for (int i = 0; i < BlockLen; i++)
            {
                _outputBuffer[i] += outBlockSpan[i];
            }
        }
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

        // Fill the second half with complex conjugates (for real signal)
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
    public bool IsPaddingPhase => _isPaddingPhase;
}