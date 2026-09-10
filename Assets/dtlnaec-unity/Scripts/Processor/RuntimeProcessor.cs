using System;
using System.Linq;
using System.Numerics;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MathNet.Numerics.IntegralTransforms;
using UnityEngine;

public class RuntimeProcessor
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

    // ── 预分配复用缓冲（减少每帧 GC 分配）──────────────────────────────────
    private float[] _outputFrame = new float[BlockShift];
    private Complex[] _inBlockFft = new Complex[FftHalfSize];
    private Complex[] _lpbBlockFft = new Complex[FftHalfSize];
    private Complex[] _fullSpectrum = new Complex[FftSize];
    private Complex[] _complexInput = new Complex[FftSize];
    private float[] _estimatedBlockTime = new float[FftSize];
    private DenseTensor<float> _inMag;
    private DenseTensor<float> _lpbMag;
    private DenseTensor<float> _estimatedBlockTensor;
    private DenseTensor<float> _inLpbTensor;
    private bool _buffersReady = false;

    // Frame counter for tracking processing state
    private int _framesProcessed = 0;

    // 输出延迟补偿：前 PaddingFrames 帧的输出对应的是 zero-padding 区域，需丢弃
    // 与 FileProcessor 的行为完全对齐：FileProcessor 从 outFile[BlockLen-BlockShift] 开始取结果，
    // 即跳过了前 3 帧（PaddingFrames）的输出。
    private int _outputDelayFrames = 0;
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
        _outputDelayFrames = 0;
        _buffersReady = false;
    }

    private void EnsureBuffers()
    {
        if (_buffersReady) return;
        _inMag = new DenseTensor<float>(dimensions: new[] { 1, 1, FftHalfSize });
        _lpbMag = new DenseTensor<float>(dimensions: new[] { 1, 1, FftHalfSize });
        _estimatedBlockTensor = new DenseTensor<float>(dimensions: new[] { 1, 1, BlockLen });
        _inLpbTensor = new DenseTensor<float>(dimensions: new[] { 1, 1, BlockLen });
        _buffersReady = true;
    }

    /// <summary>
    /// Process a frame of audio data for real-time streaming.
    ///
    /// 与 FileProcessor 对齐的正确实现：
    ///   FileProcessor 在音频首尾各填充 384 个零样本（3帧），
    ///   然后跳过前 384 个输出样本（BlockLen - BlockShift）。
    ///   这意味着模型先用 3 帧零信号"预热"，前 3 帧的输出对应零输入，需丢弃。
    ///
    ///   RuntimeProcessor 在初始化时 buffer 已为全零（ResetStates），
    ///   因此只需：
    ///     1. 每帧正常更新 buffer 并推理（包括前 3 帧）
    ///     2. 前 3 帧的输出（对应 zero-padding 区域）丢弃，返回静音
    ///     3. 第 4 帧起返回真实输出
    ///   这样模型状态与 FileProcessor 完全同步。
    /// </summary>
    /// <param name="micFrame">Microphone audio frame (must be BlockShift=128 samples)</param>
    /// <param name="lpbFrame">Loudspeaker audio frame (must be BlockShift=128 samples)</param>
    /// <returns>Processed audio frame (BlockShift samples)；前3帧为静音（预热期）</returns>
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

        EnsureBuffers();

        // === 滑动窗口更新（与 Python / FileProcessor 完全一致） ===
        // Python: in_buffer[:-block_shift] = in_buffer[block_shift:]
        Array.Copy(_inputBuffer, BlockShift, _inputBuffer, 0, BlockLen - BlockShift);
        Array.Copy(_lpbBuffer, BlockShift, _lpbBuffer, 0, BlockLen - BlockShift);

        // Python: in_buffer[-block_shift:] = new_samples
        Array.Copy(micFrame, 0, _inputBuffer, BlockLen - BlockShift, BlockShift);
        Array.Copy(lpbFrame, 0, _lpbBuffer, BlockLen - BlockShift, BlockShift);

        // === 推理 ===
        ProcessBlock(_inputBuffer, _lpbBuffer);

        // === 输出延迟补偿 ===
        // 前 PaddingFrames（3）帧的推理输出对应 zero-padding 输入区域，
        // 与 FileProcessor 丢弃 outFile[0..383] 的行为一致，返回静音。
        if (_outputDelayFrames < PaddingFrames)
        {
            _outputDelayFrames++;
            Array.Clear(_outputFrame, 0, BlockShift);
            return _outputFrame;
        }

        // === 提取有效输出 ===
        Array.Copy(_outputBuffer, 0, _outputFrame, 0, BlockShift);

        _framesProcessed++;
        return _outputFrame;
    }

    /// <summary>
    /// Process any remaining audio in the buffers (for end of stream)
    /// </summary>
    public float[] Flush()
    {
        // 处理最后的padding帧
        List<float> finalOutput = new List<float>();
        float[] zeroFrame = new float[BlockShift];

        // 输出剩余的PaddingFrames帧
        for (int i = 0; i < PaddingFrames; i++)
        {
            float[] output = ProcessFrame(zeroFrame, zeroFrame);
            finalOutput.AddRange(output);
        }

        return finalOutput.ToArray();
    }

    private void ProcessBlock(float[] inputBlock, float[] lpbBlock)
    {
        // === 1. FFT计算（复用缓冲） ===
        PerformRfft(inputBlock, _inBlockFft);
        PerformRfft(lpbBlock, _lpbBlockFft);

        // === 2. 计算幅度谱（复用 tensor） ===
        for (int i = 0; i < FftHalfSize; i++)
        {
            _inMag[0, 0, i] = (float)_inBlockFft[i].Magnitude;
            _lpbMag[0, 0, i] = (float)_lpbBlockFft[i].Magnitude;
        }

        // === 3. 运行Model 1 ===
        var inputs1 = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputNames1[0], _inMag),
            NamedOnnxValue.CreateFromTensor(_inputNames1[2], _lpbMag),
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
                _inBlockFft[i] = new Complex(
                    _inBlockFft[i].Real * maskValue,
                    _inBlockFft[i].Imaginary * maskValue
                );
            }
        }

        var estimatedBlockTime = PerformIrfft(_inBlockFft);

        // === 5. 准备Model 2的输入（复用 tensor） ===
        for (int i = 0; i < BlockLen; i++)
        {
            _estimatedBlockTensor[0, 0, i] = estimatedBlockTime[i];
            _inLpbTensor[0, 0, i] = lpbBlock[i];
        }

        // === 6. 运行Model 2 ===
        var inputs2 = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputNames2[0], _estimatedBlockTensor),
            NamedOnnxValue.CreateFromTensor(_inputNames2[2], _inLpbTensor),
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

    private void PerformRfft(float[] input, Complex[] result)
    {
        for (int i = 0; i < FftSize; i++)
        {
            _complexInput[i] = new Complex(input[i], 0);
        }

        Fourier.Forward(_complexInput, FourierOptions.Matlab);

        // Return only the first half (N/2 + 1)
        Array.Copy(_complexInput, result, FftHalfSize);
    }

    private float[] PerformIrfft(Complex[] input)
    {
        // Reconstruct the full spectrum for IFFT
        Array.Clear(_fullSpectrum, 0, FftSize);
        Array.Copy(input, _fullSpectrum, FftHalfSize);

        // Fill the second half with complex conjugates (for real signal)
        for (int i = 1; i < FftHalfSize - 1; i++)
        {
            _fullSpectrum[FftSize - i] = Complex.Conjugate(input[i]);
        }

        Fourier.Inverse(_fullSpectrum, FourierOptions.Matlab);

        // Return the real part of the result（复用 _estimatedBlockTime）
        for (int i = 0; i < FftSize; i++)
        {
            _estimatedBlockTime[i] = (float)_fullSpectrum[i].Real;
        }

        return _estimatedBlockTime;
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
    public bool IsWarmingUp => _outputDelayFrames < PaddingFrames;
}