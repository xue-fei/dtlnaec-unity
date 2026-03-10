using System;
using System.Threading;
using UnityEngine;

/// <summary>
/// 挂在播放音频的 GameObject 上，通过 OnAudioFilterRead 截取 loopback 信号。
/// 内部将 Unity 音频线程的采样率重采样到 16000Hz，供 AEC 使用。
/// </summary>
[RequireComponent(typeof(AudioSource))]
public class LoopbackCapture : MonoBehaviour
{
    public const int AecSampleRate = 16000;
    public const int BufferSeconds = 4;
    public const int BufferSize = AecSampleRate * BufferSeconds;

    // 线程安全的环形缓冲（音频线程写，主线程读）
    public static readonly float[] LoopbackBuffer = new float[BufferSize];
    private static int _writePos = 0;

    /// <summary>外部只读访问写指针（使用 Volatile 保证可见性）</summary>
    public static int WritePos => Volatile.Read(ref _writePos);

    // 重采样状态
    private int _unitySampleRate;
    private float _resampleRatio;       // Unity采样率 / 16000
    private float _resampleAccum = 0f;
    private float _lastSample = 0f;

    void Start()
    {
        _unitySampleRate = AudioSettings.outputSampleRate;
        _resampleRatio = (float)_unitySampleRate / AecSampleRate;

        Debug.Log($"[LoopbackCapture] Unity 采样率={_unitySampleRate}Hz，" +
                  $"重采样比={_resampleRatio:F3}，目标={AecSampleRate}Hz");
    }

    // 运行在 Unity 音频线程，passthrough 不修改 data
    void OnAudioFilterRead(float[] data, int channels)
    {
        int sampleCount = data.Length / channels;

        for (int i = 0; i < sampleCount; i++)
        {
            // 混成单声道
            float mono = 0f;
            for (int c = 0; c < channels; c++)
                mono += data[i * channels + c];
            mono /= channels;

            // 线性插值降采样到 16000Hz
            _resampleAccum += 1f;
            while (_resampleAccum >= _resampleRatio)
            {
                _resampleAccum -= _resampleRatio;

                float t = _resampleAccum / _resampleRatio;
                float sample = Mathf.Lerp(mono, _lastSample, t);

                int pos = Interlocked.Increment(ref _writePos) - 1;
                LoopbackBuffer[pos % BufferSize] = sample;
            }

            _lastSample = mono;
        }
    }
}