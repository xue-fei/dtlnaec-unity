using System;
using System.Threading;
using UnityEngine;

/// <summary>
/// 挂在播放音频的 GameObject 上，通过 OnAudioFilterRead 截取 loopback 信号。
/// 内部将 Unity 音频线程的采样率重采样到 16000Hz，供 AEC 使用。
///
/// 修复记录：
///   Bug1 - Lerp 方向反转：原 Lerp(mono, _lastSample, t) → 应为 Lerp(_lastSample, mono, t)
///          方向写反导致每个输出样本朝"过去"偏移，产生时间拉伸，听起来是慢速播放。
///   Bug2 - 插值参数 t 取值时机错误：原代码在 _resampleAccum -= _resampleRatio 之后
///          才计算 t，此时 accum 已是下一个输出点的余量，当前点插值位置计算错误。
///          修复：输出点的 t = 1 - (_resampleAccum / _resampleRatio)，语义为
///          "当前输出点落在 [_lastSample, mono] 区间的相对位置"。
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
    private float _resampleRatio;   // Unity采样率 / 16000，例如 48000/16000 = 3.0
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
            // _resampleAccum 累积"已走过的输入样本数"
            // 每当累积量 >= ratio，就输出一个降采样点
            _resampleAccum += 1f;
            while (_resampleAccum >= _resampleRatio)
            {
                _resampleAccum -= _resampleRatio;

                // ✅ 修复 Bug1 + Bug2：
                //   t = 当前输出点在 [_lastSample → mono] 区间的相对位置
                //   减完后 _resampleAccum 是"超出量"（余量），即输出点距离 mono 的距离
                //   所以输出点距离 _lastSample 的距离 = _resampleRatio - _resampleAccum
                //   t = (_resampleRatio - _resampleAccum) / _resampleRatio
                //     = 1f - (_resampleAccum / _resampleRatio)
                //   t=0 → 输出等于 _lastSample；t=1 → 输出等于 mono（时序正确）
                float t = 1f - (_resampleAccum / _resampleRatio);
                float sample = Mathf.Lerp(_lastSample, mono, t);  // ✅ 方向修正

                int pos = Interlocked.Increment(ref _writePos) - 1;
                LoopbackBuffer[pos % BufferSize] = sample;
            }

            _lastSample = mono;
        }
    }
}