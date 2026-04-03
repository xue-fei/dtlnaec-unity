// ════════════════════════════════════════════════════════════════════════════
//  AlignedLoopbackReader
//  带延迟补偿的 loopback 环形缓冲，GCC-PHAT 结果低通平滑后自动锁定
// ════════════════════════════════════════════════════════════════════════════

using System;
using UnityEngine;

public class AlignedLoopbackReader
{
    // ── 环形缓冲 ──────────────────────────────────────────────────────────────
    private readonly float[] _buf;
    private readonly int _bufSize;
    private int _writePos = 0;
    private int _readPos = 0;

    // ── 延迟状态 ──────────────────────────────────────────────────────────────
    private int _delaySamples;
    private bool _locked = false;
    private int _lockCounter = 0;

    // 连续 LOCK_FRAMES 次估计稳定后锁定（低通滤波收敛判定）
    private const int LOCK_FRAMES = 50;
    private const float SMOOTH_ALPHA = 0.1f;  // 低通系数（新估计权重）

    /// <summary>当前使用的延迟（samples）</summary>
    public int CurrentDelaySamples => _delaySamples;
    public bool IsLocked => _locked;

    /// <summary>已推入但尚未被 Pull 消耗的 sample 数（供 raw 读取器定位）</summary>
    public int PendingSamples => _writePos - _readPos;

    public AlignedLoopbackReader(int maxDelayMs, int sampleRate, int initialDelayMs = 80)
    {
        // 缓冲区 = 最大延迟 × 2 + 一帧余量，防止读写相遇
        _bufSize = maxDelayMs * 2 * sampleRate / 1000 + 4096;
        _buf = new float[_bufSize];
        _delaySamples = initialDelayMs * sampleRate / 1000;
        ResetReadPos();
    }

    /// <summary>每帧由 MicCapture 在处理前调用，将原始 loopback 写入缓冲。</summary>
    public void Push(float[] frame)
    {
        foreach (float s in frame)
        {
            _buf[_writePos % _bufSize] = s;
            _writePos++;
        }
    }

    /// <summary>取出延迟补偿后与当前 mic 帧对齐的 loopback 数据。</summary>
    public float[] Pull(int blockShift)
    {
        float[] frame = new float[blockShift];
        for (int i = 0; i < blockShift; i++)
            frame[i] = _buf[(_readPos + i) % _bufSize];
        _readPos += blockShift;
        return frame;
    }

    /// <summary>
    /// 接受 GCC-PHAT 测得的新延迟，进行低通平滑；
    /// 连续 LOCK_FRAMES 次后锁定，并重置读指针到新延迟位置。
    /// </summary>
    public void UpdateDelay(int measuredDelaySamples)
    {
        if (_locked) return;

        // 低通滤波：抑制单帧噪声跳变
        _delaySamples = (int)(_delaySamples * (1f - SMOOTH_ALPHA)
                              + measuredDelaySamples * SMOOTH_ALPHA);

        // 边界保护
        _delaySamples = Mathf.Clamp(_delaySamples, 0, _bufSize / 2 - 1);

        _lockCounter++;
        if (_lockCounter >= LOCK_FRAMES)
        {
            _locked = true;
            ResetReadPos();
            Debug.Log($"[AlignedLoopbackReader] 延迟锁定：{_delaySamples} samples " +
                      $"({_delaySamples * 1000f / 16000:F1} ms)");
        }
    }

    /// <summary>解锁以重新校准（如音频设备切换、蓝牙断连重连时）。</summary>
    public void Unlock()
    {
        _locked = false;
        _lockCounter = 0;
    }

    // readPos = writePos - delaySamples，让读指针退后到延迟补偿后的位置
    private void ResetReadPos() =>
        _readPos = Math.Max(0, _writePos - _delaySamples);
}