// ════════════════════════════════════════════════════════════════════════════
//  AlignedLoopbackReader
//  带延迟补偿的 loopback 环形缓冲，GCC-PHAT 结果低通平滑后自动锁定
//
//  v2 优化：
//   1. 自适应平滑系数（早期大权重快速收敛，后期小权重稳定）
//   2. 单帧最大变化量 clamp，防止异常估计造成跳变
//   3. 锁定后延迟变化趋势追踪，异常时自动解锁
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

    // v2: 自适应平滑参数
    private const int LOCK_FRAMES = 50;
    private const float ALPHA_INITIAL = 0.3f;     // 初始权重（快速收敛）
    private const float ALPHA_LOCKED = 0.05f;     // 锁定后权重（稳定）
    private const int MAX_STEP_SAMPLES = 160;     // 单帧最大变化 ±10ms @16kHz

    // v2: 锁定后异常检测
    private int _lockedStableCount = 0;
    private int _lockedLastDelay = 0;
    private const int LOCKED_CHECK_INTERVAL = 100;  // 锁定后每 100 帧检查一次稳定性
    private const int LOCKED_MAX_DRIFT = 320;       // 允许最大漂移 ±20ms @16kHz

    /// <summary>当前使用的延迟（samples）</summary>
    public int CurrentDelaySamples => _delaySamples;
    public bool IsLocked => _locked;

    /// <summary>已推入但尚未被 Pull 消耗的 sample 数（供 raw 读取器定位）</summary>
    public int PendingSamples => _writePos - _readPos;

    public AlignedLoopbackReader(int maxDelayMs, int sampleRate, int initialDelayMs = 80)
    {
        _bufSize = maxDelayMs * 2 * sampleRate / 1000 + 4096;
        _buf = new float[_bufSize];
        _delaySamples = initialDelayMs * sampleRate / 1000;
        _lockedLastDelay = _delaySamples;
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
    /// 接受 GCC-PHAT 测得的新延迟，进行自适应低通平滑；
    /// 连续 LOCK_FRAMES 次后锁定，并重置读指针到新延迟位置。
    /// 
    /// v2: 自适应系数 + 单帧变化 clamp + 锁定后异常检测
    /// </summary>
    public void UpdateDelay(int measuredDelaySamples)
    {
        if (_locked)
        {
            // v2: 锁定后周期性检查延迟是否发生异常漂移
            _lockedStableCount++;
            if (_lockedStableCount >= LOCKED_CHECK_INTERVAL)
            {
                _lockedStableCount = 0;
                int drift = Math.Abs(measuredDelaySamples - _lockedLastDelay);
                if (drift > LOCKED_MAX_DRIFT)
                {
                    Debug.LogWarning($"[AlignedLoopbackReader] 延迟漂移过大 ({drift} samples)，解锁重新校准");
                    Unlock();
                    return;
                }
                _lockedLastDelay = measuredDelaySamples;
            }
            return;
        }

        // v2: 单帧变化 clamp，防止异常估计造成跳变
        int delta = measuredDelaySamples - _delaySamples;
        if (Math.Abs(delta) > MAX_STEP_SAMPLES)
        {
            measuredDelaySamples = _delaySamples + Math.Sign(delta) * MAX_STEP_SAMPLES;
        }

        // v2: 自适应平滑系数
        // 早期使用大权重快速收敛，接近锁定时减小权重提高稳定性
        float alpha = _lockCounter > LOCK_FRAMES / 2 ? ALPHA_LOCKED : ALPHA_INITIAL;

        // 低通滤波
        _delaySamples = (int)(_delaySamples * (1f - alpha)
                              + measuredDelaySamples * alpha);

        // 边界保护
        _delaySamples = Mathf.Clamp(_delaySamples, 0, _bufSize / 2 - 1);

        _lockCounter++;
        if (_lockCounter >= LOCK_FRAMES)
        {
            _locked = true;
            _lockedLastDelay = _delaySamples;
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
        _lockedStableCount = 0;
    }

    // readPos = writePos - delaySamples，让读指针退后到延迟补偿后的位置
    private void ResetReadPos() =>
        _readPos = Math.Max(0, _writePos - _delaySamples);
}
