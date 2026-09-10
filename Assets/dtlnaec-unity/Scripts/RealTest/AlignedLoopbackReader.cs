// ════════════════════════════════════════════════════════════════════════════
//  AlignedLoopbackReader
//  带延迟补偿的 loopback 环形缓冲，GCC-PHAT 结果低通平滑后自动锁定
//
//  v3 优化（修复对齐失效 + 竞态）：
//   1. Pull 真正使用 _delaySamples 做持续补偿：读指针 = 写指针 - 当前延迟，
//      不再依赖一次性 ResetReadPos，锁定前后都能正确对齐。
//   2. 读指针边界保护：clamp 到 [0, 已写入量) 区间，防止读未写入数据或
//      读指针越过写指针（追赶竞态）。
//   3. Push 批量写入：环形分段 Array.Copy，替代逐样本 foreach。
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
    private long _writePos = 0;   // 累计写样本数（long 防回绕，取模进数组）
    private long _readPos = 0;    // 累计读样本数（同上）

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
    public int PendingSamples => (int)(_writePos - _readPos);

    public AlignedLoopbackReader(int maxDelayMs, int sampleRate, int initialDelayMs = 80)
    {
        _bufSize = maxDelayMs * 2 * sampleRate / 1000 + 4096;
        _buf = new float[_bufSize];
        _delaySamples = initialDelayMs * sampleRate / 1000;
        _lockedLastDelay = _delaySamples;
        // v3: 不再调用 ResetReadPos，读指针从 0 开始，由 Pull 内部对齐
    }

    /// <summary>每帧由 MicCapture 在处理前调用，将原始 loopback 写入缓冲。</summary>
    public void Push(float[] frame)
    {
        int n = frame.Length;
        if (n <= 0) return;

        // v3: 环形分段批量写入，避免逐样本 foreach
        int startIdx = (int)(_writePos % _bufSize);
        int firstPart = Math.Min(n, _bufSize - startIdx);
        Array.Copy(frame, 0, _buf, startIdx, firstPart);
        if (firstPart < n)
        {
            Array.Copy(frame, firstPart, _buf, 0, n - firstPart);
        }
        _writePos += n;
    }

    /// <summary>取出延迟补偿后与当前 mic 帧对齐的 loopback 数据。</summary>
    /// <remarks>
    /// v3：读指针按"当前延迟"持续退后对齐——每帧从
    ///   targetRead = _writePos - _delaySamples
    /// 处读取。这样锁定前后、以及延迟平滑值变化时都能持续正确对齐，
    /// 而不是像旧实现那样锁定后读指针就变成普通 FIFO 不再补偿。
    /// </remarks>
    public float[] Pull(int blockShift)
    {
        float[] frame = new float[blockShift];

        // v3：读指针对齐到"写指针 - 当前延迟"
        long targetRead = _writePos - _delaySamples;

        // 边界保护：targetRead 不能为负（写指针还没超过延迟量），
        // 也不能超过已写入量（此时会读到未写入的垃圾数据）。
        if (targetRead < 0)
        {
            // 尚未写入足够样本：输出静音并推进读指针，保持与写指针不脱节
            _readPos = _writePos;
            return frame;
        }

        // 若读指针落后太多（例如刚初始化/长时间停滞后恢复），
        // 直接跳到目标位置，避免一次性输出大量陈旧数据。
        if (_readPos < targetRead - _bufSize / 2 || _readPos > targetRead)
        {
            _readPos = targetRead;
        }

        int startIdx = (int)(_readPos % _bufSize);
        int firstPart = Math.Min(blockShift, _bufSize - startIdx);
        Array.Copy(_buf, startIdx, frame, 0, firstPart);
        if (firstPart < blockShift)
        {
            Array.Copy(_buf, 0, frame, firstPart, blockShift - firstPart);
        }

        _readPos += blockShift;

        // 防止读指针越过写指针（追赶竞态）：clamp 到不超前写指针
        if (_readPos > _writePos)
        {
            _readPos = _writePos;
        }

        return frame;
    }

    /// <summary>
    /// 接受 GCC-PHAT 测得的新延迟，进行自适应低通平滑；
    /// 连续 LOCK_FRAMES 次后锁定。
    ///
    /// v2: 自适应系数 + 单帧变化 clamp + 锁定后异常检测
    /// v3: 锁定不再触发 ResetReadPos，读指针由 Pull 按延迟自动对齐
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
}
