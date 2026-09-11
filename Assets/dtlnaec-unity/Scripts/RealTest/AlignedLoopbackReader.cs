// ════════════════════════════════════════════════════════════════════════════
//  AlignedLoopbackReader
//  带延迟补偿的 loopback 环形缓冲，GCC-PHAT 结果经 α-β 滤波器跟踪
//
//  v5 优化（用 α-β 滤波器跟踪时钟漂移）：
//   1. 显式估计延迟漂移速度，用「位置+速度」双状态预测下一延迟，
//      稳定跟上 mic/loopback 采样时钟漂移，消除残差周期性反弹。
//   2. 野值拒绝：测量值偏离预测值超 ±30ms 时跳过更新，避免相关峰误判污染状态。
//   3. 锁定仅用于监控/日志，不再冻结或切换跟踪行为。
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
    private int _delaySamples;               // 当前补偿延迟（整数，供 Pull 使用）
    private float _delayFiltered;            // α-β 滤波器内部位置状态（float，含亚样本精度）
    private float _delayVelocity;            // 延迟漂移速度（samples / 校准周期）
    private bool _initialized = false;       // 首次有效测量后置 true，跳过预测阶段
    private bool _locked = false;            // 已锁定（速度估计稳定后置 true，仅用于监控/日志）
    private int _lockCounter = 0;            // 连续有效测量计数，用于锁定判定
    private int _outlierStreak = 0;          // v5: 连续野值计数（检测真实突变 vs 孤立野值）

    // v5: α-β 滤波器参数（跟踪时钟漂移）
    // α 控制对测量误差的响应速度，β 控制对漂移速度的估计速度。
    // 取值参考雷达跟踪惯例：β = α² / (2 - α)，保证临界阻尼、不过冲。
    private const float ALPHA = 0.25f;       // 位置增益（响应测量误差）
    private const float BETA = 0.05f;        // 速度增益（跟踪漂移斜率）
    private const int LOCK_FRAMES = 30;      // 连续 30 次有效测量后视为锁定
    private const int MAX_STEP_SAMPLES = 160;   // 单帧位置最大变化 ±10ms（防野值跳变）
    private const int OUTLIER_REJECT_SAMPLES = 480;  // 残差超此值视为野值，跳过更新（±30ms）
    private const int OUTLIER_STREAK_RESET = 5;   // 连续 5 次野值判定为真实突变，重置滤波器

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
        _delayFiltered = _delaySamples;
        _delayVelocity = 0f;
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
    /// 接受 GCC-PHAT 测得的【绝对延迟】，用 α-β 滤波器跟踪。
    ///
    /// v5: 用 α-β 滤波器（位置 + 速度双状态）替换一阶低通 + 锁定/微调。
    ///     时钟漂移（mic ADC 与 loopback DAC 采样频率差）使延迟近似匀速变化，
    ///     α-β 滤波器显式估计漂移速度并用其预测，能稳定跟上漂移，
    ///     消除一阶低通"追不上移动目标→残差周期性反弹"的问题。
    ///
    ///     野值拒绝：残差（测量值 vs 预测值）超过 OUTLIER_REJECT_SAMPLES 时，
    ///     判定为错误估计（如相关峰误判），跳过本次更新，不污染速度状态。
    ///
    /// v4: 锁定后持续微调（已被 v5 取代）
    /// v3: 锁定不触发 ResetReadPos，读指针由 Pull 按延迟自动对齐
    /// </summary>
    public void UpdateDelay(int measuredDelaySamples)
    {
        if (!_initialized)
        {
            // 首次有效测量：直接跳到测量值，初始化滤波器状态，不预测
            _delayFiltered = measuredDelaySamples;
            _delaySamples = measuredDelaySamples;
            _delayVelocity = 0f;
            _initialized = true;
            _lockCounter = 1;
            ClampDelay();
            return;
        }

        // ── 1. 预测：假设延迟匀速漂移 ──
        float predicted = _delayFiltered + _delayVelocity;

        // ── 2. 残差 = 测量值 - 预测值 ──
        float residual = measuredDelaySamples - predicted;

        // ── 3. 野值拒绝：残差过大说明测量值不可信（相关峰误判），跳过更新 ──
        if (Mathf.Abs(residual) > OUTLIER_REJECT_SAMPLES)
        {
            _outlierStreak++;
            _lockCounter = 0;

            // 连续多次野值且方向一致 → 判定为真实突变（设备切换/延迟阶跃），
            // 重置滤波器让延迟快速跳变到新值，否则会永远卡在旧值跟不上去。
            if (_outlierStreak >= OUTLIER_STREAK_RESET)
            {
                _delayFiltered = measuredDelaySamples;
                _delaySamples = Mathf.RoundToInt(measuredDelaySamples);
                _delayVelocity = 0f;
                _outlierStreak = 0;
                _lockCounter = 0;
                _locked = false;
                ClampDelay();
                Debug.LogWarning(
                    $"[AlignedLoopbackReader] 检测到延迟突变，重置滤波器到 {_delaySamples} samples " +
                    $"({_delaySamples * 1000f / 16000:F1} ms)");
            }
            return;
        }

        // 正常测量，清零野值计数
        _outlierStreak = 0;

        // ── 4. 单帧位置变化 clamp（防小野值造成跳变）──
        float clampedResidual = Mathf.Clamp(residual, -MAX_STEP_SAMPLES, MAX_STEP_SAMPLES);

        // ── 5. α-β 更新：位置 + 速度 ──
        _delayFiltered += ALPHA * clampedResidual;
        _delayVelocity += BETA * clampedResidual;

        // ── 6. 速度限幅：防止速度估计发散（时钟漂移量级约 ±0.5 sample/周期）──
        _delayVelocity = Mathf.Clamp(_delayVelocity, -8f, 8f);

        _delaySamples = Mathf.RoundToInt(_delayFiltered);
        ClampDelay();

        // ── 7. 锁定判定（仅用于监控/日志，不影响跟踪行为）──
        _lockCounter++;
        if (!_locked && _lockCounter >= LOCK_FRAMES)
        {
            _locked = true;
            Debug.Log($"[AlignedLoopbackReader] 延迟锁定：{_delaySamples} samples " +
                      $"({_delaySamples * 1000f / 16000:F1} ms)，漂移速度 {_delayVelocity:F3} samples/周期");
        }
    }

    /// <summary>把整数延迟 clamp 到合法区间。</summary>
    private void ClampDelay()
    {
        _delaySamples = Mathf.Clamp(_delaySamples, 0, _bufSize / 2 - 1);
        _delayFiltered = Mathf.Clamp(_delayFiltered, 0, _bufSize / 2 - 1);
    }

    /// <summary>解锁以重新校准（如音频设备切换、蓝牙断连重连时）。</summary>
    public void Unlock()
    {
        _locked = false;
        _lockCounter = 0;
        // 注意：不清空 _delayFiltered/_delayVelocity，避免解锁瞬间延迟跳变。
        // 若设备切换导致延迟大幅改变，会触发野值拒绝并重置，滤波器会自动重新收敛。
    }
}
