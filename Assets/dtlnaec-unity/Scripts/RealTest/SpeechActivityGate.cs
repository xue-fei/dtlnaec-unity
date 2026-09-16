using System;
using UnityEngine;

/// <summary>
/// 延迟更新门控：远端 VAD + 双讲检测（借鉴 WebRTC AEC3 的延迟更新门控思想）。
///
/// 背景：
///   GCC-PHAT 靠「近端 mic 里的回声」和「远端 loopback 参考」的相关性来测延迟。
///   只有满足两个条件时，测出的延迟才可靠：
///     1. 远端（数字人）确实在说话 —— 否则 loopback 是静音，GCC-PHAT 在噪声里瞎猜；
///     2. 近端（用户）没有同时说话 —— 否则 mic 里混入近端语音，相关峰被污染（双讲）。
///
/// 设计（参考 TenVad 的滞回 VAD）：
///   - 能量自适应阈值：噪声底（noise floor）用「历史能量最小值」跟踪，语音阈值 =
///     底噪 + 固定 dB 余量，而不是写死绝对阈值。这样能适应不同麦克风增益/环境噪声。
///   - 滞回机制：语音段用 MinSpeechDuration（防短促噪声误触发）、
///     静音段用 MinSilenceDuration（防字间停顿过早判为静音）；语音中的短暂停顿
///     用「缓慢衰减」而非「立即归零」容忍。
///   - 双讲检测：近端 mic 和远端 loopback 各自独立做 VAD，两者同时判为"说话"
///     即为双讲（此时 mic 里混入近端语音，相关峰被污染）。
///
/// 输出：
///   AllowDelayUpdate —— 只有「远端有语音 且 无双讲」时才为 true，调用方据此
///   决定是否用本次 GCC-PHAT 结果更新延迟。
///
/// 纯 C# + UnityEngine（Mathf），不依赖 sherpa-onnx，可独立单测。
/// </summary>
public class SpeechActivityGate
{
    // ── 远端 VAD 参数（参考 TenVad，但用能量自适应阈值）───────────────────
    private const float MIN_SPEECH_DURATION_SEC = 0.25f;   // 最短语音时长（防短噪声误触发）
    private const float MIN_SILENCE_DURATION_SEC = 0.5f;   // 最短静音时长（防字间停顿误判）
    private const float SPEECH_MARGIN_DB = 12f;            // 语音阈值 = 底噪能量比 + 12dB（能量域）
    private const float NOISE_FLOOR_ALPHA = 0.05f;         // 底噪更新系数（静音段缓慢收敛）
    private const float NOISE_UP_RATIO = 2.0f;             // 能量超过底噪 2× 视为语音（底噪冻结）

    // ── 双讲检测参数 ────────────────────────────────────────────────────────
    private const float DT_MIC_MARGIN_DB = 10f;            // 近端判定为"在说话"的阈值余量

    // ── 状态 ────────────────────────────────────────────────────────────────
    private bool _farEndSpeech = false;       // 远端当前是否在说话（VAD 状态）
    private float _speechTimer = 0f;          // 连续语音计时
    private float _silenceTimer = 0f;         // 连续静音计时

    private bool _nearEndActive = false;      // 近端当前是否在说话
    private float _nearSpeechTimer = 0f;
    private float _nearSilenceTimer = 0f;

    // 能量底噪（能量域，未初始化时 < 0）
    private float _farNoiseEnergy = -1f;      // 远端底噪能量（能量域，未初始化）
    private float _nearNoiseEnergy = -1f;     // 近端底噪能量（能量域，未初始化）

    // 底噪初始化窗口：前 N 帧取能量最小值作为底噪，快速到位且避开语音污染。
    private int _farInitCount = 0;
    private int _nearInitCount = 0;
    private const int NOISE_INIT_FRAMES = 32;   // 约 256ms @16kHz/128，足够覆盖启动静音

    // 采样率与帧长（用于把 duration 转成帧计数）
    private readonly int _sampleRate;
    private readonly int _frameSize;

    /// <summary>是否允许用当前测量值更新延迟（远端有语音 && 无双讲）。</summary>
    public bool AllowDelayUpdate => _farEndSpeech && !_nearEndActive;

    /// <summary>远端是否正在说话（供监控/日志）。</summary>
    public bool FarEndSpeech => _farEndSpeech;

    /// <summary>是否正在双讲（供监控/日志）。</summary>
    public bool DoubleTalk => _farEndSpeech && _nearEndActive;

    public SpeechActivityGate(int sampleRate = 16000, int frameSize = 128)
    {
        _sampleRate = sampleRate;
        _frameSize = frameSize;
    }

    /// <summary>
    /// 喂入一帧数据，更新 VAD/双讲状态。
    /// </summary>
    /// <param name="farFrame">远端 loopback 帧（数字人播放的声音）</param>
    /// <param name="nearFrame">近端 mic 帧（用户声音 + 回声）</param>
    public void ProcessFrame(float[] farFrame, float[] nearFrame)
    {
        float frameDur = (float)_frameSize / _sampleRate;

        float farEnergy = FrameEnergy(farFrame);
        float nearEnergy = FrameEnergy(nearFrame);

        // ── 更新底噪（单向门控 + 初始化窗口）────────────────────────────
        TrackNoiseFloor(farEnergy, ref _farNoiseEnergy, ref _farInitCount);
        TrackNoiseFloor(nearEnergy, ref _nearNoiseEnergy, ref _nearInitCount);

        // ── 远端 VAD（滞回状态机）──────────────────────────────────────────
        bool farVoice = IsVoice(farEnergy, _farNoiseEnergy, SPEECH_MARGIN_DB);
        UpdateHysteresis(
            farVoice, ref _farEndSpeech,
            ref _speechTimer, ref _silenceTimer, frameDur
        );

        // ── 近端 VAD（滞回状态机，用于双讲检测）────────────────────────────
        bool nearVoice = IsVoice(nearEnergy, _nearNoiseEnergy, DT_MIC_MARGIN_DB);
        UpdateHysteresis(
            nearVoice, ref _nearEndActive,
            ref _nearSpeechTimer, ref _nearSilenceTimer, frameDur
        );
    }

    /// <summary>
    /// 帧能量（线性域，均方值）。数值极小，仅用于阈值比较，无需归一化。
    /// </summary>
    private float FrameEnergy(float[] frame)
    {
        double sumSq = 0;
        for (int i = 0; i < frame.Length; i++)
            sumSq += (double)frame[i] * frame[i];
        return (float)(sumSq / frame.Length);
    }

    /// <summary>
    /// 跟踪底噪：噪声底 = 历史能量的「长期最小值」。
    ///
    /// 核心规则（关键：语音段底噪完全冻结，绝不被拉高）：
    ///   - 能量 ≤ 底噪 × NOISE_UP_RATIO：视为静音/噪声，底噪向当前能量缓慢收敛。
    ///   - 能量 > 底噪 × NOISE_UP_RATIO：视为语音，底噪【完全冻结】。
    ///
    /// 为什么不能「能量 ≥ 底噪就极慢向上回归」：长时间持续语音时，即使 DECAY 很小，
    /// 底噪也会逐步爬升到接近语音能量，最终把语音误判成静音（能量/底噪 比值跌破阈值）。
    ///
    /// 用 NOISE_UP_RATIO（如 2.0）留出「接近底噪的轻微波动」作为噪声，超过 2× 底噪
    /// 才判语音并冻结底噪。这样：
    ///   - 环境噪声缓慢升高（能量缓慢 > 旧底噪但 < 2× 旧底噪）→ 底噪跟随升高；
    ///   - 语音（能量远高于底噪）→ 底噪冻结，语音永远能正确识别。
    /// </summary>
    private void TrackNoiseFloor(float energy, ref float noiseEnergy, ref int initCount)
    {
        // 初始化窗口：前 NOISE_INIT_FRAMES 帧取能量最小值作为底噪。
        if (initCount < NOISE_INIT_FRAMES)
        {
            initCount++;
            if (noiseEnergy < 0f || energy < noiseEnergy)
            {
                noiseEnergy = energy;
            }
            return;
        }

        if (noiseEnergy < 0f)
        {
            noiseEnergy = energy;
        }

        // 仅当能量接近底噪（静音/噪声）时更新底噪；语音段（能量远高于底噪）冻结。
        if (energy <= noiseEnergy * NOISE_UP_RATIO)
        {
            noiseEnergy = noiseEnergy * (1f - NOISE_FLOOR_ALPHA) + energy * NOISE_FLOOR_ALPHA;
        }
        // else: 语音段，底噪冻结，不更新。
    }

    /// <summary>
    /// 判断能量是否构成"语音"（能量比底噪高指定 dB）。
    /// 能量比 dB 换算：10^(dB/10)（能量域，非幅值域）。
    /// </summary>
    private static bool IsVoice(float energy, float noiseEnergy, float marginDb)
    {
        if (noiseEnergy <= 0f) return false;
        float ratio = energy / noiseEnergy;
        float marginLinear = Mathf.Pow(10f, marginDb / 10f); // 能量比 dB → 线性比
        return ratio > marginLinear;
    }

    /// <summary>
    /// 滞回状态机：用最短语音/静音时长平滑状态切换，防止抖动。
    ///
    /// 关键修正：语音中的短暂停顿（字间/音节间，通常几十 ms）不能把
    /// speechTimer 立即归零，否则「连续语音 0.25s」永远凑不齐（说话总会有
    /// 停顿）。正确做法是：静音帧让 speechTimer 缓慢衰减（而非硬归零），
    /// 这样真实的连续语音段能累计过阈值，而长静音才会切回静音态。
    /// </summary>
    private void UpdateHysteresis(
        bool voiceNow, ref bool state,
        ref float speechTimer, ref float silenceTimer, float frameDur)
    {
        if (voiceNow)
        {
            silenceTimer = 0f;
            speechTimer += frameDur;
            // 连续语音超过最短时长才置为"说话"
            if (!state && speechTimer >= MIN_SPEECH_DURATION_SEC)
            {
                state = true;
            }
        }
        else
        {
            // 静音帧：speechTimer 缓慢衰减（速率 = 静音速率 ×2，约 2× 快速退出），
            // 而不是立即归零，从而容忍语音中的短暂停顿。
            speechTimer = Mathf.Max(0f, speechTimer - frameDur * 2f);
            silenceTimer += frameDur;
            // 连续静音超过最短时长才置为"静音"
            if (state && silenceTimer >= MIN_SILENCE_DURATION_SEC)
            {
                state = false;
            }
        }
    }

    /// <summary>重置状态（如设备切换、重新校准时调用）。</summary>
    public void Reset()
    {
        _farEndSpeech = false;
        _nearEndActive = false;
        _speechTimer = _silenceTimer = 0f;
        _nearSpeechTimer = _nearSilenceTimer = 0f;
        _farNoiseEnergy = -1f;
        _nearNoiseEnergy = -1f;
        _farInitCount = 0;
        _nearInitCount = 0;
    }
}
