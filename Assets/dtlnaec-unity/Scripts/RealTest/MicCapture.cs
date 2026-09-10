using System;
using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// 麦克风采集 + AEC 处理 + 三路 WAV 录制（用于验证效果）
///
/// 使用方式：
///   1. 将本脚本挂到任意 GameObject
///   2. 确保场景中有挂了 LoopbackCapture 的 AudioSource GameObject
///   3. 运行后自动开始录制，退出时自动保存 WAV
///   4. 到 Application.persistentDataPath 目录对比三路音频：
///      - mic_raw.wav    原始麦克风（未处理）
///      - lpb_raw.wav    对齐后的 loopback 参考信号
///      - aec_out.wav    AEC 处理后的干净语音
///
/// 修复记录：
///   Bug1 - ReadLoopbackFrameRaw 用 WritePos-PendingSamples 计算 readPos：
///          LoopbackCapture.WritePos 由音频线程均匀递增，
///          MicCapture 由 Update() 驱动帧率不均匀，一个 Update 帧内
///          while 循环消耗 N 个 BLOCK，但每次都取"当前最新"的 WritePos-128，
///          导致同一段 loopback 数据被重复读取（实测重复率约 49%），
///          写入 lpb_raw.wav 后播放音调偏低、断续不正常。
///   Fix1 - 增加独立的 _lastLpbPos 顺序追踪 LoopbackCapture 读取位置，
///          每消耗一帧就前进 BLOCK_SHIFT，保证每帧 loopback 数据只读一次。
///
///   Bug2 - 未播放音频时 LoopbackCapture.WritePos 接近 0，
///          _lastLpbPos = WritePos - initDelay 结果为负数。
///          C# 取模（%）对负数返回负值，用作数组索引时抛出
///          IndexOutOfRangeException。
///   Fix2 - 初始化时将 _lastLpbPos 夹紧为非负（Math.Max(0, ...)）；
///          ReadLoopbackFrameRaw 中使用防负数取模公式
///          ((x % n) + n) % n；
///          ProcessAvailableFrames 中对负数 _lastLpbPos 做快进修正，
///          使读指针尽快追上写指针后再正常消费。
/// </summary>
public class MicCapture : MonoBehaviour
{
    // ── AEC 处理器 ──────────────────────────────────────────────────────────
    private RuntimeProcessor _aec;
    private bool _init = false;

    private string _localPath1;
    private string _localPath2;
    private string _sbPath1;
    private string _sbPath2;

    // ── 录音相关 ─────────────────────────────────────────────────────────────
    private AudioClip _micClip;
    private int _lastMicPos = 0;

    private const int SAMPLE_RATE = 16000;
    private const int BLOCK_SHIFT = 128;

    // ── 验证录制 ─────────────────────────────────────────────────────────────
    private bool _isRecording = false;
    private WavWriter _writerMic;
    private WavWriter _writerLpb;
    private WavWriter _writerAec;

    // ── 对齐模块 ─────────────────────────────────────────────────────────────
    private AlignedLoopbackReader _lpbReader;

    // ✅ Fix1：独立追踪 LoopbackCapture 环形缓冲的读取位置
    // 不再依赖 WritePos 实时值，每消耗一帧就顺序前进 BLOCK_SHIFT
    private int _lastLpbPos = 0;
    private bool _lpbPosInitialized = false;

    // GCC-PHAT 校准：每积累 CALIB_FRAMES 个 sample 估计一次延迟
    private const int CALIB_FRAMES = 8192;  // v3: 512ms @16kHz，更大窗口减少静音段比例，延迟估计更稳定
    private const int RECALIB_INTERVAL = 500;   // 锁定后每 500 帧重新校准（~4s）
    private const float CONFIDENCE_THRESHOLD = 2.0f;  // 置信度阈值，低于此值拒绝更新

    private float[] _micAccum;
    private float[] _lpbAccum;
    private int _accumPos = 0;
    private int _recalibCount = 0;
    private int _rejectedCount = 0;  // v2: 被拒绝的校准次数统计

    // ── 对齐质量监控 ────────────────────────────────────────────────────────
    // 实时追踪 GCC-PHAT 置信度、估计延迟与残差，周期性输出质量报告。
    private const float MONITOR_INTERVAL_SEC = 2.0f;   // 每 2 秒打印一次质量快照
    private const float CONFIDENCE_GOOD = 3.0f;        // 置信度 > 3.0 视为"对齐良好"
    private float _monitorTimer = 0f;

    // 累积统计（用于报告周期内的均值）
    private int _monitorCalibCount = 0;      // 本周期内成功校准次数
    private float _monitorConfSum = 0f;      // 本周期置信度累加
    private float _monitorConfMin = float.MaxValue;
    private int _monitorRejected = 0;        // 本周期被拒绝次数
    private float _monitorResidualSum = 0f;  // 本周期残差累加（对齐后残余延迟）
    private float _monitorResidualMax = 0f;

    // 当前瞬时值（供外部/Inspector 查询）
    public float LastConfidence { get; private set; } = 0f;
    public float LastResidualMs { get; private set; } = 0f;
    public float AlignmentScore { get; private set; } = 0f;  // 0~100 对齐质量分

    // ── 生命周期 ──────────────────────────────────────────────────────────────

    void Start()
    {
        Init();
    }

    void Init()
    {
        _localPath1 = Application.streamingAssetsPath + "/dtln_aec_128_1.onnx";
        _localPath2 = Application.streamingAssetsPath + "/dtln_aec_128_2.onnx";
        _sbPath1 = Application.persistentDataPath + "/dtln_aec_128_1.onnx";
        _sbPath2 = Application.persistentDataPath + "/dtln_aec_128_2.onnx";

        StartCoroutine(CopyModel(_localPath1, _sbPath1, ok1 =>
        {
            if (!ok1) return;
            StartCoroutine(CopyModel(_localPath2, _sbPath2, ok2 =>
            {
                if (!ok2) return;

                // 初始化 AEC
                _aec = new RuntimeProcessor();
                _init = _aec.Initialize(_sbPath1, _sbPath2);
                if (!_init)
                {
                    Debug.LogError("[MicCapture] AEC 初始化失败");
                    enabled = false;
                    return;
                }
                Debug.Log("[MicCapture] AEC 初始化完成");

                // 初始化对齐读取器（最大延迟 300ms；初始假设 80ms）
                _lpbReader = new AlignedLoopbackReader(
                    maxDelayMs: 300,
                    sampleRate: SAMPLE_RATE,
                    initialDelayMs: 80
                );

                // 校准累积缓冲
                _micAccum = new float[CALIB_FRAMES];
                _lpbAccum = new float[CALIB_FRAMES];

                // 启动麦克风
                _micClip = Microphone.Start(null, true, 10, SAMPLE_RATE);

                // ✅ Fix2：与 LoopbackCapture 写指针对齐，退后初始延迟量。
                // 若 WritePos 尚小（未播放音频时），夹紧为 0，避免负数索引崩溃。
                int initDelay = 80 * SAMPLE_RATE / 1000;  // 80ms = 1280 samples
                _lastLpbPos = Math.Max(0, LoopbackCapture.WritePos - initDelay);
                _lpbPosInitialized = true;

                Debug.Log("[MicCapture] 启动完成");
                StartRecording();
            }));
        }));
    }

    void Update()
    {
        if (_init) ProcessAvailableFrames();
        UpdateAlignmentMonitor();
    }

    /// <summary>
    /// 对齐质量监控：周期性汇总 GCC-PHAT 置信度、残差等指标，
    /// 计算对齐质量分并输出报告，异常时告警。
    /// </summary>
    void UpdateAlignmentMonitor()
    {
        _monitorTimer += Time.deltaTime;
        if (_monitorTimer < MONITOR_INTERVAL_SEC) return;
        _monitorTimer = 0f;

        if (_monitorCalibCount == 0)
        {
            // 本周期没有成功校准：要么尚未锁定（预热），要么全被拒绝（远端静音）
            return;
        }

        float avgConf = _monitorConfSum / _monitorCalibCount;
        float avgResidualMs = (_monitorResidualSum / _monitorCalibCount) * 1000f / SAMPLE_RATE;

        // 质量评分：置信度贡献 60%，残差贡献 40%
        // confidence 映射：<2.0 → 0，>6.0 → 满分
        float confScore = Mathf.InverseLerp(2.0f, 6.0f, avgConf);
        // residual 映射：0ms → 满分，>30ms → 0 分
        float residualScore = 1f - Mathf.InverseLerp(0f, 30f, avgResidualMs);
        AlignmentScore = (confScore * 0.6f + residualScore * 0.4f) * 100f;

        string state = _lpbReader.IsLocked ? "锁定" : "校准中";
        string quality;
        if (AlignmentScore >= 80f) quality = "优秀";
        else if (AlignmentScore >= 60f) quality = "良好";
        else if (AlignmentScore >= 40f) quality = "一般";
        else quality = "差";

        Debug.Log(
            $"[对齐监控] 状态={state} | 质量分={AlignmentScore:F0}({quality}) | " +
            $"置信度均值={avgConf:F2}(min={_monitorConfMin:F2}) | " +
            $"残差均值={avgResidualMs:F2}ms(峰值={_monitorResidualMax * 1000f / SAMPLE_RATE:F2}ms) | " +
            $"成功校准={_monitorCalibCount} 拒绝={_monitorRejected}"
        );

        if (AlignmentScore < 40f)
        {
            Debug.LogWarning(
                "[对齐监控] ⚠️ 对齐质量差，可能原因：远端未播放声音(loopback静音)、" +
                "延迟估计漂移、或麦克风/loopback 时序错位。请检查 mic_raw/lpb_raw 波形是否重合。"
            );
        }

        // 重置周期累积
        _monitorCalibCount = 0;
        _monitorConfSum = 0f;
        _monitorConfMin = float.MaxValue;
        _monitorRejected = 0;
        _monitorResidualSum = 0f;
        _monitorResidualMax = 0f;
    }

    IEnumerator CopyModel(string sourcePath, string destPath, Action<bool> action = null)
    {
        using (UnityWebRequest www = UnityWebRequest.Get(sourcePath))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                try
                {
                    File.WriteAllBytes(destPath, www.downloadHandler.data);
                    Debug.Log($"[MicCapture] 复制成功：{destPath}");
                    action?.Invoke(true);
                }
                catch (Exception e)
                {
                    Debug.LogError($"[MicCapture] 写入失败：{e.Message}");
                    action?.Invoke(false);
                }
            }
            else
            {
                Debug.LogError($"[MicCapture] 读取失败：{www.error}");
                action?.Invoke(false);
            }
        }
    }

    void OnDestroy()
    {
        if (_isRecording) StopRecording();
        Microphone.End(null);
        _aec?.Dispose();
    }

    private void OnApplicationQuit()
    {
        StopRecording();
    }

    // ── 核心处理循环 ──────────────────────────────────────────────────────────

    void ProcessAvailableFrames()
    {
        if (!_lpbPosInitialized) return;

        int micPos = Microphone.GetPosition(null);

        // 麦克风缓冲区环绕处理
        if (micPos < _lastMicPos) _lastMicPos = 0;

        int available = micPos - _lastMicPos;

        while (available >= BLOCK_SHIFT)
        {
            // ── Fix2：_lastLpbPos 为负时（WritePos 起始很小导致的遗留负值），
            // 快进到 max(0, WritePos - initDelay) 后再做正常的可用量判断，
            // 避免负数传入 ReadLoopbackFrameRaw 的取模运算。
            if (_lastLpbPos < 0)
            {
                int initDelay = 80 * SAMPLE_RATE / 1000;
                _lastLpbPos = Math.Max(0, LoopbackCapture.WritePos - initDelay);
                Debug.LogWarning($"[MicCapture] _lastLpbPos 为负，已修正为 {_lastLpbPos}");
            }

            // ── 安全检查：loopback 写入是否已追上 ──────────────────────────
            // 若 loopback 写指针落后于我们要读的位置，本次 while 循环暂停
            // （音频线程还没写够一帧，等下一个 Update）
            int lpbAvailable = LoopbackCapture.WritePos - _lastLpbPos;
            if (lpbAvailable < BLOCK_SHIFT) break;

            // 1. 读取麦克风帧
            float[] micFrame = new float[BLOCK_SHIFT];
            _micClip.GetData(micFrame, _lastMicPos);

            // 2. ✅ Fix1：顺序读取 loopback，_lastLpbPos 每帧递增 BLOCK_SHIFT，不重复不跳帧
            float[] rawLpb = ReadLoopbackFrameRaw();

            // 3. 推入延迟环形缓冲
            _lpbReader.Push(rawLpb);

            // 4. 取出延迟补偿后的对齐 loopback 帧
            float[] lpbAligned = _lpbReader.Pull(BLOCK_SHIFT);

            // 5. GCC-PHAT 自适应校准
            //    v3：用「原始未补偿的 loopback」估计延迟，避免用已对齐数据
            //        校准自己形成反馈循环（对齐值会污染延迟估计）。
            RunCalibration(micFrame, rawLpb);

            // 6. AEC 推理
            float[] aecOut = _aec.ProcessFrame(micFrame, lpbAligned);

            // 7. 验证录制
            if (_isRecording)
            {
                _writerMic.Write(micFrame);
                _writerLpb.Write(lpbAligned);
                _writerAec.Write(aecOut);
            }

            _lastMicPos += BLOCK_SHIFT;
            available -= BLOCK_SHIFT;
        }
    }

    /// <summary>
    /// 从 LoopbackCapture 环形缓冲顺序读取 BLOCK_SHIFT 个样本。
    ///
    /// ✅ Fix1：使用独立的 _lastLpbPos 追踪读取位置，每帧前进 BLOCK_SHIFT。
    ///    原来用 WritePos - PendingSamples - BLOCK_SHIFT 计算 readPos，
    ///    在 Update 帧率不均匀时会将同一帧重复读取 N 次（实测重复率 ~49%），
    ///    导致 lpb_raw.wav 播放音调偏低、断续。
    ///
    /// ✅ Fix2：取模使用 ((x % n) + n) % n 防御负数索引。
    ///    C# 的 % 对负数操作数返回负值，直接用于数组索引会抛
    ///    IndexOutOfRangeException。
    /// </summary>
    float[] ReadLoopbackFrameRaw()
    {
        float[] frame = new float[BLOCK_SHIFT];
        int bufSize = LoopbackCapture.BufferSize;

        for (int i = 0; i < BLOCK_SHIFT; i++)
        {
            // ✅ Fix2：防负数取模，保证索引始终在 [0, bufSize) 范围内
            int raw = (_lastLpbPos + i) % bufSize;
            int idx = (raw + bufSize) % bufSize;
            frame[i] = LoopbackCapture.LoopbackBuffer[idx];
        }

        // 顺序前进，下一帧从这里继续读，不依赖 WritePos 实时值
        _lastLpbPos += BLOCK_SHIFT;
        return frame;
    }

    /// <summary>
    /// 累积 mic / lpb 样本，每满 CALIB_FRAMES 触发一次 GCC-PHAT 估计。
    /// 锁定后每 RECALIB_INTERVAL 帧解锁一次以应对设备变化。
    ///
    /// v2 优化：置信度校验 — 静音/噪声时拒绝校准，防止错误偏移
    /// </summary>
    void RunCalibration(float[] micFrame, float[] lpbFrame)
    {
        if (_lpbReader.IsLocked)
        {
            _recalibCount++;
            if (_recalibCount >= RECALIB_INTERVAL)
            {
                _lpbReader.Unlock();
                _recalibCount = 0;
                _accumPos = 0;
                Debug.Log("[MicCapture] 延迟重新校准中...");
            }
            return;
        }

        int copy = Math.Min(BLOCK_SHIFT, CALIB_FRAMES - _accumPos);
        Array.Copy(micFrame, 0, _micAccum, _accumPos, copy);
        Array.Copy(lpbFrame, 0, _lpbAccum, _accumPos, copy);
        _accumPos += copy;

        if (_accumPos >= CALIB_FRAMES)
        {
            int lagDelta = DelayEstimator.Estimate(
                _micAccum, _lpbAccum,
                maxLagSamples: SAMPLE_RATE * 300 / 1000,
                out float confidence
            );

            // ── 对齐监控采集：无论接受与否都记录置信度 ──
            LastConfidence = confidence;
            _monitorConfSum += confidence;
            if (confidence < _monitorConfMin) _monitorConfMin = confidence;

            // v2: 置信度校验 — 静音/噪声时拒绝校准
            if (confidence < CONFIDENCE_THRESHOLD)
            {
                _rejectedCount++;
                _monitorRejected++;
                if (_rejectedCount % 10 == 1)
                {
                    Debug.LogWarning($"[MicCapture] 校准置信度过低 ({confidence:F2} < {CONFIDENCE_THRESHOLD})，跳过更新");
                }
            }
            else
            {
                // v3 修复：lagDelta 是 GCC-PHAT 在「原始 loopback」与 mic 之间
                // 测出的【绝对延迟】，不是增量。旧代码误把绝对延迟当增量叠加
                // （CurrentDelaySamples + lagDelta），导致延迟值被反复累加、残差
                // 永远等于完整延迟无法收敛。
                int prevDelay = _lpbReader.CurrentDelaySamples;
                _lpbReader.UpdateDelay(lagDelta);

                // 残差 = 新旧延迟估计之差，对齐收敛后应趋近 0
                float residual = Mathf.Abs(_lpbReader.CurrentDelaySamples - prevDelay);
                _rejectedCount = 0;

                // ── 监控采集：记录残差与成功校准 ──
                LastResidualMs = residual * 1000f / SAMPLE_RATE;
                _monitorResidualSum += residual;
                if (residual > _monitorResidualMax) _monitorResidualMax = residual;
                _monitorCalibCount++;
            }

            _accumPos = 0;
        }
    }

    // ── 录制控制 ──────────────────────────────────────────────────────────────

    void StartRecording()
    {
        _writerMic = new WavWriter(SAMPLE_RATE);
        _writerLpb = new WavWriter(SAMPLE_RATE);
        _writerAec = new WavWriter(SAMPLE_RATE);
        _isRecording = true;
        Debug.Log("[MicCapture] 开始录制验证");
    }

    void StopRecording()
    {
        if (!_isRecording) return;
        _isRecording = false;

        string dir = (Application.platform == RuntimePlatform.Android)
            ? Application.persistentDataPath
            : Application.dataPath;

        _writerMic.Save(Path.Combine(dir, "mic_raw.wav"));
        _writerLpb.Save(Path.Combine(dir, "lpb_raw.wav"));
        _writerAec.Save(Path.Combine(dir, "aec_out.wav"));

        Debug.Log($"[MicCapture] 录制完成，文件保存至：{dir}");
        Debug.Log("  mic_raw.wav  → 原始麦克风");
        Debug.Log("  lpb_raw.wav  → 对齐后 loopback 参考信号");
        Debug.Log("  aec_out.wav  → AEC 处理后输出");
    }
}