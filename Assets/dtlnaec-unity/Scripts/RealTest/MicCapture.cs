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
///   Bug - ReadLoopbackFrameRaw 用 WritePos-PendingSamples 计算 readPos：
///         LoopbackCapture.WritePos 由音频线程均匀递增，
///         MicCapture 由 Update() 驱动帧率不均匀，一个 Update 帧内
///         while 循环消耗 N 个 BLOCK，但每次都取"当前最新"的 WritePos-128，
///         导致同一段 loopback 数据被重复读取（实测重复率约 49%），
///         写入 lpb_raw.wav 后播放音调偏低、断续不正常。
///   Fix  - 增加独立的 _lastLpbPos 顺序追踪 LoopbackCapture 读取位置，
///          每消耗一帧就前进 BLOCK_SHIFT，保证每帧 loopback 数据只读一次。
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

    // ✅ 修复核心：独立追踪 LoopbackCapture 环形缓冲的读取位置
    // 不再依赖 WritePos 实时值，每消耗一帧就顺序前进 BLOCK_SHIFT
    private int _lastLpbPos = 0;
    private bool _lpbPosInitialized = false;

    // GCC-PHAT 校准：每积累 CALIB_FRAMES 个 sample 估计一次延迟
    private const int CALIB_FRAMES = 4096;  // ~256ms @16kHz
    private const int RECALIB_INTERVAL = 500;   // 锁定后每 500 帧重新校准（~4s）

    private float[] _micAccum;
    private float[] _lpbAccum;
    private int _accumPos = 0;
    private int _recalibCount = 0;

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

                // ✅ 与 LoopbackCapture 写指针对齐：从当前写入位置开始读
                // 同时退后一个初始延迟量，让 AlignedLoopbackReader 有足够历史可 Pull
                int initDelay = 80 * SAMPLE_RATE / 1000;  // 80ms = 1280 samples
                _lastLpbPos = LoopbackCapture.WritePos - initDelay;
                _lpbPosInitialized = true;

                Debug.Log("[MicCapture] 启动完成");
                StartRecording();
            }));
        }));
    }

    void Update()
    {
        if (_init) ProcessAvailableFrames();
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
            // ── 安全检查：loopback 写入是否已追上 ──────────────────────────
            // 若 loopback 写指针落后于我们要读的位置，本次 while 循环暂停
            // （音频线程还没写够一帧，等下一个 Update）
            int lpbAvailable = LoopbackCapture.WritePos - _lastLpbPos;
            if (lpbAvailable < BLOCK_SHIFT) break;

            // 1. 读取麦克风帧
            float[] micFrame = new float[BLOCK_SHIFT];
            _micClip.GetData(micFrame, _lastMicPos);

            // 2. ✅ 顺序读取 loopback，_lastLpbPos 每帧递增 BLOCK_SHIFT，不重复不跳帧
            float[] rawLpb = ReadLoopbackFrameRaw();

            // 3. 推入延迟环形缓冲
            _lpbReader.Push(rawLpb);

            // 4. 取出延迟补偿后的对齐 loopback 帧
            float[] lpbAligned = _lpbReader.Pull(BLOCK_SHIFT);

            // 5. GCC-PHAT 自适应校准
            RunCalibration(micFrame, lpbAligned);

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
    /// ✅ 修复：使用独立的 _lastLpbPos 追踪读取位置，每帧前进 BLOCK_SHIFT。
    ///    原来用 WritePos - PendingSamples - BLOCK_SHIFT 计算 readPos，
    ///    在 Update 帧率不均匀时会将同一帧重复读取 N 次（实测重复率 ~49%），
    ///    导致 lpb_raw.wav 播放音调偏低、断续。
    /// </summary>
    float[] ReadLoopbackFrameRaw()
    {
        float[] frame = new float[BLOCK_SHIFT];
        int bufSize = LoopbackCapture.BufferSize;

        for (int i = 0; i < BLOCK_SHIFT; i++)
        {
            int idx = (_lastLpbPos + i) % bufSize;
            frame[i] = LoopbackCapture.LoopbackBuffer[idx];
        }

        // 顺序前进，下一帧从这里继续读，不依赖 WritePos 实时值
        _lastLpbPos += BLOCK_SHIFT;
        return frame;
    }

    /// <summary>
    /// 累积 mic / lpb 样本，每满 CALIB_FRAMES 触发一次 GCC-PHAT 估计。
    /// 锁定后每 RECALIB_INTERVAL 帧解锁一次以应对设备变化。
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
                maxLagSamples: SAMPLE_RATE * 300 / 1000
            );
            //Debug.LogWarning("lagDelta:"+ lagDelta);
            _lpbReader.UpdateDelay(_lpbReader.CurrentDelaySamples + lagDelta);
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