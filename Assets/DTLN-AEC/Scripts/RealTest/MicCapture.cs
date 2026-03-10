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
///   3. 运行后按 R 键开始录制，再按 R 键停止并保存 WAV
///   4. 到 Application.persistentDataPath 目录对比三路音频：
///      - mic_raw.wav    原始麦克风（未处理）
///      - lpb_raw.wav    loopback 参考信号（播放出去的声音）
///      - aec_out.wav    AEC 处理后的干净语音
/// </summary>
public class MicCapture : MonoBehaviour
{
    // ── AEC 处理器 ──────────────────────────────────────────────────────────
    private RuntimeProcessor _aec;
    bool init = false;

    string localPath1;
    string localPath2;
    string sbPath1;
    string sbPath2;

    // ── 录音相关 ─────────────────────────────────────────────────────────────
    private AudioClip _micClip;
    private int _lastMicPos = 0;

    // BlockShift 必须与 DtlnaecProcessor2.BlockShift 一致（128 samples）
    private const int SAMPLE_RATE = 16000;
    private const int BLOCK_SHIFT = 128;   // ← 修正：原代码错误地用了 512

    // ── 验证录制 ─────────────────────────────────────────────────────────────
    private bool _isRecording = false;
    private WavWriter _writerMic;
    private WavWriter _writerLpb;
    private WavWriter _writerAec;

    // 上次读取 loopback 的绝对位置（与 LoopbackCapture.WritePos 对齐）
    private int _lastLpbPos = 0;

    // ── 生命周期 ──────────────────────────────────────────────────────────────

    void Start()
    {
        Init();
    }

    void Init()
    {
        localPath1 = Application.streamingAssetsPath + "/dtln_aec_128_1.onnx";
        localPath2 = Application.streamingAssetsPath + "/dtln_aec_128_2.onnx";
        sbPath1 = Application.persistentDataPath + "/dtln_aec_128_1.onnx";
        sbPath2 = Application.persistentDataPath + "/dtln_aec_128_2.onnx";

        StartCoroutine(CopyModel(localPath1, sbPath1, (value) =>
        {
            if (value)
            {
                StartCoroutine(CopyModel(localPath2, sbPath2, (value) =>
                {
                    if (value)
                    {
                        // 初始化 AEC
                        _aec = new RuntimeProcessor();
                        init = _aec.Initialize(
                            sbPath1,
                            sbPath2
                        );
                        if (!init)
                        {
                            Debug.LogError("[MicCapture] AEC 初始化失败");
                            enabled = false;
                            return;
                        }
                        else
                        {
                            Debug.Log("AEC 初始化完成");
                        }

                        // 启动麦克风，采样率必须与模型要求一致（16000Hz）
                        _micClip = Microphone.Start(null, true, 10, SAMPLE_RATE);
                        _lastLpbPos = LoopbackCapture.WritePos;

                        Debug.Log("[MicCapture] 启动完成，按鼠标左键开始/停止录制验证");
                    }
                }));
            }
        }));
    }

    void Update()
    {
        // R 键切换录制状态
        if (Input.GetMouseButtonDown(0))
        {
            if (!_isRecording)
            {
                StartRecording();
            }
            else
            {
                StopRecording();
            }
        }
        if (init)
        {
            ProcessAvailableFrames();
        }
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
                    // 写入文件
                    File.WriteAllBytes(destPath, www.downloadHandler.data);
                    Debug.Log($"复制成功：{destPath}");
                    if (action != null)
                    {
                        action(true);
                    }
                }
                catch (Exception e)
                {
                    Debug.LogError($"写入失败：{e.Message}");
                    if (action != null)
                    {
                        action(false);
                    }
                }
            }
            else
            {
                Debug.LogError($"读取失败：{www.error}");
                if (action != null)
                {
                    action(false);
                }
            }
        }
    }

    void OnDestroy()
    {
        if (_isRecording) StopRecording();
        Microphone.End(null);
        _aec?.Dispose();
    }

    // ── 核心处理循环 ──────────────────────────────────────────────────────────

    /// <summary>
    /// 每帧检查麦克风缓冲区，按 BLOCK_SHIFT（128）为单位取帧送 AEC。
    /// </summary>
    void ProcessAvailableFrames()
    {
        int micPos = Microphone.GetPosition(null);

        // 麦克风缓冲区环绕处理
        if (micPos < _lastMicPos)
            _lastMicPos = 0;

        int available = micPos - _lastMicPos;

        // 一次性消耗所有可用帧，避免积压
        while (available >= BLOCK_SHIFT)
        {
            float[] micFrame = new float[BLOCK_SHIFT];
            _micClip.GetData(micFrame, _lastMicPos);

            // 取对应时刻的 loopback 帧
            float[] lpbFrame = ReadLoopbackFrame();

            // AEC 推理，返回去回声后的干净语音
            float[] aecOut = _aec.ProcessFrame(micFrame, lpbFrame);

            // 验证录制：写入三路数据
            if (_isRecording)
            {
                _writerMic.Write(micFrame);
                _writerLpb.Write(lpbFrame);
                _writerAec.Write(aecOut);
            }

            _lastMicPos += BLOCK_SHIFT;
            available -= BLOCK_SHIFT;
        }
    }

    /// <summary>
    /// 从 LoopbackCapture 的环形缓冲中读取 BLOCK_SHIFT 个样本。
    /// 使用绝对位置跟踪，避免重复读或跳帧。
    /// </summary>
    float[] ReadLoopbackFrame()
    {
        float[] frame = new float[BLOCK_SHIFT];
        int bufSize = LoopbackCapture.BufferSize;

        for (int i = 0; i < BLOCK_SHIFT; i++)
        {
            int idx = (_lastLpbPos + i) % bufSize;
            frame[i] = LoopbackCapture.LoopbackBuffer[idx];
        }

        _lastLpbPos += BLOCK_SHIFT;
        return frame;
    }

    // ── 录制控制 ──────────────────────────────────────────────────────────────

    void StartRecording()
    {
        _writerMic = new WavWriter(SAMPLE_RATE);
        _writerLpb = new WavWriter(SAMPLE_RATE);
        _writerAec = new WavWriter(SAMPLE_RATE);
        _isRecording = true;
        Debug.Log("[MicCapture] 开始录制验证，再按鼠标左键停止保存");
    }

    void StopRecording()
    {
        _isRecording = false;

        string dir = Application.dataPath;
        if (Application.platform == RuntimePlatform.Android)
        {
            dir = Application.persistentDataPath;
        }
        _writerMic.Save(Path.Combine(dir, "mic_raw.wav"));
        _writerLpb.Save(Path.Combine(dir, "lpb_raw.wav"));
        _writerAec.Save(Path.Combine(dir, "aec_out.wav"));

        Debug.Log($"[MicCapture] 录制完成，文件保存至：{dir}");
        Debug.Log("  mic_raw.wav  → 原始麦克风");
        Debug.Log("  lpb_raw.wav  → loopback 参考信号");
        Debug.Log("  aec_out.wav  → AEC 处理后输出");
    }
}