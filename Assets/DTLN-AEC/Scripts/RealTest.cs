using System.Collections.Generic;
using uMicrophoneWebGL;
using UnityEngine;

public class RealTest : MonoBehaviour
{
    DtlnaecProcessor2 dtlnaecProcessor;
    public MicrophoneWebGL microphoneWebGL;
    bool isPlay = false;
    List<float> mic = new List<float>();
    List<float> lpb = new List<float>();
    List<float> output = new List<float>();

    // Start is called before the first frame update
    void Start()
    {
        dtlnaecProcessor = new DtlnaecProcessor2();
        dtlnaecProcessor.Initialize(Application.streamingAssetsPath + "/dtln_aec_128_1.onnx",
            Application.streamingAssetsPath + "/dtln_aec_128_2.onnx");

        microphoneWebGL.dataEvent.AddListener(OnData);
        microphoneWebGL.Begin(128);

        isPlay = true;
    }

    // Update is called once per frame
    void Update()
    {

    }

    float[] temp = new float[128];
    void OnData(float[] data)
    {
        mic.AddRange(data);
        if (farQueue.Count >= 128)
        {
            for (int i = 0; i < temp.Length; i++)
            {
                temp[i] = farQueue.Dequeue();
            }
        }
        float[] processedFrame = dtlnaecProcessor.ProcessFrame(data, temp);
        output.AddRange(processedFrame);
    }

    Queue<float> farQueue = new Queue<float>();

    private void OnAudioFilterRead(float[] data, int channels)
    {
        if (isPlay)
        {
            lpb.AddRange(data);
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = data[i] * 0.25f;
                farQueue.Enqueue(data[i]);
            }
        }
    }

    private void OnDestroy()
    {
        if (dtlnaecProcessor != null)
        {
            float[] end = dtlnaecProcessor.Flush();
            output.AddRange(end);
            Util.SaveClip(1, 16000, output.ToArray(), Application.dataPath + "/output.wav");
            Util.SaveClip(1, 16000, mic.ToArray(), Application.dataPath + "/mic.wav");
            Util.SaveClip(1, 16000, lpb.ToArray(), Application.dataPath + "/lpb.wav");
        }
        isPlay = false;
    }
}