using UnityEngine;

public class Test : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        DtlnaecProcessor dtlnaecProcessor = new DtlnaecProcessor();
        dtlnaecProcessor.Initialize(Application.streamingAssetsPath + "/dtln_aec_128_1.onnx",
            Application.streamingAssetsPath + "/dtln_aec_128_2.onnx");
        float[] micData = Util.ReadWav(Application.dataPath + "/mic.wav");
        float[] lpbData = Util.ReadWav(Application.dataPath + "/lpb.wav");
        float[] data = dtlnaecProcessor.ProcessAudio(micData, lpbData);
        Util.SaveClip(1, 16000, data, Application.dataPath + "/output.wav");
    }

    // Update is called once per frame
    void Update()
    {

    }
}