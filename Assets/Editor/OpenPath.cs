using UnityEditor;
using UnityEngine;

public class OpenPath : MonoBehaviour
{
    [MenuItem("Tools/打开沙盒路径")]
    static void OpenPersistentDataPath()
    {
        System.Diagnostics.Process.Start(@Application.persistentDataPath);
    }
}