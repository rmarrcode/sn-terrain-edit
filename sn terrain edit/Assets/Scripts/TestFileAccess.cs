using System.IO;
using UnityEngine;
using AssetStudio;
using System.Runtime.InteropServices;

public class TestFileAccess : UnityEngine.MonoBehaviour
{
    // Import the setrlimit function from libc
    [DllImport("libc")]
    private static extern int setrlimit(int resource, ref rlimit rlim);

    [StructLayout(LayoutKind.Sequential)]
    private struct rlimit
    {
        public ulong rlim_cur;
        public ulong rlim_max;
    }

    private const int RLIMIT_NOFILE = 8; // Resource limit for number of open files

    void Start()
    {
        // Increase the file handle limit
        try
        {
            rlimit limit = new rlimit();
            limit.rlim_cur = 65536; // Soft limit
            limit.rlim_max = 65536; // Hard limit
            setrlimit(RLIMIT_NOFILE, ref limit);
            Debug.Log("Successfully increased file handle limit");
        }
        catch (System.Exception ex)
        {
            Debug.LogWarning($"Could not increase file handle limit: {ex.Message}");
        }

        string dataPath = "/Users/ryanmarr/Documents/Data";
        string resourcesPath = Path.Combine(dataPath, "resources.assets");
        string resourcesSplitPath = Path.Combine(dataPath, "resources.assets.resS");

        Debug.Log("Testing directory access...");
        try
        {
            // Test directory access
            string[] files = Directory.GetFiles(dataPath);
            Debug.Log($"Found {files.Length} files in directory");
            foreach (string file in files)
            {
                Debug.Log($"File: {file}");
            }

            // Test AssetStudio loading
            Debug.Log("Testing AssetStudio loading...");
            var assetsManager = new AssetsManager();
            
            // Try loading files one at a time
            if (File.Exists(resourcesPath))
            {
                Debug.Log($"Attempting to load {resourcesPath}");
                try
                {
                    assetsManager.LoadFiles(resourcesPath);
                    Debug.Log($"Successfully loaded {resourcesPath}");
                }
                catch (System.Exception ex)
                {
                    Debug.LogError($"Failed to load {resourcesPath}: {ex.Message}");
                }
            }
            else
            {
                Debug.LogError($"File not found: {resourcesPath}");
            }

            if (File.Exists(resourcesSplitPath))
            {
                Debug.Log($"Attempting to load {resourcesSplitPath}");
                try
                {
                    assetsManager.LoadFiles(resourcesSplitPath);
                    Debug.Log($"Successfully loaded {resourcesSplitPath}");
                }
                catch (System.Exception ex)
                {
                    Debug.LogError($"Failed to load {resourcesSplitPath}: {ex.Message}");
                }
            }
            else
            {
                Debug.LogError($"File not found: {resourcesSplitPath}");
            }

            if (assetsManager.assetsFileList.Count > 0)
            {
                Debug.Log($"Successfully loaded {assetsManager.assetsFileList.Count} asset files");
                foreach (var assetsFile in assetsManager.assetsFileList)
                {
                    Debug.Log($"Loaded asset file: {assetsFile.fileName}");
                }
            }
            else
            {
                Debug.LogError("No asset files were loaded");
            }
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"Failed to access directory or load assets: {ex.Message}\nStack trace: {ex.StackTrace}");
        }
    }
} 