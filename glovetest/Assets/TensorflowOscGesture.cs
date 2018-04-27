using System.Collections;
using System.Net;
using HI5;
using Rug.Osc;
using UnityEngine;

/// <summary>
/// Class for communicating with the tensorglove Osc server.
/// </summary>
public class TensorflowOscGesture : MonoBehaviour
{
    //ip address the tensorglove server is using.
    [SerializeField] private string ipAddress = "127.0.0.1";

    // port to transmit to server.
    [SerializeField] private int sendPort = 54321;

    // port to receive from server.
    [SerializeField] private int recvPort = 54322;

    private OscListener listener;

    private OscSender sender;

    // how long to wait before transmitting positions to tensorflow server.
    [Range(0f, 10f)] [SerializeField] private float waitTime = 5f;

    // the hand to use (only works with right hand at the moment).
    [SerializeField] private HI5_TransformInstance hand;

    // glove material, for color.
    [SerializeField] private Material gloveMaterial;

    // colors to assign to each gesture.
    [SerializeField] private Color[] gestureColors;

    private bool runningGestures = false;

    [SerializeField] private GameObject laser;

    private int prediction = 0;

    // Use this for initialization
    void Start()
    {
        listener = new OscListener(IPAddress.Parse(ipAddress), recvPort);
        listener.Connect();
        listener.Attach("/prediction", OnPrediction);


        sender = new OscSender(IPAddress.Parse(ipAddress), sendPort);
        sender.Connect();
    }

    private void OnPrediction(OscMessage message)
    {
        Debug.Log("Prediction Received " + message[0]);
        prediction = (int) message[0];
    }

    // Update is called once per frame
    void Update()
    {
        if (runningGestures == false)
        {
            runningGestures = true;
            StartCoroutine(GestureDetection());
        }

        gloveMaterial.color = gestureColors[prediction];
        // pointing class.
        if (prediction == 3)
        {
            laser.gameObject.SetActive(true);
        }
        else
        {
            laser.gameObject.SetActive(false);
        }
    }

    IEnumerator GestureDetection()
    {
        while (runningGestures)
        {
            SendCurrentPositions();
            yield return new WaitForSeconds(waitTime);
        }
    }

    private void SendCurrentPositions()
    {
        // send the quaternions of each finger. 
        string[] coords = {"X", "Y", "Z", "W"};
        object[] featureValues = new object[hand.HandBones.Length * 4];
        int boneIndex = 0;
        foreach (var t in hand.HandBones)
        {
            for (int i = 0; i < 4; i++)
            {
                featureValues[boneIndex * 4 + i] = (t.localRotation[i]);
            }

            boneIndex++;
        }

        //send the quaternion.
        OscMessage message = new OscMessage("/predict", featureValues);
        sender.Send(message);
    }
}