using System;
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
    
    private enum GestureType { None, Fist, Click, Point}
    
    //ip address the tensorglove server is using.
    [SerializeField] private string ipAddress = "127.0.0.1";

    // port to transmit to server.
    [SerializeField] private int sendPort = 54321;

    // port to receive from server.
    [SerializeField] private int recvPort = 54322;

    private OscListener listener;

    private OscSender sender;

    // how long to wait before transmitting positions to tensorflow server.
    [Range(0f, 10f)] [SerializeField] private float waitTime = 0.3f;

    [SerializeField] private HI5_TransformInstance leftHand;
    // the rightHand to use (only works with right rightHand at the moment).
    [SerializeField] private HI5_TransformInstance rightHand;

    // colors to assign to each gesture.
    [SerializeField] private Color[] gestureColors;

    private bool runningGestures = false;

    private GestureType leftPrediction = GestureType.None;

    private GestureType rightPrediction = GestureType.None;
    
    
    MaterialPropertyBlock block; 

    private Renderer leftRenderer;

    private Renderer rightRenderer; 
    
    // Use this for initialization
    void Start()
    {
        listener = new OscListener(IPAddress.Parse(ipAddress), recvPort);
        listener.Connect();
        listener.Attach("/prediction", OnPrediction);

        block = new MaterialPropertyBlock();

        sender = new OscSender(IPAddress.Parse(ipAddress), sendPort);
        sender.Connect();

        leftRenderer = leftHand.GetComponentInChildren<Renderer>();
        rightRenderer = rightHand.GetComponentInChildren<Renderer>();
    }

    private void OnPrediction(OscMessage message)
    {
        Debug.Log(string.Format("Prediction Received {0} {1}", message[0], message[1]));
        var hand = (string) message[0];
        if (hand == "left")
            leftPrediction = (GestureType) message[1];
        else
            rightPrediction = (GestureType) message[1];
    }

    // Update is called once per frame
    void Update()
    {
        if (runningGestures == false)
        {
            runningGestures = true;
            StartCoroutine(GestureDetection());
        }

        SetColor(leftRenderer, gestureColors[(int)leftPrediction]);
        SetColor(rightRenderer, gestureColors[(int)rightPrediction]);
    }

    private void SetColor(Renderer r, Color color)
    {
        r.GetPropertyBlock(block);
        block.SetColor("_Color", color);
        r.SetPropertyBlock(block);
    }

    IEnumerator GestureDetection()
    {
        while (runningGestures)
        {
            SendCurrentPositions();
            yield return new WaitForSeconds(waitTime);
        }
    }

    private void SendPositions(string handId, HI5_TransformInstance hand)
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
        OscMessage message = new OscMessage("/predict", handId, featureValues);
        sender.Send(message);
    }
    private void SendCurrentPositions()
    {
        SendPositions("left", leftHand);
        SendPositions("right", rightHand);
    }
}