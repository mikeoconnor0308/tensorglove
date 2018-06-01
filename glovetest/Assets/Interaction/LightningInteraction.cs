using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Interaction that allows lightning to be fired from the hands
/// </summary>
public class LightningInteraction : MonoBehaviour {

    // Roots for the left and right hand
    public Transform LeftHand;
    public Transform RightHand;

    // The right hand gesture
    public TensorflowOscGesture RightHandPose;

    // List of bolts (one for each of the fingers)
    public List<LightningBolt> Bolts;

    // Start distance/angle is when it kicks in, end distance/angle is where it stops
    public float StartDistance = 0.3f;
    public float EndDistance = 0.5f;
    public float StartAngle = 10;
    public float EndAngle = 20;

	// Use this for initialization
	void Start () {
        foreach (var bolt in Bolts)
            bolt.gameObject.SetActive(false);
	}

    private bool CurrentlyActive = false;
	
	// Update is called once per frame
	void Update () {
        var angle = Vector3.Angle(-LeftHand.right, RightHand.right);
        var distance = (LeftHand.position - RightHand.position).magnitude;
        if(!CurrentlyActive && angle < StartAngle && distance < StartDistance && RightHandPose.GetCurrentPrediction() == 0)
        {
            foreach (var bolt in Bolts)
            {
                bolt.gameObject.active = true;
            }
            CurrentlyActive = true;
            GetComponent<AudioSource>().Play();
        }
        if (CurrentlyActive && (angle > EndAngle || distance > EndDistance || RightHandPose.GetCurrentPrediction() != 0))
        {
            foreach (var bolt in Bolts)
            {
                bolt.gameObject.active = false;
            }
            CurrentlyActive = false;
            GetComponent<AudioSource>().Stop();
        }
    }
}
