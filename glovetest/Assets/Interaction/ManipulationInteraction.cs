using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Interaction that allows grabbing of objects and moving them around
/// </summary>
public class ManipulationInteraction : MonoBehaviour {
    
	void Start () {
        // Listen for when the prediction changes
        Glove.OnPredictionChanged += Glove_OnPredictionChanged;
	}

    // Triggers that are set when the prediction is changed (occurs in another thread)
    private bool GrabActivated;
    private bool GrabDeactivated;

    private void Glove_OnPredictionChanged(object sender, int prediction)
    {
        GrabActivated = prediction == 1;
        GrabDeactivated = prediction != 1;
    }

    public Vector3? GrabOffset = null;

    public int ControllerID = 0;

    public TensorflowOscGesture Glove;

    public Transform Root;

    // Update is called once per frame
    void Update () {
        var controlRoot = Root.transform.position;
        var controlForward = Root.transform.right;
        var controlTransform = Root.transform;
        var GrabHeld = Glove.GetCurrentPrediction() == 3;

        var ray = new Ray(controlRoot, controlForward);

        if (GrabActivated)
        {
            GrabActivated = false;
            StartGrab(ray, controlTransform);
        }
        if (GrabOffset.HasValue && (GrabDeactivated || !GrabHeld))
        {
            GrabDeactivated = false;
            this.GetComponent<SpringJoint>().connectedBody = null;
            GrabOffset = null;
        }

        if (GrabOffset.HasValue && GrabHeld) {
            var newPoint = controlTransform.TransformPoint(GrabOffset.Value);
            this.transform.position = newPoint;
        }

        // If initial grab was unsuccessful, ocassionaly try again
        if(GrabHeld && !GrabOffset.HasValue && Random.value < 0.1f)
        {
            StartGrab(ray, controlTransform);
        }
	}

    void StartGrab(Ray ray, Transform root)
    {
        RaycastHit hitinfo;
        if (Physics.SphereCast(ray, 0.2f, out hitinfo))
        {
            var collider = hitinfo.collider;

            var rigidbody = hitinfo.rigidbody;
            var character = collider.GetComponent<Character>();

            if (character != null)
            {
                rigidbody = character.RagdollRoot.GetComponent<Rigidbody>();
                character.SetRagdoll(true);
            }
            GrabOffset = root.InverseTransformPoint(hitinfo.point);

            this.transform.position = hitinfo.point;
            this.GetComponent<SpringJoint>().connectedBody = rigidbody;
        }
    }
}
