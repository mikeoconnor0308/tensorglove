using HI5;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// An interaction which allows collisions with the hands, with a fire effect and sound.
/// </summary>
public class PunchingInteraction : MonoBehaviour {
    // Base transforms for the hands, used to position the colliders
    public Transform LeftHandRoot;
    public Transform RightHandRoot;

    // References to the two colliders
    private SphereCollider RightHandCollider;
    private SphereCollider LeftHandCollider;

    // Gesture controller for the right hand
    public TensorflowOscGesture RightHandController;

    // Offset and radius for the sphere colliders
    public float ColliderRadius = 0.1f;
    public float ColliderOffset = 0.05f;

    // Is this interaction currently active
    public bool CurrentlyActive = false;

    // The particle systems that display fire on the hands
    public List<ParticleSystem> ParticleSystems;

    // Use this for initialization
    void Start () {
        // Generate spherical colliders
        GenerateCollider(out RightHandCollider, RightHandRoot, false);
        GenerateCollider(out LeftHandCollider, LeftHandRoot, true);

        // Initially disable the fire FX
        foreach (var system in ParticleSystems)
            system.gameObject.SetActive(false);
    }

    void GenerateCollider(out SphereCollider collider, Transform root, bool flip = false)
    {
        var sphere = new GameObject();
        sphere.transform.parent = root;
        sphere.transform.localRotation = Quaternion.identity;
        sphere.transform.localPosition = (flip ? -1f : 1f) * Vector3.right * ColliderOffset;
        collider = sphere.AddComponent<SphereCollider>();
        collider.radius = ColliderRadius;
        collider.enabled = false;
        var rigid = sphere.AddComponent<Rigidbody>();
        rigid.isKinematic = true;
        sphere.name = "Fist";
    }

    // Update is called once per frame
    void Update() {
        var current = RightHandController.GetCurrentPrediction() == 1;
        if (!CurrentlyActive && current)
            OnChangeActive(true);
        if (CurrentlyActive && !current)
            OnChangeActive(false);
    }

    // Handles what occurs then the active state is changed
    void OnChangeActive(bool active)
    {
        CurrentlyActive = active;
        RightHandCollider.enabled = active;
        LeftHandCollider.enabled = active;
        foreach (var system in ParticleSystems)
            system.gameObject.SetActive(active);
        if(active)
            GetComponent<AudioSource>().Play();
        else
            GetComponent<AudioSource>().Stop();
    }
}
