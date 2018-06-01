using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;
using UnityStandardAssets.Characters.ThirdPerson;

public class Character : MonoBehaviour {

    bool IsRagdoll = false;

    public GameObject RagdollRoot;

    public static List<Character> Characters = new List<Character>();

    public List<Renderer> BodyRenderers = new List<Renderer>();

	// Use this for initialization
	void Awake () {
        SetRagdoll(false);
        var color = new Color(UnityEngine.Random.value, UnityEngine.Random.value, UnityEngine.Random.value);
        foreach(var renderer in BodyRenderers) {
            renderer.material.color = color;
        }
        Characters.Add(this);
	}
	
	// Update is called once per frame
	void Update () {
		
	}

    public void SetRagdoll(bool active) {
        foreach(var collider in this.transform.GetComponentsInChildren<Collider>()) {
            if (collider.transform == this.transform)
                collider.enabled = !active;
            else
                collider.enabled = active;
        }
        foreach (var rigidbody in this.transform.GetComponentsInChildren<Rigidbody>())
        {
            if (rigidbody.transform == this.transform)
                rigidbody.isKinematic = active;
            else
                rigidbody.isKinematic = !active;
        }
        if(active) {
            GetComponent<Animator>().enabled = false;
            GetComponent<ThirdPersonCharacter>().enabled = false;
            GetComponent<NavMeshAgent>().enabled = false;
        }
        if(active) {
            Characters.Remove(this);
        }
    }

	private void OnCollisionEnter(Collision collision)
	{
        if (collision.gameObject.name != "Ground" && !IsRagdoll && collision.rigidbody != null)
        {
            if(collision.rigidbody.velocity.magnitude > 4f || collision.transform.gameObject.name == "Fist")
                SetRagdoll(true);
        }
	}
}
