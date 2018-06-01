using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LightningBolt : MonoBehaviour
{

    // Use this for initialization
    void Start()
    {

    }

    public List<DigitalRuby.LightningBolt.LightningBoltScript> Bolts;

    public float Length = 5f;
    public float RandomJitter = 0.2f;

    public Transform Root;

    public int ChainLength = 3;

    public float Angle = 90f;

    private void Awake()
    {
        this.transform.parent = Root;
        this.transform.localPosition = Vector3.zero;
        this.transform.localRotation = Quaternion.AngleAxis(Angle, Vector3.up);
    }

    // Update is called once per frame
    void Update()
    {
        if (UnityEngine.Random.value < Time.deltaTime * 4f)
        {
            UpdateLightning();
        }
    }

    private void UpdateLightning()
    {
        Vector3 root = this.transform.position;
        Vector3 forward = this.transform.forward;

        List<Vector3> points = new List<Vector3>();
        points.Add(root);
        int totalChain = ChainLength;
        for (int i = 0; i < ChainLength; i++)
        {
            var dir = (forward + RandomJitter * UnityEngine.Random.insideUnitSphere).normalized;
            var len = UnityEngine.Random.Range(0.8f, 1.2f) * Length;

            RaycastHit hitinfo;
            if (Physics.Raycast(points[points.Count - 1], dir, out hitinfo, len))
            {
                OnPhysicsHit(hitinfo);
                points.Add(hitinfo.point);
                totalChain = i + 1;
                break;
            }
            else
            {
                points.Add(points[points.Count - 1] + dir * len);
            }
        }

        for (int i = 0; i < totalChain; i++)
        {
            Bolts[i].gameObject.active = true;
            Bolts[i].StartPosition = this.transform.InverseTransformPoint(points[i]);
            Bolts[i].EndPosition = this.transform.InverseTransformPoint(points[i + 1]);
        }
        for (int i = totalChain; i < ChainLength; i++)
        {
            Bolts[i].gameObject.active = false;
        }



    }

    void OnPhysicsHit(RaycastHit hit)
    {
        if(hit.transform.gameObject.GetComponent<Character>() != null)
        {
            hit.transform.gameObject.GetComponent<Character>().SetRagdoll(true);
        }
        if (hit.rigidbody != null)
        {
            hit.rigidbody.AddForce(UnityEngine.Random.insideUnitSphere * 40, ForceMode.Impulse);
        }
    }
}