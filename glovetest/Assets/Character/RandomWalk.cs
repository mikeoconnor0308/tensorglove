using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityStandardAssets.Characters.ThirdPerson;

[RequireComponent(typeof(UnityEngine.AI.NavMeshAgent))]
[RequireComponent(typeof(ThirdPersonCharacter))]
public class RandomWalk : MonoBehaviour
{
    public UnityEngine.AI.NavMeshAgent agent { get; private set; }             // the navmesh agent required for the path finding
    public ThirdPersonCharacter character { get; private set; } // the character we are controlling
    public Vector3? target = null;  // target to aim for

    public Bounds Area;

    private void Start()
    {
        // get the components on the object we need ( should not be null due to require component so no need to check )
        agent = GetComponentInChildren<UnityEngine.AI.NavMeshAgent>();
        character = GetComponent<ThirdPersonCharacter>();

        agent.updateRotation = false;
        agent.updatePosition = true;
    }


    private void Update()
    {
        if(!target.HasValue || Random.value < Time.deltaTime * 0.2f)
            SetTarget(new Vector3(Random.Range(Area.min.x, Area.max.x),
                                  Random.Range(Area.min.y, Area.max.y),
                                  Random.Range(Area.min.z, Area.max.z)));

        if (agent.enabled)
        {
            agent.SetDestination(target.Value);


            if (agent.remainingDistance > agent.stoppingDistance)
                character.Move(agent.desiredVelocity, false, false);
            else
            {
                character.Move(Vector3.zero, false, false);
                SetTarget(null);
            }
        }
    }


    public void SetTarget(Vector3? target)
    {
        this.target = target;
    }

	private void OnDrawGizmosSelected()
	{
        Gizmos.color = Color.blue;
        Gizmos.DrawWireCube(Area.center, Area.size);
	}
}
