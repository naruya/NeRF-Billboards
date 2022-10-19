using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class ControlCamera : MonoBehaviour
{
    public byte[] img = new byte[8];

    float dt = 0.1f;
    float dr = 1.0f;

    // Use this for initialization
    void Start()
    {
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        try
        {
            if (Input.GetKey("w"))
            {
                transform.Translate(0f, 0f, dt); // front
            }
            if (Input.GetKey("s"))
            {
                transform.Translate(0f, 0f, -dt); // back
            }
            if (Input.GetKey("d"))
            {
                transform.Translate(dt, 0f, 0f); // right
            }
            if (Input.GetKey("a"))
            {
                transform.Translate(-dt, 0f, 0f); // left
            }
            if (Input.GetKey("r"))
            {
                transform.Translate(0f, dt, 0f); // top
            }
            if (Input.GetKey("f"))
            {
                transform.Translate(0f, -dt, 0f); // botom
            }
            if (Input.GetKey("q"))
            {
                transform.Rotate(0f, dr, 0f); // yaw++
            }
            if (Input.GetKey("e"))
            {
                transform.Rotate(0f, -dr, 0f); // yaw--
            }
            if (Input.GetKey("z"))
            {
                transform.Rotate(dr, 0f, 0f); // pitch++
            }
            if (Input.GetKey("x"))
            {
                transform.Rotate(-dr, 0f, 0f); // pitch--
            }
            if (Input.GetKey("c"))
            {
                transform.Rotate(0f, 0f, dr); // roll++
            }
            if (Input.GetKey("v"))
            {
                transform.Rotate(0f, 0f, -dr); // roll--
            }
            // for usability
            transform.rotation = Quaternion.Euler(transform.eulerAngles.x, transform.eulerAngles.y, 0.0f);

        }
        finally
        {
        }
    }
}
