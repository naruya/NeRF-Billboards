using System;
using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using UnityEngine;
using System.IO;

public class ControlBillboard : MonoBehaviour
{

    static string data;
    static string resMsg1;
    static string resMsg2;
    System.Net.Sockets.NetworkStream ns;
    System.Net.Sockets.TcpClient tcp;

    Vector3 relativePos = new Vector3(0, 0, 0);
    Quaternion relativeQua = new Quaternion(0, 0, 0, 0);
    Camera mainCamera;

    public byte[] img = new byte[8];
    public Texture2D texture;
    public Texture2D texture_last;
    public int mode = 0;

    void Startfn()
    {
        int port = 36006;

        tcp = new System.Net.Sockets.TcpClient("localhost", port);
        Debug.Log("connected!" +
            ((System.Net.IPEndPoint)tcp.Client.RemoteEndPoint).Address + "," +
            ((System.Net.IPEndPoint)tcp.Client.RemoteEndPoint).Port + "," +
            ((System.Net.IPEndPoint)tcp.Client.LocalEndPoint).Address + "," +
            ((System.Net.IPEndPoint)tcp.Client.LocalEndPoint).Port);

        ns = tcp.GetStream();
    }

    // Start is called before the first frame update
    void Start()
    {
        //script = GameObject.Find("Main Camera").GetComponent<CameraControl>();
        mainCamera = GameObject.Find("Main Camera").GetComponent<Camera>();

        // https://stackoverflow.com/questions/49315959/what-causes-unity-memory-leaks
        texture = new Texture2D(1, 1);

        // Startfn();
    }

    void FixedUpdate()
    {
        if (tcp == null)
        {
           Startfn();
        }
        try
        {
            if (Input.GetKey("m"))
            {
                mode = (mode + 1) % 3;
            }
            this.transform.LookAt(mainCamera.transform);
            this.transform.Rotate(new Vector3(90, 0, 0));

            // position
            relativePos = mainCamera.transform.position - transform.position;
            // pose
            relativeQua = transform.rotation;

            float[] tmp = new float[7+1];
            tmp[0] = relativePos.z;
            tmp[1] = -relativePos.x;
            tmp[2] = relativePos.y;
            tmp[3] = -relativeQua.w;
            tmp[4] = relativeQua.z;
            tmp[5] = -relativeQua.x;
            tmp[6] = relativeQua.y;
            tmp[7] = (float)mode;

            string str = "";
            for (int i = 0; i < tmp.Length; i++)
            {
                if (i == (tmp.Length - 1))
                {
                    str = str + tmp[i].ToString();
                }
                else
                {
                    str = str + tmp[i].ToString() + ",";
                }
            }

            data = str.ToString();
            System.Text.Encoding enc = System.Text.Encoding.UTF8;

            //send
            byte[] sendBytes = enc.GetBytes(data);
            ns.Write(sendBytes, 0, sendBytes.Length);

            //res
            System.IO.MemoryStream ms = new System.IO.MemoryStream();
            byte[] resBytes = new byte[4194304];
            int resSize = 4194304;
            resSize = ns.Read(resBytes, 0, resBytes.Length);
            ms.Write(resBytes, 0, resSize);

            // get num
            resMsg1 = enc.GetString(ms.GetBuffer(), 0, 8);
            int num = int.Parse(resMsg1);

            // get XYZRPY
            resMsg2 = enc.GetString(ms.GetBuffer(), 8, num);

            // get img
            byte[] temp = ms.GetBuffer(); //, 8+num, (int)ms.Length-(8+num));
            img = new byte[(int)ms.Length - (8 + num)];
            Array.Copy(temp, 8 + num, img, 0, (int)ms.Length - (8 + num));

            ms.Close();

            Debug.Log(img.Length);

            if (img.Length != 4057)
            {
                Debug.Log(texture.width.ToString() + ", " + texture.height);
                texture.LoadImage(img);
                texture_last = texture;
            }

        }
        catch (FormatException e)
        {
            Debug.Log(e);
        }
        catch (OverflowException e)
        {
            Debug.Log(e);
        }
        catch (ArgumentOutOfRangeException e)
        {
            Debug.Log(e);
        }
        GetComponent<Renderer>().material.mainTexture = texture_last;
    }
}
