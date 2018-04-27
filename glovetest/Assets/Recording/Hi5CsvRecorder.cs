using System;
using System.Collections.Generic;
using HI5;
using UnityEngine;
using UnityEngine.UI;

namespace Recording
{
	public class Hi5CsvRecorder : MonoBehaviour
	{

		private enum GestureType { None, Fist, Click, Point}
		
		private CsvFileWriter csvLeftWriter;
		private CsvFileWriter csvRightWriter;

		[SerializeField]
		private HI5_TransformInstance leftHand;
		[SerializeField]
		private HI5_TransformInstance rightHand;

		[SerializeField] private GestureType gesture; 
		
		private bool recordingRightHand;
		private bool recordingLeftHand;

		[SerializeField]
		private Text recordingLeftText; 
		[SerializeField]
		private Text recordingRightText;

		[SerializeField]
		private bool recordPositions = false;
		// Use this for initialization
		void Start () {
		
		}
	
		// Update is called once per frame
		void Update () {

			if (Input.GetKeyDown(KeyCode.L))
			{
				if (recordingLeftHand)
				{
					csvLeftWriter.Dispose();
					recordingLeftHand = false;
					recordingLeftText.text = "Left Hand: Not Recording";
				}
				else
				{
					InitialiseWriter(true, gesture);
					recordingLeftHand = true;
					recordingLeftText.text = "Left Hand: Recording " + gesture.ToString();
				}
			}

			if (Input.GetKeyDown(KeyCode.R))
			{
				if (recordingRightHand)
				{
					csvRightWriter.Dispose();
					recordingRightHand = false;
					recordingRightText.text = "Right Hand: Not Recording";
				}
				else
				{
					InitialiseWriter(false, gesture);
					recordingRightHand = true;
					recordingRightText.text = "Right Hand: Recording " + gesture.ToString();
				}
			}

			if (recordingLeftHand)
			{
				WriteDataFrame(csvLeftWriter, leftHand, gesture);
			}

			if (recordingRightHand)
			{
				WriteDataFrame(csvRightWriter, rightHand, gesture);
			}
		}

		private void InitialiseWriter(bool left, GestureType gesture)
		{
			if (left)
			{
				
				csvLeftWriter = new CsvFileWriter("C:\\Users\\Mike.DESKTOP-CA70LTI\\Code\\iSci\\research\\nsb\\glovetest\\TrainingData\\" + "left_" + gesture + "_" +  System.DateTime.Now.ToString("yyyyMMddHHmmss"));
				Debug.Log("Writing to path: " + Application.dataPath);
				WriteHeader(csvLeftWriter, leftHand);
			}
			else
			{
				csvRightWriter = new CsvFileWriter("C:\\Users\\Mike.DESKTOP-CA70LTI\\Code\\iSci\\research\\nsb\\glovetest\\TrainingData\\" + "right_" + gesture + "_" +  System.DateTime.Now.ToString("yyyyMMddHHmmss"));
				Debug.Log("Writing to path: " + Application.dataPath);
				WriteHeader(csvRightWriter, rightHand);
			}
			
		}

		private void WriteHeader(CsvFileWriter writer, HI5_TransformInstance hand)
		{
			List<string> header = new List<string>();
			string[] coords = {"X", "Y", "Z", "W"};
			foreach (var t in hand.HandBones)
			{
				if (recordPositions)
				{
					for (int i = 0; i < 3; i++)
					{
						header.Add(string.Format("{0}_Pos_{1}", t.name, coords[i]));
					}
				}
				for (int i = 0; i < 4; i++)
				{
					header.Add(string.Format("{0}_Quat_{1}", t.name, coords[i]));
				}
			}
			header.Add("Gesture");
			writer.WriteRow(header);
			writer.Flush();
		}


		private void WriteDataFrame(CsvFileWriter writer, HI5_TransformInstance hand, GestureType gesture)
		{
			List<string> frame = new List<string>();
			foreach (var t in hand.HandBones)
			{
				if (recordPositions)
				{
					for (int i = 0; i < 3; i++)
					{
						frame.Add(t.localPosition[i].ToString());
					}
				}
				for (int i = 0; i < 4; i++)
				{
					frame.Add(t.localRotation[i].ToString());
				}
			}
			frame.Add(gesture.ToString());
			writer.WriteRow(frame);
			writer.Flush();
		}
	}
}
