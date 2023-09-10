import React, { useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import Webcam from 'react-webcam';

const ObjectDetection = () => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    const runObjectDetection = async () => {
      // Load your custom TensorFlow.js model
      console.log('Custom model going to load.');
      const model = await tf.loadLayersModel('https://raw.githubusercontent.com/VarunCypherV/LivePreviewObjectDetectionReactApp/main/model/waste/model.json');
      console.log('Custom model loaded.');
      //
      //https://raw.githubusercontent.com/VarunCypherV/ObjectDetectionReactApp/main/model.json
      // Create a function for real-time object detection
      const detectObjects = async () => {
        const webcam = webcamRef.current;

        // Ensure the webcam is initialized
        if (!webcam || !webcam.video) {
          return;
        }

        const videoElement = webcam.video;

        // Check if the video element has valid dimensions
        if (videoElement.videoWidth === 0 || videoElement.videoHeight === 0) {
          return;
        }

        // Capture a frame from the webcam
        const imageSrc = webcam.getScreenshot();

        // Create a canvas for rendering object detection results
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        // Ensure the canvas is initialized
        if (!canvas) {
          return;
        }

        // Resize the webcam frame to match the model's input size (256x256)
        const desiredWidth = 256;
        const desiredHeight = 256;
        canvas.width = desiredWidth;
        canvas.height = desiredHeight;
        ctx.drawImage(videoElement, 0, 0, desiredWidth, desiredHeight);

        // Convert the resized frame to a tensor and normalize pixel values
        const inputTensor = tf.browser.fromPixels(canvas).toFloat().div(255.0).expandDims(0);

        // Make predictions
        const predictions = await model.predict(inputTensor).data();

        // Define the class labels
        // const classLabels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'];
        const classLabels = ['organic','recycle'];
        // Find the index with the highest probability
        const maxIndex = predictions.indexOf(Math.max(...predictions));

        // Log the predicted class
        const predictedClass = classLabels[maxIndex];
        console.log('Predicted Class:', predictedClass);

        // Draw a red bounding box around the predicted object
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.strokeRect(0, 0, desiredWidth, desiredHeight);

        // Customize the text appearance
        ctx.fillStyle = 'red';
        ctx.font = '18px Arial';

        // Draw the predicted class label below the bounding box
        ctx.fillText(predictedClass, 10, desiredHeight - 10);
      };

      // Request frames for real-time processing every 100 ms
      const frameInterval = 100;
      setInterval(detectObjects, frameInterval);
    };

    runObjectDetection();
  }, []);

  return (
    <div>
      <Webcam
        ref={webcamRef}
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top:100,
          margin: "auto",
          textAlign: "center",
          zIndex: 9,
          height: 480, // Adjust to your desired video dimensions
          width: 640,  // Adjust to your desired video dimensions
        }}
      />
      <canvas
        ref={canvasRef}
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top:100,
          margin: "auto",
          textAlign: "center",
          zIndex: 9,
          height: 480, // Adjust to your desired video dimensions
          width: 640,  // Adjust to your desired video dimensions
          display: 'none', // Hide the canvas since we use it for processing only
        }}
      />
    </div>
  );
};

export default ObjectDetection;
