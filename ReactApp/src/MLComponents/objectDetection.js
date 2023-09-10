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
      const model = await tf.loadLayersModel('https://raw.githubusercontent.com/VarunCypherV/ObjectDetectionReactApp/main/model.json');
      console.log('Custom model loaded.');

      // Create a function for real-time object detection
      const detectObjects = async () => {
        const webcam = webcamRef.current;

        // Ensure the webcam is initialized
        if (!webcam || !webcam.video) {
          requestAnimationFrame(detectObjects);
          return;
        }

        const videoElement = webcam.video;

        // Check if the video element has valid dimensions
        if (videoElement.videoWidth === 0 || videoElement.videoHeight === 0) {
          requestAnimationFrame(detectObjects);
          return;
        }

        // Capture a frame from the webcam
        const imageSrc = webcam.getScreenshot();

        // Create a canvas for rendering object detection results
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        // Ensure the canvas is initialized
        if (!canvas) {
          requestAnimationFrame(detectObjects);
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
        const classLabels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'];

        // Find the index with the highest probability
        const maxIndex = predictions.indexOf(Math.max(...predictions));

        // Log the predicted class
        console.log('Predicted Class:', classLabels[maxIndex]);

        // Request the next animation frame for real-time processing
        requestAnimationFrame(detectObjects);
      };

      // Start real-time object detection
      detectObjects();
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
          display: "none", // Hide the canvas since we use it for processing only
        }}
      />
    </div>
  );
};

export default ObjectDetection;
