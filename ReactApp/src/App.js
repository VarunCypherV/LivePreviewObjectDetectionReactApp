import React from 'react';
import './App.css';
import ObjectDetection from './MLComponents/objectDetection'; // Import your ObjectDetection component

function App() {
  return (
    <div className="App">
      <ObjectDetection /> {/* Render your ObjectDetection component here */}
    </div>
  );
}

export default App;
