import React, { useState } from "react";
import { saveAs } from "file-saver";
import lamejs from "lamejs";

const FileUpload = () => {
  const [file, setFile] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleConversion = () => {
    if (!file) {
      alert("Please select a file to convert");
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      const mp3Data = new Uint8Array(event.target.result);
      const wavData = mp3ToWav(mp3Data);
      saveAs(new Blob([wavData]), "converted.wav");
    };
    reader.readAsArrayBuffer(file);
  };

  const mp3ToWav = (mp3Data) => {
    const mp3Decoder = new lamejs.Mp3Decoder();
    const wav = mp3Decoder.decode(mp3Data);
    return wav;
  };

  return (
    <div>
      <h1>Upload and Convert Audio Files</h1>
      <input type="file" accept=".mp3,.wav" onChange={handleFileChange} />
      <button onClick={handleConversion}>Convert to WAV</button>
    </div>
  );
};

export default FileUpload;