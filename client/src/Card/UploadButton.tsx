import React, { useRef, useState } from "react";
import { useMidi } from "../Midi/MidiContext";

const UploadButton: React.FC = () => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [isSubmitClicked, setIsSubmitClicked] = useState(false); // <-- separate state for submit button click
  const [isLoading, setIsLoading] = useState(false);
  const { setMidiArray } = useMidi();

  // Handle file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) {
      const selectedFile = e.target.files[0];
      if (
        selectedFile.type === "audio/wav" ||
        selectedFile.type === "audio/mp3"
      ) {
        setFile(selectedFile);
      } else {
        alert("Please upload a valid .wav or .mp3 file.");
      }
    }
  };

  // Handle file upload
  const handleUpload = async () => {
    // Trigger the green color flash only on the submit button
    setIsSubmitClicked(true);
    setTimeout(() => {
      setIsSubmitClicked(false);
    }, 200);

    if (!file) {
      alert("Please select a file to upload");
      return;
    }

    setIsLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      console.log("Received MIDI Data:", data);

      if (Array.isArray(data.midi_array)) {
        setMidiArray(data.midi_array); // Store in global context
        // 1) SUCCESS ALERT
        alert("File successfully uploaded!");
      } else {
        throw new Error("Invalid MIDI data format received");
      }
    } catch (error) {
      console.error("Error:", error);
      alert("Error during upload");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      {/* Hidden file input */}
      <input
        type="file"
        ref={fileInputRef}
        accept=".wav,.mp3"
        style={{ display: "none" }}
        onChange={handleFileChange}
      />

      {/* Button to open file picker -- stays blue, does NOT turn green on click */}
      <button
        onClick={() => fileInputRef.current?.click()}
        className="overflow-hidden h-[80px] text-[20px] font-extrabold rounded-md transition-colors duration-300 bg-blue-950"
      >
        {file ? `File selected: ${file.name}` : "Upload a new file here (.wav, mp3, etc)"}
      </button>

      {/* Submit button -- flashes green when clicked, shows "Loading..." when isLoading */}
      <button
        onClick={handleUpload}
        className={`overflow-hidden h-[80px] text-[20px] font-extrabold rounded-md transition-colors duration-300 
          ${isSubmitClicked ? "bg-green-500" : "bg-blue-950"}
        `}
      >
        {isLoading ? "Loading..." : "Submit File"}
      </button>
    </>
  );
};

export default UploadButton;
