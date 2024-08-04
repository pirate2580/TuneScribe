// src/components/MidiVisualizer.js
import React, { useState } from "react";
import { Midi } from "@tonejs/midi";
import ABCJS from "abcjs";

const MidiVisualizer = () => {
  const [file, setFile] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleVisualization = () => {
    if (!file) {
      alert("Please select a MIDI file to visualize");
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      const midiData = new Uint8Array(event.target.result);
      const midi = new Midi(midiData);
      const notes = midi.tracks.flatMap((track) => track.notes);

      // Convert notes to ABC notation for visualization
      const abcString = notes
        .map((note) => {
          return `${note.name}${note.octave}`;
        })
        .join(" ");

      // Render the notes using ABCJS
      ABCJS.renderAbc("notation", abcString);
    };
    reader.readAsArrayBuffer(file);
  };

  return (
    <div>
      <h1>Visualize MIDI Files</h1>
      <input type="file" accept=".midi" onChange={handleFileChange} />
      <button onClick={handleVisualization}>Visualize MIDI</button>
      <div id="notation"></div>
    </div>
  );
};

export default MidiVisualizer;