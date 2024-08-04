import React, { useState, useEffect } from "react";
import { saveAs } from "file-saver";
import { Midi } from "@tonejs/midi";
import * as Tone from "tone";
import { Vex } from "vexflow";
import '../Styles/FileUpload.css';

const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [midiData, setMidiData] = useState(null);
  const [sheetMusic, setSheetMusic] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a WAV file to upload");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorMessage = await response.text();
        throw new Error(`Failed to convert file: ${response.status} - ${errorMessage}`);
      }

      const midiBlob = await response.blob();
      const midiArrayBuffer = await midiBlob.arrayBuffer();
      const midi = new Midi(midiArrayBuffer);
      setMidiData(new Uint8Array(midiArrayBuffer));
      setSheetMusic(midi);
    } catch (error) {
      console.error("Error:", error);
    }
  };

  const handleDownload = () => {
    if (midiData) {
      saveAs(new Blob([midiData], { type: "audio/midi" }), "converted.mid");
    } else {
      alert("No MIDI data available to download.");
    }
  };

  const visualizeMidi = async () => {
    if (!midiData) {
      alert("No MIDI data to visualize.");
      return;
    }

    const midi = new Midi(midiData);

    // Dispose of any previously created synth to avoid memory leaks and multiple synths playing
    if (window.currentSynth) {
      window.currentSynth.dispose();
    }

    // Create a new synth and store it in a global variable
    window.currentSynth = new Tone.PolySynth(Tone.Synth).toDestination();

    // Start the audio context
    await Tone.start();

    // Schedule each note to play
    midi.tracks.forEach((track) => {
      track.notes.forEach((note) => {
        window.currentSynth.triggerAttackRelease(note.name, note.duration, note.time);
      });
    });

    // Start the transport
    Tone.Transport.start();
  };

  const stopMidi = () => {
    Tone.Transport.stop();
    Tone.Transport.cancel(); // This clears the scheduled events in the transport

    // Dispose of the synth to ensure all notes are released
    if (window.currentSynth) {
      window.currentSynth.dispose();
      window.currentSynth = null;
    }
  };

  useEffect(() => {
    if (sheetMusic) {
      renderSheetMusic();
    }
  }, [sheetMusic]); // This effect runs when sheetMusic changes.

  const renderSheetMusic = () => {
    const VF = Vex.Flow;
    const div = document.getElementById("sheet-music");
    if (!div) return;
  
    div.innerHTML = ""; // Clear existing content
  
    // Create a renderer tied to the div
    const renderer = new VF.Renderer(div, VF.Renderer.Backends.SVG);
    const rendererWidth = 1000; // Set width as required to fit your page
    const rendererHeight = 500; // Set an initial height, can be adjusted based on the number of staves
    renderer.resize(rendererWidth, rendererHeight);
    const context = renderer.getContext();
  
    // Initial coordinates and width for the stave
    let x = 10, y = 40, width = 950;
  
    let staves = [];
    let allNotes = [];
  
    // Prepare notes and staves
    sheetMusic.tracks.forEach(track => {
      track.notes.forEach(note => {
        const noteName = note.name.toLowerCase().replace(/(\d+)/, '/$1');
        allNotes.push(new VF.StaveNote({
          clef: "treble",
          keys: [noteName],
          duration: "q"
        }));
      });
  
      // Distribute notes across multiple staves
      let notesPerStave = 16; // Adjust based on the average notes that fit within your width
      for (let i = 0; i < allNotes.length; i += notesPerStave) {
        let notes = allNotes.slice(i, i + notesPerStave);
        let stave = new VF.Stave(x, y, width);
        if (i === 0) {
          stave.addClef("treble");
        }
        staves.push({ stave, notes });
        y += 120; // Adjust vertical spacing as needed
      }
    });
  
    // Draw staves and notes
    staves.forEach(({ stave, notes }) => {
      stave.setContext(context).draw();
      const voice = new VF.Voice({ num_beats: notes.length, beat_value: 4 });
      voice.addTickables(notes);
      const formatter = new VF.Formatter().joinVoices([voice]).format([voice], width);
      voice.draw(context, stave);
    });
  
    // Adjust the renderer height based on the number of staves
    renderer.resize(rendererWidth, y + 50); // Ensure there's some margin
  };

  return (
    <div>
      <h1>Welcome to TuneScribe!</h1>
      <input type="file" accept=".wav" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload and Convert</button>
      <button onClick={handleDownload}>Download MIDI</button>
      <button onClick={visualizeMidi}>Visualize MIDI</button>
      <button onClick={stopMidi}>Stop MIDI</button>
      <div>
        <h2>Sheet Music</h2>
        <div id="sheet-music"></div>
      </div>
    </div>
  );
};

export default FileUpload;
