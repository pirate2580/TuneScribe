import React from 'react';
import './App.css';
import { MidiProvider } from "./Midi/MidiContext";
import MidiBoard from './Midi/Board';
import Card from './Card/Card';

function App() {
  return (
    <MidiProvider>
      <div className="App overflow-hidden h-screen w-screen flex relative bg-black ">
        <p className="absolute top-[-40px] left-[50%] translate-x-[-50%] text-white text-[150px] font-extrabold">
          MIDI.AI
        </p>

        <Card/>
        
        <MidiBoard className="absolute overflow-hidden top-[160px] left-[400px]" />
    </div>
    </MidiProvider>
  );
}

export default App;