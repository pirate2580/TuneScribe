import React from 'react';
import './App.css';
import MidiBoard from './Midi/MidiBoard'
import Card from './Card/Card';

function App() {
  return (
    <div className="App h-screen w-screen flex relative bg-black ">
      <p className="absolute top-[-40px] left-[50%] translate-x-[-50%] text-white text-[150px] font-extrabold">
        MIDI.AI
      </p>

      <Card/>
       
      <MidiBoard className="absolute top-[160px] left-[400px]" />
    </div>
  );
}

export default App;