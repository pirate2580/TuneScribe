import React from 'react';
import logo from './logo.svg';
import './App.css';
import MidiBoard from './midi/MidiBoard'
// import Note from "./midi/Note"
function App() {
  return (
    <div className="App h-screen w-screen flex justify-center items-center">
        <MidiBoard/>
    </div>
  );
}

export default App;
