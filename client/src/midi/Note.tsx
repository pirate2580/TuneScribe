import React, {useState, useEffect} from "react";
import { pianoNoteMap } from './pianoNoteMapping'; 
import {useMidi} from "./MidiContext"
import * as Tone from 'tone';

// TODO: for performance optimization
const tailwindColors: { [key: string]: string } = {
  white: "bg-white",
  black: "bg-black",
};


interface NoteProps {
  noteNum: number,
  isPressed: boolean,
  color: string,
  style?: React.CSSProperties;
  
}


const Note: React.FC<NoteProps> = ({ noteNum, isPressed, color, style }) => {
  // const [pressed, setPressed] = useState(false);
  // const {currentIndex} = useMidi();
  const playNote = async () => {
    // Most browsers block audio until user interaction
    // so we need to 'resume' the AudioContext at least once.
    // setPressed(true);
    if (isPressed) {
      await Tone.start();

      // Create a simple synth and connect it to the master output
      const synth = new Tone.Synth().toDestination();
  
      // Trigger a note (C4) for 0.232 seconds
      // console.log(noteNum)
      synth.triggerAttackRelease(`${pianoNoteMap[noteNum]}`, "0.232");
    }
  };

  useEffect(() => {
    if (isPressed) {
      playNote();
    }
  }, [isPressed]);

  return (
    <div
      // Actually apply the style prop here
      style={style}
      // onMouseDown={() => console.log(noteNum)}
      // onMouseDown={() => console.log(pianoNoteMap)}
      className={`
        flex items-end justify-center border border-black z-10 
        ${isPressed ? "bg-amber-200" : tailwindColors[color]}
        ${color === "white"? "h-[100px] w-[20px]": "h-[70px] w-[14px]"}`}
    >
      <span className={`font-bold text-[6px] text-black ${color === "white"? "text-black": "text-white"}` }>
        {pianoNoteMap[noteNum]}
      </span>
    </div>
  );
};

// export default Note;


export default Note;