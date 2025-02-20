import React, {useState} from "react";
import { useMidi } from "./MidiContext";
import Note from './Note'
import SeekBar from "./SeekBar";

interface MidiBoardProps {
  className?: string;
}


const blackKeyOffsetsPerOctave = [0.5, 2.5, 3.5, 5.5, 6.5];

const MidiBoard: React.FC<MidiBoardProps> = ({className}) => {

  const { midiArray } = useMidi();

  if (midiArray) {
    console.log("MIDI Array Shape:", [midiArray.length, midiArray[0]?.length]);
  }
  
  const whiteKeyPositions: number[] = [];
  const blackKeyPositions: number[] = [];

  for (let i = 0; i < 1040; i+= 20) {
    whiteKeyPositions.push(i);
  }

  // Generate positions for 7 full octaves worth of white keys (7 × 7 = 49)
  for (let octave = 0; octave < 7; octave++) {
    blackKeyOffsetsPerOctave.forEach((offset) => {
      const pos = (octave * 7 + offset) * 20; // each white key is 20px
      if (pos / 20 < 52) {
        blackKeyPositions.push(pos);
      }
    });
  }

  // After 49, the last 3 white keys are A (49), B (50), C (51).
  // There's only 1 black key between A and B.
  blackKeyPositions.push(49.5 * 20);

  // Combine positions into one array
  const allKeys: number[] = [];

  // push White keys
  for (let i = 0; i < 52; i++) {
    allKeys.push(i * 20);
  }
  // push Black keys
  blackKeyPositions.forEach((pos) => {
    allKeys.push(pos);
  });

  // Sort them by horizontal position (left to right)
  allKeys.sort((a, b) => a - b);

  // Assign note numbers in ascending order of positions
  // then store them in a lookup (pos → noteNum).
  const noteNumForPos: Record<number, number> = {};
  allKeys.forEach((key, index) => {
    noteNumForPos[key] = index; // index goes 0..87
  });

  return (
    <div>
      <div className={`${className || ''} relative w-[1040px] h-[650px] bg-gradient-to-t from-[#505cb9] to-[#140e52] flex items-end`}>
        {/* piano roll */}
        <div>
            <div className="flex mb-[40px]">
              {Array.from({ length: 52 }).map((_, i) => {
                // Left position is i * 20
                const pos = i * 20;
                const noteNum = noteNumForPos[pos]; // from our lookup
                return <Note key={i} noteNum={noteNum} color="white" />;
              })}
            </div>

            {blackKeyPositions.map((leftPos, i) => {
              const noteNum = noteNumForPos[leftPos];
              return (
                <Note
                  key={`black-${i}`}
                  noteNum={noteNum}
                  color="black"
                  style={{
                    position: "absolute",
                    bottom: "72px",
                    left: leftPos,
                    zIndex: 20,
                  }}
                />
              );
            })}
          </div>
        </div>
        {/* seek bar */}
        <SeekBar/>

    </div>
  );
}

export default MidiBoard;