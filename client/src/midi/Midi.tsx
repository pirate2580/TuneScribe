import React from "react";

type KeyTuple = [number, "white" | "black"];

interface MidiProps {
  row: number
  boardPos: Record<number, KeyTuple>;
  top: number; 
  midiPressed: number[];
}

const Midi: React.FC<MidiProps> = ({ row, midiPressed, boardPos, top }) => {
  // console.log(`${row} ${midiPressed}`)
  return (
    <div
      className="absolute z-40 left-[400px] w-[1040px] h-[42px]"
      style={{ top: `${top}px` }} 
    >
      {Object.entries(boardPos).map(([pos, [noteNum, color]]) => {
        const isWhite = color === "white";
        return (
          <div
            key={noteNum}
            className={`flex justify-center items-end ${midiPressed[21 + noteNum] === 1? "bg-purple-600": "bg-transparent"}`}
            style={{
              position: "absolute",
              width: isWhite ? "20px" : "14px",
              height: "100%",
              left: `${pos}px`,
              zIndex: isWhite ? 1 : 2,
            }}
          />
        );
      })}
    </div>
  );
};

export default Midi;