import React from "react";

type KeyTuple = [number, "white" | "black"];

interface MidiProps {
  row: number
  boardPos: Record<number, KeyTuple>;
  top: number; 
}

const Midi: React.FC<MidiProps> = ({ row, boardPos, top }) => {
  return (
    <div
      className="absolute z-50 left-[400px] w-[1040px] h-[42px]"
      style={{ top: `${top}px` }} 
    >
      {Object.entries(boardPos).map(([pos, [noteNum, color]]) => {
        const isWhite = color === "white";
        return (
          <div
            key={noteNum}
            className="flex justify-center items-end bg-transparent"
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
