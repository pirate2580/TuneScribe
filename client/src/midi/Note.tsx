import React, {useState} from "react";
import { pianoNoteMap } from './pianoNoteMapping'; 

// TODO: for performance optimization
const tailwindColors: { [key: string]: string } = {
  white: "bg-white",
  black: "bg-black",
};


interface NoteProps {
  noteNum: number,
  color: string,
  style?: React.CSSProperties;
  
}


const Note: React.FC<NoteProps> = ({ noteNum, color, style }) => {
  const [pressed, setPressed] = useState(false);

  return (
    <div
      // Actually apply the style prop here
      style={style}
      onMouseDown={() => setPressed(true)}
      onMouseUp={() => setPressed(false)}
      className={`
        flex items-end justify-center border border-black h-[100px] w-[20px] z-10 
        ${pressed ? "bg-amber-200" : tailwindColors[color]}
        ${color === "white"? "h-[100px] w-[20px]": "h-[70px] w-[15px]"}`}
    >
      <span className={`font-bold text-[6px] text-black ${color === "white"? "text-black": "text-white"}` }>
        {pianoNoteMap[noteNum]}
      </span>
    </div>
  );
};

// export default Note;


export default Note;