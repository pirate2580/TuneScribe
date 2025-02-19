// 1) Define a helper function that converts 0..87 into "A0".."C8".
function noteNameFromIndex(index: number): string {
  // Standard MIDI note numbers: 
  // A0 = 21, A#0/Bb0 = 22, B0 = 23, C1 = 24, ..., C8 = 108.
  const midiNumber = index + 21; // Because index=0 => A0 => MIDI 21
  
  // Cycle of note names (using flats for black keys).
  const NOTE_NAMES = [
    "C",  "Db", "D",  "Eb", "E",  "F",
    "Gb", "G",  "Ab", "A",  "Bb", "B"
  ];
  
  // Which of the 12 notes in the octave?
  const noteInOctave = midiNumber % 12; // 0..11
  
  // Octave calculation. 
  // For example: 
  //   MIDI #60 => 60 / 12 = 5 => minus 1 => octave 4 (C4)
  const octave = Math.floor(midiNumber / 12) - 1;
  
  // Compose the note name + octave, e.g. "C4", "Ab3", etc.
  return NOTE_NAMES[noteInOctave] + octave;
}

// 2) Build the array of length 88
//    Each index i => noteNameFromIndex(i).
export const pianoNoteMap = Array.from({ length: 88 }, (_, i) => 
  noteNameFromIndex(i)
);
