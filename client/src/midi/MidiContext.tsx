import React, { createContext, useState, useContext, ReactNode } from "react";

interface MidiContextType {
  midiArray: number[][] | null;
  setMidiArray: (midi: number[][] | null) => void;
  currentIndex: number;
  setCurrentIndex: React.Dispatch<React.SetStateAction<number>>;
  totalLength: number;
  setTotalLength: React.Dispatch<React.SetStateAction<number>>;
  playContext: boolean;
  setPlayContext: React.Dispatch<React.SetStateAction<boolean>>;
}

// Create the context with a default value
const MidiContext = createContext<MidiContextType | undefined>(undefined);

// Provider component
export const MidiProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [midiArray, setMidiArray] = useState<number[][] | null>(null);

  const [currentIndex, setCurrentIndex] = useState<number>(0);
  const [totalLength, setTotalLength] = useState<number>(0);
  const [playContext, setPlayContext] = useState(false);

  return (
    <MidiContext.Provider value={{ midiArray, setMidiArray, currentIndex, setCurrentIndex, totalLength, setTotalLength, playContext, setPlayContext }}>
      {children}
    </MidiContext.Provider>
  );
};

// midi context hook
export const useMidi = (): MidiContextType => {
  const context = useContext(MidiContext);
  if (!context) {
    throw new Error("useMidi must be used within a MidiProvider");
  }
  return context;
};
