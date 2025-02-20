import React, { createContext, useState, useContext, ReactNode } from "react";

interface MidiContextType {
  midiArray: number[][] | null;
  setMidiArray: (midi: number[][] | null) => void;
}

// Create the context with a default value
const MidiContext = createContext<MidiContextType | undefined>(undefined);

// Provider component
export const MidiProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [midiArray, setMidiArray] = useState<number[][] | null>(null);

  return (
    <MidiContext.Provider value={{ midiArray, setMidiArray }}>
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
