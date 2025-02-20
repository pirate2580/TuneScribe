import React, { useEffect, useState, useRef } from 'react';
import { useMidi } from "./MidiContext";
const SeekBar: React.FC = () => {
  // const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);

  const {playContext, setPlayContext, midiArray, currentIndex, setCurrentIndex, totalLength} = useMidi();
  // console.log(playContext);
  const currentIndexRef = useRef(currentIndex);

  useEffect(() => {
    currentIndexRef.current = currentIndex;
  }, [currentIndex]);


  const handlePlayPause = () => {
    if (!midiArray){
      return;
    }
    // setIsPlaying(!isPlaying);
    // console.log("either play or stop");
    setPlayContext(!playContext);
    console.log(playContext);
  };

  useEffect(() => {
    if (!playContext || totalLength === 0) return;
    console.log("testing seek");
    const interval = setInterval(() => {
      // Always use the latest currentIndex from the ref
      setProgress((currentIndexRef.current / totalLength));
    }, 232);
  
    return () => clearInterval(interval);
  }, [playContext, totalLength, setProgress]);

  // When the user clicks on the progress bar, update the "progress" state
  const handleProgressClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!midiArray){
      return;
    }
    const bar = event.currentTarget;
    const rect = bar.getBoundingClientRect();
    // Calculate where the user clicked relative to the bar width
    const clickX = event.clientX - rect.left;
    const newProgress = (clickX / rect.width) * 100;
    setProgress(newProgress);

    setProgress(newProgress);
    setCurrentIndex(Math.round((newProgress / 100) * totalLength));
  };

  return (
    <div className="absolute top-[770px] left-[400px] h-[40px] w-[1040px] bg-gray-800 p-4 flex items-center gap-4">
      {/* Play/Pause Button */}
      <button
        onClick={handlePlayPause}
        className="py-0.5 w-[60px] bg-red-500 text-white rounded-md"
      >
        {playContext ? 'Pause' : 'Play'}
      </button>
      
      {/* Progress Bar Container */}
      <div
        className="relative h-2 bg-gray-400 w-full cursor-pointer"
        onClick={handleProgressClick}
      >
        {/* Filled portion of the bar */}
        <div
          className="absolute top-0 left-0 h-2 bg-red-500"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Display current "progress" for reference */}
      <span className="text-white">{Math.round(progress)}%</span>
    </div>
  );
};

export default SeekBar;