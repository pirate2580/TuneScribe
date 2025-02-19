import React from 'react';

const Card: React.FC = () => {
  return(
    <>
      <div className="bg-slate-800 absolute h-[650px] top-[160px] left-[200px] max-w-[380px] translate-x-[-50%] text-white text-[16px] font-bold space-y-2 rounded-2xl shadow-lg p-3 flex flex-col justify-between">
        <p className="text-left mb-4">
          <p className="text-left mb-4">
            Have you ever just heard an instrumental song and wanted to learn it on piano? <br/><br/>
            Frustrated by the lack of available transcriptions?<br/><br/>
            <span className="text-cyan-400">MIDI.AI</span> is a free audio-to-MIDI transcriber based on the &nbsp; 
            <a 
              href="https://arxiv.org/abs/1710.11153" 
              target="_blank" 
              rel="noopener noreferrer"
              className="italic text-cyan-300 hover:underline"
            >
              Onsets and Frames
            </a> paper.
          </p>

        </p>

         <p className="text-left text-xl font-bold mb-3">Steps to Use:</p>
         <ol className="text-left list-decimal list-inside space-y-1 text-lg">
           <li className="flex items-center gap-2">
             ✅ <span>Upload your audio file.</span>
           </li>
           <li className="flex items-center gap-2">
             ⚡ <span>Let the AI process and generate a MIDI visualization.</span>
           </li>
           <li className="flex items-center gap-2">
             🎼 <span>Preview playback and download the transcribed PDF.</span>
           </li>
         </ol>

         
         <button className=" bg-blue-950 rounded-md ">
          Drop or Upload a new file here (.wav, mp3, etc)
          </button>
      </div>
    </>
  )
}

export default Card;