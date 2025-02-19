import React, {useState} from 'react';

const UploadButton: React.FC = () => {
  const [isClicked, setIsClicked] = useState(false);
  const [file, setFile] = useState<File | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) {
      const selectedFile = e.target.files[0];
      if (selectedFile.type === "audio/wav" || selectedFile.type === "audio/mp3" || selectedFile.name.endsWith(".mp3") || selectedFile.name.endsWith(".wav")) {
        setFile(selectedFile);
      } else {
        alert("Please upload a valid .wav or .mp3 file.");
      }
    }
  };

  const handleUpload = () => {
    setIsClicked(true);
    setTimeout(() => {
      setIsClicked(false);
    }, 200);
  };


  return (
    <button
      onClick={() => handleUpload()}
      className={`h-[100px] text-[24px] font-extrabold rounded-md transition-colors duration-300 
        ${isClicked ? "bg-green-500" : "bg-blue-950"}
      `}
    >
      Upload a new file here (.wav, mp3, etc)
    </button>
  )
}

export default UploadButton;