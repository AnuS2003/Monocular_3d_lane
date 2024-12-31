import React, { useState } from "react";

const FileUpload = ({ onFileSelect }) => {
  const [file, setFile] = useState(null);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);
    onFileSelect(selectedFile);
  };

  return (
    <div className="p-4 border-2 border-dashed border-gray-300 rounded-lg flex flex-col items-center">
      <label
        htmlFor="file-upload"
        className="cursor-pointer text-blue-600 font-medium hover:underline"
      >
        Click to upload an file
      </label>
      <input
        id="file-upload"
        type="file"
        accept="image/*,video/*"
        className="hidden"
        onChange={handleFileChange}
      />
      {file && (
        <p className="mt-2 text-sm text-gray-500">
          {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
        </p>
      )}
    </div>
  );
};

export default FileUpload;
