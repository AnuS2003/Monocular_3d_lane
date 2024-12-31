import React from "react";

const Preview = ({ file }) => {
  if (!file) return null;

  const fileType = file.type.startsWith("video") ? "video" : "image";

  return (
    <div className="mt-4 border rounded-lg p-2">
      {fileType === "image" ? (
        <img
          src={URL.createObjectURL(file)}
          alt="Preview"
          className="max-w-full h-auto rounded-md"
        />
      ) : (
        <video controls className="max-w-full rounded-md">
          <source src={URL.createObjectURL(file)} type={file.type} />
          Your browser does not support the video tag.
        </video>
      )}
    </div>
  );
};

export default Preview;
