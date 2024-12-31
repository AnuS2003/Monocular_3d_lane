import React, { useState } from "react";
import FileUpload from "./components/FileUpload";
import Preview from "./components/Preview";

const App = () => {
  const [file, setFile] = useState(null);
  const [imageUrl, setImageUrl] = useState(null); // To hold the result image URL
  const [isLoading, setIsLoading] = useState(false); // To handle loading state
  const [isModalOpen, setIsModalOpen] = useState(false); // Modal visibility state

  const handleFileSelect = (selectedFile) => {
    setFile(selectedFile);
  };

  const handleSubmit = async () => {
    if (!file) {
      alert("Please upload a file before submitting!");
      return;
    }

    setIsLoading(true); // Start loading

    try {
      // Create a FormData object to send the file
      const formData = new FormData();
      formData.append("file", file);

      // Send the data to the backend API
      const response = await fetch("http://127.0.0.1:8000/visualize-3d-lane/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to get a response from the server");
      }

      const data = await response.json();
      setImageUrl(data.result); // Set the result image from the backend
      setIsModalOpen(true); // Open the modal when the result is available
    } catch (error) {
      console.error(error);
      alert("Error submitting the file. Please try again.");
    } finally {
      setIsLoading(false); // Stop loading
    }
  };

  // Close modal
  const closeModal = () => setIsModalOpen(false);

  return (
    <div className="min-h-screen bg-gray-100 flex justify-center items-center p-4">
      <div className="w-full max-w-xl bg-white rounded-lg shadow-lg p-6">
        <h1 className="text-3xl font-semibold text-gray-800 mb-6 text-center">
          Monocular 3D Lane Detection
        </h1>

        {/* File Upload Component */}
        <FileUpload onFileSelect={handleFileSelect} />

        {/* Preview Component */}
        <Preview file={file} />

        {/* Submit Button */}
        <div className="mt-6">
          <button
            onClick={handleSubmit}
            className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-500 transition"
          >
            {isLoading ? (
              <div className="flex justify-center items-center">
                <div className="w-5 h-5 border-4 border-t-transparent border-blue-500 border-solid rounded-full animate-spin" />
                <span className="ml-2">Processing...</span>
              </div>
            ) : (
              "Submit"
            )}
          </button>
        </div>
      </div>

      {/* Modal to display the processed result */}
      {isModalOpen && imageUrl && (
        <div className="fixed inset-0 bg-gray-900 bg-opacity-75 flex justify-center items-center z-50">
          <div className="bg-white rounded-lg shadow-lg p-6 max-w-6xl w-full">
            <button
              onClick={closeModal}
              className="absolute top-4 right-4 text-white font-semibold text-2xl"
            >
              &times;
            </button>
            <h3 className="text-xl font-bold text-gray-800 mb-4 text-center">
              Processed 3D Lane Detection Result
            </h3>
            <img
              src={`data:image/png;base64,${imageUrl}`}
              alt="Processed 3D Lane Result"
              className="w-full h-auto rounded-lg"
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default App;