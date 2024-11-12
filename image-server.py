from flask import Flask, send_file
import os
import zipfile

app = Flask(__name__)


@app.route("/")
def index():
    # Define the directory containing images
    path = 'images'

    # Define the zip file path
    zip_path = 'images/images.zip'

    # Create a zip file containing all files in the images directory
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(path):
            for file in files:
                zipf.write(os.path.join(root, file), arcname=file)

    # Send the zip file to the client
    return send_file(zip_path, as_attachment=True)


if __name__ == "__main__":
    app.run()
