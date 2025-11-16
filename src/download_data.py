import requests
import os

def download_dataset():
    """
    Downloads the spam dataset and saves it to the dataset directory.
    """
    url = "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
    dataset_dir = "dataset"
    dataset_path = os.path.join(dataset_dir, "sms_spam.csv")

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(dataset_path, "wb") as f:
            f.write(response.content)
        print(f"Dataset downloaded successfully and saved to {dataset_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download_dataset()
