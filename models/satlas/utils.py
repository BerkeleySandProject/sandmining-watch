import os
import requests


def adjust_state_dict_prefix(
    state_dict, needed, prefix=None, prefix_allowed_count=None
):
    """
    Adjusts the keys in the state dictionary by replacing 'backbone.backbone' prefix with 'backbone'.

    Args:
        state_dict (dict): Original state dictionary with 'backbone.backbone' prefixes.

    Returns:
        dict: Modified state dictionary with corrected prefixes.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        # Assure we're only keeping keys that we need for the current model component.
        if not needed in key:
            continue

        # Update the key prefixes to match what the model expects.
        if prefix is not None:
            while key.count(prefix) > prefix_allowed_count:
                key = key.replace(prefix, "", 1)

        new_state_dict[key] = value
    return new_state_dict


def swin_feature_extractor(model):
    def forward(x):
        outputs = []
        for layer in model.features:
            x = layer(x)
            outputs.append(x.permute(0, 3, 1, 2))
        return [outputs[-7], outputs[-5], outputs[-3], outputs[-1]]

    return lambda x: forward(x)


def download_model_weights(url: str, save_dir: str):
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Extract filename from URL
    filename = url.split("/")[-1].split("?")[0]

    # Path to save the downloaded file
    save_path = os.path.join(save_dir, filename)

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Open the file in binary write mode and write the content
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"Model weights downloaded successfully to {save_path}")
    else:
        print(f"Failed to download model weights from {url}")
