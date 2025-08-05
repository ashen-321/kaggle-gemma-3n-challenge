try:
    import os
    import asyncio
    from openai import OpenAI
    from base64 import b64encode
except ImportError:
    raise ImportError('Error importing modules. Ensure all packages from ../requirements.txt are installed. Run `pip '
          'install -r requirements.txt` in the terminal to install the packages.')

# Create ./input-files if it does not exist
input_relative_path = './input-files'
input_abspath = os.path.normpath(
    os.path.join(os.path.dirname(__file__), input_relative_path)
)
if not os.path.exists(input_abspath):
    print(f'Input files directory at {input_abspath} does not exist, creating new directory.')
    os.mkdir(input_abspath)


# Model ID
MODEL_ID = "google/gemma-3n-E4B-it"
FINETUNED_MODEL_ID = "alfredcs/gemma-3N-finetune"

# Create vLLM client
OPENAI_API_KEY = "EMPTY"

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="http://video.cavatar.info:8083/v1",
)

finetuned_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="http://video.cavatar.info:8087/v1",
)


# See if file is an image or audio based on the extension
def get_file_info(file_abspath: str):
    # Get file as bytes
    with open(file_abspath, "rb") as file:
        encoded_bytes = b64encode(file.read()).decode('utf-8')

    # Get file extension
    _, extension = os.path.splitext(file_abspath)
    extension = extension[1:]

    match extension:
        case "jpg" | "jpeg" | "png" | "webp":
            return "image_url", {"url": f"data:image/jpeg;base64,{encoded_bytes}"}
        case "mp3" | "wav":
            return "input_audio", {"data": encoded_bytes, "format": extension}
        case _:
            return "UNKNOWN", None


# Process a query
async def process_query(query: str):
    global messages

    # Add relevant files to message content
    message_content = []
    files = []
    query_with_files = False
    for entry in os.listdir(input_abspath):
        entry_abspath = os.path.join(input_abspath, entry)

        # Continue if entry is a directory
        if not os.path.isfile(entry_abspath):
            continue

        # Continue if file is marked with a #
        if entry.startswith("#"):
            continue

        # Add message content based on file type
        file_type, file_contents = get_file_info(entry_abspath)

        # Abort for unknown types
        if file_type == "UNKNOWN":
            print(f'\nError: Attempted to load file {entry} but could not determine its file type. Continuing...\n')
            continue

        query_with_files = True
        files.append(entry)
        message_content.append({"type": file_type, file_type: file_contents})

    # Add query to message content
    message_content.append({"type": "text", "text": query})

    # Add all content to message memory
    messages.append({"role": "user", "content": message_content})

    # Call model with messages
    # Fine-tuned does not support images or audio so use standard Gemma-3N otherwise
    if query_with_files:
        print(f'Querying with files {", ".join(files)}')
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
        )
    else:
        response = finetuned_client.chat.completions.create(
            model=FINETUNED_MODEL_ID,
            messages=messages,
        )
    response = response.choices[0].message.content
    print("Chat response:", response)

    # Save model output to message memory
    messages.append({"role": "assistant", "content": response})
    return response


# Main chat loop
messages = []
async def main():
    global messages

    while True:
        # try:
        query = input("\nQuery: ").strip()

        # Exit the program if the user enters "quit"
        if query.lower() == "quit":
            break

        # Wipe all message memory if the user enters "wipe"
        if query.lower() == "wipe":
            messages = []
            continue

        response = await process_query(query)
        print("\n" + response)

        # except Exception as e:
        #     print(f"\n{type(e).__name__}: {str(e)}")


if __name__ == '__main__':
    asyncio.run(main())
