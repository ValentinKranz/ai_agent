import argparse
import os

from dotenv import load_dotenv
from google import genai

def main():
    # Create a command-line argument parser and define a required positional argument for the user prompt
    parser = argparse.ArgumentParser(description="AI Code Assistant")
    parser.add_argument(
        "user_prompt",
        type=str,
        help="Prompt to send to Gemini"
    )
    args = parser.parse_args() # Parse command-line arguments

    # Load environment variables from a .env file (if present)
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY") # Retrieve the Gemini API key from environment variables
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set") # Fail early if the API key is missing

    # Initialize the Gemini API client
    client = genai.Client(api_key=api_key)

    # Send the user's prompt to the Gemini model and generate a response
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=args.user_prompt,
    )

    # Validate that the response contains usage metadata
    if not response.usage_metadata:
        raise RuntimeError("Gemini API response appears to be malformed")

    # Print token usage statistics and the generated response text
    print("Prompt tokens:", response.usage_metadata.prompt_token_count)
    print("Response tokens:", response.usage_metadata.candidates_token_count)
    print("Response:")
    print(response.text)

if __name__ == "__main__":
# Only run main() when this file is executed directly, not when it is imported as a module
    main()