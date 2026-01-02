import argparse
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from prompts import system_prompt

from call_function import available_functions

def main():
    # Create a command-line argument parser and define a required positional argument for the user prompt
    parser = argparse.ArgumentParser(description="AI Code Assistant")
    parser.add_argument(
        "user_prompt",
        type=str,
        help="Prompt to send to Gemini"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output")
    args = parser.parse_args() # Parse command-line arguments

    # Load environment variables from a .env file (if present)
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY") # Retrieve the Gemini API key from environment variables
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set") # Fail early if the API key is missing

    # Initialize the Gemini API client
    client = genai.Client(api_key=api_key)

    # List of messages in the conversation
    messages = [types.Content(role="user", parts=[types.Part(text=args.user_prompt)])]

    # Send the user's prompt to the Gemini model and generate a response
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=messages,
        config=types.GenerateContentConfig(tools=[available_functions], system_instruction=system_prompt)
    )

    # Verbose output
    if args.verbose:
        print(f"User prompt: {args.user_prompt}")
        print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
        print(f"Response tokens: {response.usage_metadata.candidates_token_count}")

    # Output: function calls OR text
    function_calls = getattr(response, "function_calls", None)

    if function_calls:  # function_calls is a list when present, otherwise None/empty
        print("Response (function calls):")
        for function_call in function_calls:
            print(f"Calling function: {function_call.name}({function_call.args})")
    else:
        print("Response:")
        print(response.text)

if __name__ == "__main__":
# Only run main() when this file is executed directly, not when it is imported as a module
    main()