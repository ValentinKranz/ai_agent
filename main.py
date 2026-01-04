import argparse
import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types

from prompts import system_prompt
from call_function import available_functions, call_function


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
        help="Enable verbose output"
    )
    args = parser.parse_args()  # Parse command-line arguments

    # Load environment variables from a .env file (if present)
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")  # Retrieve the Gemini API key from environment variables
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")  # Fail early if the API key is missing

    # Initialize the Gemini API client
    client = genai.Client(api_key=api_key)

    # List of messages in the conversation (conversation history)
    messages = [
        types.Content(role="user", parts=[types.Part(text=args.user_prompt)])
    ]

    final_response_text = None

    # Wrap the entire model-calling logic in a loop so the agent can iterate
    for _ in range(20):
        # Send the conversation history to the Gemini model and generate a response
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=messages,
            config=types.GenerateContentConfig(
                tools=[available_functions],
                system_instruction=system_prompt
            )
        )

        # Add all model candidates to the conversation history
        # This ensures the model can see its own previous outputs
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            if candidate is not None and getattr(candidate, "content", None) is not None:
                messages.append(candidate.content)

        # Verbose output
        if args.verbose:
            print(f"User prompt: {args.user_prompt}")
            print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
            print(f"Response tokens: {response.usage_metadata.candidates_token_count}")

        # Check whether the model requested any function calls
        function_calls = getattr(response, "function_calls", None)

        # If there are no function calls, the model has produced a final response
        if not function_calls:
            final_response_text = response.text
            break

        # Otherwise, execute each function call and collect the results
        function_responses = []

        for function_call_obj in function_calls:
            # Call the function selected by the model
            function_call_result = call_function(function_call_obj, verbose=args.verbose)

            # Validate that the function returned a proper tool response
            if not function_call_result.parts:
                raise RuntimeError("Function call result had no parts")

            function_response = function_call_result.parts[0].function_response
            if function_response is None:
                raise RuntimeError("Function call result had no function_response")

            tool_response = function_response.response
            if tool_response is None:
                raise RuntimeError("Function call result had no response")

            # Collect the Part so it can be passed back to the model
            function_responses.append(function_call_result.parts[0])

            # Print the tool response in verbose mode
            if args.verbose:
                print(f"-> {tool_response}")

        # Append function results so the model can see them in the next iteration
        messages.append(types.Content(role="user", parts=function_responses))

    # If we exited the loop with a final response, print it
    if final_response_text is not None:
        print("Final response:")
        print(final_response_text)
        return

    # If we hit the iteration limit without a final answer, fail explicitly
    print("Error: Reached maximum iterations without a final response.")
    sys.exit(1)


if __name__ == "__main__":
    # Only run main() when this file is executed directly, not when it is imported as a module
    main()