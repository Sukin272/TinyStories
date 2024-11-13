import os
# from together import Together
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

model = None

def load_model():
    global model
    genai.configure(api_key="AIzaSyC73mt4Sv6F26cc-_AJDzmWpF0V02dzG8U")
    model=genai.GenerativeModel("gemini-1.5-flash")
    # model = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    return model


# def generate_response(messages):
#     load_model()

#     completion = model.chat.completions.create(
#         model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
#         messages=messages,
#     )

#     return completion.choices[0].message.content


def evaluate_story(story):
    model=load_model()

    res1 = model.generate_content(f"In the following exercise, the student is given a beginning of a story. The student needs to complete it into a full story. The exercise tests the student's language abilities and creativity. The symbol *** marks the separator between the prescribed beginning and the student's completion. Evaluate the student's completion of the story (following the *** separator), focusing on their language abilities and creativity. Provide a concise, holistic evaluation on their writing, without discussing the story's content in at most 1 paragraph. Do not generate a sample completion or provide feedback at this time. It is ok if the story is incomplete. Evaluate the story written so far.\n\n{story}")
    # print(holistic_feedback.text)
    try:
        if res1 and res1.candidates:
                candidate = res1.candidates[0]  # Take the first candidate response
                safety_ratings = candidate.safety_ratings

                # Verify if content is blocked
                if safety_ratings and any(rating.blocked for rating in safety_ratings):
                    print("Response blocked due to safety filters.")
                    holistic_feedback = ""
                else:
                    holistic_feedback = res1.text
    except:
        holistic_feedback = ""

    print("*********************")
    print("holistic_feedback: ", holistic_feedback)
    response = model.generate_content(f"{story}\n\nHolistic Evaluation:{holistic_feedback}\n\nNow, grade the student's completion in terms of grammar, creativity, consistency with the story's beginning, and whether the plot makes sense.\n\nPlease provide the grading in JSON format with the keys 'grammar', 'creativity', 'consistency', 'plot_sense', as a number from 1 to 10. DO NOT OUTPUT ANYTHING BUT THE JSON.")
    return response.text


if __name__ == "__main__":
    story = "Once upon a time, *** there might was a land far from here, the Popel of this land whwe ghossts who were scary."
    # story = "Once upon a time, in an ancient house, there lived a girl named Lily. She loved to decorate her room with pretty things. One day, she found a big box in the attic. She opened it and saw many shiny decorations. Lily was very happy and decided to use them in her room.\nAs Lily was decorating her room, the sky outside became dark. There was a loud *** clap of thunder that startled her, causing her to drop one of the shiny decorations. To her amazement, instead of breaking, it began to glow softly. Suddenly, all the decorations she had placed started to shimmer and float into the air. The room transformed into a magical wonderland filled with dancing lights and gentle melodies. Lily watched in awe as the decorations painted stories of distant lands on her walls, filling her heart with wonder. That night, she realized that the old house held secrets beyond her imagination, and she felt grateful to be a part of its enchanting world."
#     story = """Once upon a time, in an ancient house, there lived a girl named Lily. She loved to decorate her room with pretty things. One day, she found a big box in the attic. She opened it and saw many shiny decorations. Lily was very happy and decided to use them in her room.
    
# As Lily was decorating her room, the sky outside became dark. There was a loud *** noise. Lily was scared and wanted to help her mom. She asked her mom if she could help her. 

# Her mom said yes and they went to the store. Lily was so happy to have a new friend. She had a new friend and she was happy to have a new friend."""
    response = evaluate_story(story)
    print(response)
