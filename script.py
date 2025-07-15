import praw
from openai import OpenAI
import re
from tqdm import tqdm

REDDIT_CLIENT_ID = 'id'
REDDIT_CLIENT_SECRET = 'secret'
USER_AGENT = "user agent"
OPENAI_API_KEY = 'openai key'  

client = OpenAI(api_key=OPENAI_API_KEY)

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=USER_AGENT
)

def extract_username(profile_url):
    match = re.search(r"reddit.com\/user\/([A-Za-z0-9_\-]+)", profile_url)
    return match.group(1) if match else None


def fetch_user_content(username, limit=100):
    user = reddit.redditor(username)
    comments = []
    posts = []

    for comment in tqdm(user.comments.new(limit=limit), desc="Fetching comments"):
        comments.append(f"[{comment.subreddit}]: {comment.body}")
    for submission in tqdm(user.submissions.new(limit=limit), desc="Fetching posts"):
        posts.append(f"[{submission.subreddit}]: {submission.title}\n{submission.selftext}")

    return comments, posts

def build_prompt(username, comments, posts):
    combined_text = "\n".join(comments + posts)[:12000]
    return f"""
You are an expert behavioral analyst.

Analyze the following Reddit posts and comments by the user u/{username}.
Generate a structured user persona in the following format:

1. Username:
2. Traits (3-5 keywords):
3. Motivations:
4. Behavior & Habits:
5. Frustrations:
6. Goals & Needs:
7. Personality (MBTI style estimate):
8. Citations (include short comment/post snippets that support the conclusions):

Text:
{combined_text}
"""

def analyze_with_openai(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()

def save_output(username, content):
    filename = f"{username}_persona.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\nPersona saved to {filename}")

def main():
    profile_url = input("Enter Reddit profile URL: ").strip()
    username = extract_username(profile_url)

    if not username:
        print("Invalid Reddit URL.")
        return

    print(f"\nExtracting content for u/{username}...\n")
    comments, posts = fetch_user_content(username)

    print("\nGenerating user persona with OpenAI...\n")
    prompt = build_prompt(username, comments, posts)
    persona = analyze_with_openai(prompt)

    save_output(username, persona)

if __name__ == "__main__":
    main()
